from os.path import join
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from data import (
    ADDED_TOKENS,
    NA_TOKEN,
    NON_OFFENSIVE_TOKEN,
    SPLIT,
    extract_fields_from_generation,
)
from metrics import get_classification_metrics
from utils import get_device, load_best_model, load_last_model, save_model, to_device


def initialize_model_and_tokenizer(
    conf: DictConfig,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(conf.model_name)
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name)

    return model, tokenizer

def train(
    conf: DictConfig,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_df: pd.DataFrame,
    train_dl: DataLoader,
    dev_df: pd.DataFrame,
    dev_dl: DataLoader,
) -> None:
    train_conf = conf.train_config
    device = get_device()
    step = 0
    grad_accumulation_steps = train_conf.get("grad_accumulation_steps", 1)
    train_iter = iter(train_dl)

    model, step = load_last_model(conf.output_dir, model)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=train_conf.lr)

    do_validation = dev_dl and train_conf.get("eval_every", -1) > 0

    with tqdm(total=train_conf.train_steps, desc="training..") as pbar:
        while step < train_conf.train_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                print("train_dataloader reset")
                train_iter = iter(train_dl)
                batch = next(train_iter)

            batch = to_device(batch, device)
            output = model(**batch)
            loss = output.loss / grad_accumulation_steps
            loss.backward()

            step += 1

            if step % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if do_validation and step % train_conf.eval_every == 0:
                evaluate(conf, model, tokenizer, "dev", dev_df, dev_dl, step=step)
                model.train()

            pbar.update(1)
            pbar.set_postfix(loss=output.loss.item())
    save_model(conf.output_dir, model, step)


def evaluate(
    conf: DictConfig,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    eval_mode: SPLIT,
    eval_df: pd.DataFrame,
    eval_dl: DataLoader,
    step: Optional[int] = -1,
    eval_prefix: Optional[str] = None,
):

    eval_config = conf.eval_config
    if eval_mode == "test":
        try:
            model = load_best_model(
                conf.output_dir, model, eval_config.monitor, eval_config.maximize
            )
        except Exception:
            print("loading last saved model for testing")
            model, _ = load_last_model(conf.output_dir, model)
    else:
        save_model(conf.output_dir, model, step)

    model.eval()
    device = get_device()
    model.to(device)

    if not eval_prefix:
        eval_prefix = eval_mode

    all_preds = []
    positive_probs = []

    with torch.no_grad():
        for batch in tqdm(eval_dl):
            batch = to_device(batch, device)
            output = model(**batch)
            preds = output.logits.argmax(dim=1)
            all_preds.extend(preds.tolist())
            positive_probs.extend(output.logits.softmax(1)[:, 1].tolist())
    
    eval_df["offensivePrediction"] = all_preds
    eval_df["offensiveProbability"] = positive_probs
    eval_metrics = get_classification_metrics(eval_df)


    filename = (
        f"dev-{step}-df.jsonl" if eval_mode == "dev" else f"{eval_prefix}-df.jsonl"
    )
    eval_df.to_json(join(conf.output_dir, filename), lines=True, orient="records")

    filename = (
        f"dev-{step}-metrics.json"
        if eval_mode == "dev"
        else f"{eval_prefix}-metrics.json"
    )
    eval_metrics.to_json(join(conf.output_dir, filename))
