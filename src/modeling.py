import torch
import pandas as pd
from os.path import join
from typing import Optional
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Adafactor
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import ADDED_TOKENS, SPLIT, extract_fields_from_generation
from utils import get_device, to_device, load_last_model, load_best_model, save_model
from metrics import get_metrics


def initialize_model_and_tokenizer(
    conf: DictConfig,
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        conf.model_name, use_fast=False, model_max_length=512
    )

    # adding custom tokens to vocab
    new_tokens = set(ADDED_TOKENS) - set(tokenizer.get_vocab().keys())
    tokenizer.add_tokens(sorted(new_tokens))
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train(
    conf: DictConfig,
    model: AutoModelForSeq2SeqLM,
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

    if train_conf.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=train_conf.lr)
    elif train_conf.optimizer == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=train_conf.lr,
            scale_parameter=False,
            relative_step=False,
        )
    do_validation = dev_dl and train_conf.get("eval_every", -1) > 0

    with tqdm(total=train_conf.train_steps, desc="training..") as pbar:
        while step < train_conf.train_steps:
            try:
                batch = next(train_iter)
            except:
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
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    eval_mode: SPLIT,
    eval_df: pd.DataFrame,
    eval_dl: DataLoader,
    step: Optional[int] = -1,
    eval_prefix: Optional[str] = None,
):

    eval_config = conf.eval_config
    generate_config = conf.generate_config
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

    generated = []

    with torch.no_grad():
        for batch in tqdm(eval_dl, desc=f"running {eval_mode}..."):
            batch = to_device(batch, device)
            outputs = model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=128,
                **generate_config,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded)

    eval_df["generation"] = generated
    eval_df = pd.concat(
        (eval_df, eval_df["generation"].apply(extract_fields_from_generation)), axis=1
    )
    filename = f"dev-{step}-df.jsonl" if eval_mode == "dev" else "test-df.jsonl"
    eval_df.to_json(join(conf.output_dir, filename), lines=True, orient="records")

    eval_metrics = get_metrics(eval_df)
    filename = f"dev-{step}-metrics.json" if eval_mode == "dev" else "test-metrics.json"
    eval_metrics.to_json(join(conf.output_dir, filename))
