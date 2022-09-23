import torch
from typing import Optional
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import Dataset
from data import ADDED_TOKENS
from utils import get_device, to_device, load_last_model, save_model


def initialize_model_and_tokenizer(
    conf: DictConfig,
) -> tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
    model = AutoModelForSeq2SeqLM.from_pretrained(conf.model_name)
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=False)

    # adding custom tokens to vocab
    new_tokens = set(ADDED_TOKENS) - set(tokenizer.get_vocab().keys())
    tokenizer.add_tokens(list(new_tokens))
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def train(
    conf: DictConfig,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    train_dl: DataLoader,
    train_data: Dataset,
    val_dl: DataLoader,
    val_data: Dataset,
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
    do_validation = val_dl and train_conf.get("eval_every", -1) > 0

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
                evaluate(conf, model, tokenizer, "val", val_dl, val_data, step=step)
                model.train()

            pbar.update(1)
    save_model(conf.output_dir, model, step)


def evaluate(
    conf: DictConfig,
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    eval_mode: str,
    eval_dl: DataLoader,
    eval_data: Dataset,
    step: Optional[int] = None,
    eval_prefix: Optional[str] = None,
):
    if eval_mode == "test":
        try:
            model = load_best_model(conf.output_dir, model)
        except Exception:
            print("cannot find best model, loading last saved model for testing")
            model, _ = load_last_model(conf.output_dir, model)

    model.eval()
    device = get_device()
    model.to(device)

    generated = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(eval_dl):
            batch = to_device(batch, device)
            outputs = model.generate(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated.extend(decoded)
