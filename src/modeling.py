from os.path import join
from typing import Optional

import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    NoBadWordsLogitsProcessor,
)

from data import (
    ADDED_TOKENS,
    NA_TOKEN,
    NON_OFFENSIVE_TOKEN,
    SPLIT,
    extract_fields_from_generation,
)
from metrics import get_entailment_metrics, get_metrics
from utils import get_device, load_best_model, load_last_model, save_model, to_device


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


def initialize_entailment_classifier_and_tokenizer(
    conf: DictConfig,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    model = AutoModelForSequenceClassification.from_pretrained(
        conf.model_name, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained(conf.model_name, use_fast=False)

    return model, tokenizer


def load_entail_model(
    entail_config: DictConfig,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    entail_dir = entail_config["output_dir"]
    conf = OmegaConf.load(join(entail_dir, "config.yaml"))
    model, tokenizer = initialize_entailment_classifier_and_tokenizer(conf)

    model = load_best_model(
        conf.output_dir, model, conf.eval_config.monitor, conf.eval_config.maximize
    )
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
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    eval_mode: SPLIT,
    eval_df: pd.DataFrame,
    eval_dl: DataLoader,
    step: Optional[int] = -1,
    eval_prefix: Optional[str] = None,
):

    mode = conf.mode
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

    rank_with_entailment = "entailment_config" in conf and eval_mode == "test"
    if rank_with_entailment:
        entail_model, entail_tokenizer = load_entail_model(conf["entailment_config"])
        candidate_banned = [
            tokenizer(NON_OFFENSIVE_TOKEN, add_special_tokens=False).input_ids,
            tokenizer(NA_TOKEN, add_special_tokens=False).input_ids,
        ]
        candidate_logits_proc = NoBadWordsLogitsProcessor(
            candidate_banned, tokenizer.eos_token_id
        )

    if not eval_prefix:
        eval_prefix = eval_mode

    if mode == "generation":
        generate_config = conf.generate_config
        generated = []
        with torch.no_grad():
            for batch in tqdm(eval_dl, desc=f"running {eval_prefix}..."):
                batch = to_device(batch, device)
                outputs = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_new_tokens=128,
                    **generate_config,
                )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                if rank_with_entailment:
                    extracted = extract_fields_from_generation(decoded[0])
                    if extracted["generatedStereotype"] != NA_TOKEN:
                        outputs = model.generate(
                            batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_new_tokens=128,
                            do_sample=True,
                            num_return_sequences=10,
                            logits_processor=[candidate_logits_proc],
                        )
                        candidates = decoded + tokenizer.batch_decode(
                            outputs, skip_special_tokens=True
                        )
                        candidate_stereotypes = [
                            extract_fields_from_generation(candidate)[
                                "generatedStereotype"
                            ]
                            for candidate in candidates
                        ]
                        candidate_batch = entail_tokenizer(
                            tokenizer.batch_decode(
                                batch["input_ids"], skip_special_tokens=True
                            )
                            * len(candidate_stereotypes),
                            candidate_stereotypes,
                            return_tensors="pt",
                            padding="longest",
                        )
                        candidate_outputs = entail_model(**candidate_batch)
                        candidate_goodness = candidate_outputs.logits.softmax(1)[:, 1]
                        decoded = [candidates[candidate_goodness.argmax().item()]]

                generated.extend(decoded)

        eval_df["generation"] = generated
        eval_df = pd.concat(
            (eval_df, eval_df["generation"].apply(extract_fields_from_generation)),
            axis=1,
        )
        eval_metrics = get_metrics(eval_df)

    elif mode == "entailment":
        all_preds = []
        with torch.no_grad():
            for batch in tqdm(eval_dl):
                batch = to_device(batch, device)
                output = model(**batch)
                preds = output.logits.argmax(dim=1)
                all_preds.extend(preds.tolist())
        eval_df["entailment_pred"] = all_preds
        eval_metrics = get_entailment_metrics(eval_df)
    else:
        raise Exception(f"unrecognized mode {mode}")

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
