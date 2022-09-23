import functools
import pandas as pd
from os.path import join
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader

# globals
OFFENSIVE_TOKEN = "[OffY]"
NON_OFFENSIVE_TOKEN = "[OffN]"
GROUP_TOKEN = "[grp]"
STEREOTYPE_TOKEN = "[ste]"
NA_TOKEN = "None"
ADDED_TOKENS = [
    OFFENSIVE_TOKEN,
    NON_OFFENSIVE_TOKEN,
    GROUP_TOKEN,
    STEREOTYPE_TOKEN,
    NA_TOKEN,
]


def keep_example(example: dict) -> bool:
    # drop examples with ambiguous labels
    return example["offensiveYN"] in (0.0, 1.0)


def process_example(example: dict) -> dict:
    # getting generation target for examples
    offensive = example["offensiveYN"] == 1.0
    if offensive:
        target = " ".join(
            [
                OFFENSIVE_TOKEN,
                GROUP_TOKEN,
                NA_TOKEN
                if example["targetMinority"] is None
                else example["targetMinority"],
                STEREOTYPE_TOKEN,
                NA_TOKEN
                if example["targetStereotype"] is None
                else example["targetStereotype"],
            ]
        )
    else:
        target = NON_OFFENSIVE_TOKEN
    return {"text": example["post"], "target": target}


def tokenize_func(tokenizer: AutoTokenizer, example: dict) -> dict:
    tokenized = tokenizer(example["text"])
    tokenized["labels"] = tokenizer(example["target"]).input_ids
    return tokenized


def prepare_data(
    data_conf: DictConfig, tokenizer: AutoTokenizer
) -> tuple[DatasetDict, DataLoader]:

    set_verbosity_error()  # datasets progbars kind of annoying
    splits = ["train", "dev", "test"]
    dataframes = {
        split: pd.read_csv(
            join(data_conf.data_dir, "SBIC.v2", f"{split}.csv")
        ).drop_duplicates(subset=["post", "targetMinority", "targetStereotype"])
        for split in splits
    }
    dataset_raw = DatasetDict(
        {split: Dataset.from_pandas(dataframes[split]) for split in splits}
    )

    tokenize_example = functools.partial(tokenize_func, tokenizer)
    dataset = (
        dataset_raw.filter(keep_example, num_proc=16, desc="filtering data..")
        .map(process_example, num_proc=16, desc="processing..")
        .map(tokenize_example, num_proc=16, desc="tokenizing..")
    )
    dataset_torch = dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # this collator pads batches to the same length on the fly
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    dataloaders = {
        split: DataLoader(
            dataset_torch[split],
            shuffle=(split == "train"),
            batch_size=data_conf.batch_size,
            collate_fn=collator,
            pin_memory=True,
        )
        for split in dataset_torch.keys()
    }

    return dataset, dataloaders
