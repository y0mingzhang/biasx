import functools, re
from typing import Any, Literal
import pandas as pd
from os.path import join
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset, DatasetDict
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader
from utils import num_workers

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
CONTROL_TOKENS = [OFFENSIVE_TOKEN, NON_OFFENSIVE_TOKEN, GROUP_TOKEN, STEREOTYPE_TOKEN]
SPLIT = Literal["train", "dev", "test"]


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["offensiveYN"].isin([0.0, 1.0])]


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
) -> tuple[dict[SPLIT, pd.DataFrame], dict[SPLIT, DataLoader]]:
    disable_progress_bar()  # datasets progbars kind of annoying
    splits = ["train", "dev", "test"]
    dataframes = {}

    # for each post, get all reference groups/stereotypes for BLEU/ROUGE/WMD evaluation
    for split in splits:
        df = filter_dataframe(
            pd.read_csv(
                join("/home/yimingz0/src/sbf-modeling/data", "SBIC.v2", f"{split}.csv")
            )
        ).drop_duplicates(subset=["post", "targetMinority", "targetStereotype"])

        reference_minority_groups = (
            df[(df["offensiveYN"] == 1.0) & (df["targetMinority"].notna())]
            .groupby("post")
            .agg(referenceMinorityGroups=("targetMinority", "unique"))
        )

        reference_stereotypes = (
            df[(df["offensiveYN"] == 1.0) & (df["targetStereotype"].notna())]
            .groupby("post")
            .agg(referenceStereotypes=("targetStereotype", "unique"))
        )

        df = df.merge(reference_minority_groups, how="left", on="post").merge(
            reference_stereotypes, how="left", on="post"
        )

        isna = df["referenceMinorityGroups"].isna()
        df.loc[isna, "referenceMinorityGroups"] = pd.Series(
            [[NA_TOKEN]] * isna.sum()
        ).values

        isna = df["referenceStereotypes"].isna()
        df.loc[isna, "referenceStereotypes"] = pd.Series(
            [[NA_TOKEN]] * isna.sum()
        ).values

        dataframes[split] = df

    dataset_raw = DatasetDict(
        {split: Dataset.from_pandas(dataframes[split]) for split in splits}
    )

    tokenize_example = functools.partial(tokenize_func, tokenizer)
    dataset = dataset_raw.map(
        process_example, num_proc=num_workers(), desc="processing.."
    ).map(tokenize_example, num_proc=num_workers(), desc="tokenizing..")
    dataset_torch = dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # this collator pads batches to the same lengths on the fly
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    dataloaders = {
        split: DataLoader(
            dataset_torch[split],
            shuffle=(split == "train"),
            batch_size=data_conf.batch_size,
            collate_fn=collator,
            pin_memory=True,
            num_workers=num_workers(),
        )
        for split in dataset_torch.keys()
    }

    return dataframes, dataloaders


def extract_fields_from_generation(generation: str) -> dict[str, Any]:
    if generation.startswith(OFFENSIVE_TOKEN):
        try:
            # split on all control tokens, should have exactly 2 fields
            pattern = "|".join(map(lambda t: "\\" + t, CONTROL_TOKENS))
            splitted = re.split(pattern, generation)
            minority, stereotype = filter(len, map(str.strip, splitted))
            return pd.Series(
                {
                    "offensivePrediction": 1.0,
                    "generatedMinorityGroup": minority,
                    "generatedStereotype": stereotype,
                }
            )

        except:
            print("failed to parse", generation)
            print("default to negative prediction")
    return pd.Series(
        {
            "offensivePrediction": 0.0,
            "generatedMinorityGroup": NA_TOKEN,
            "generatedStereotype": NA_TOKEN,
        }
    )
