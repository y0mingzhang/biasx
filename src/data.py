import functools, re, itertools, json, random
from collections import defaultdict
import numpy as np
from typing import Any, Literal, Union
import pandas as pd
from os.path import join
from omegaconf import DictConfig
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorWithPadding
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
SPLIT = Union[Literal["train", "dev", "test"], str]


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["offensiveYN"].isin([0.0, 1.0])]


def process_example(example: dict) -> dict:
    if "targetMinority" not in example:
        # eval example does not have targets
        return {"text": example["post"], "target": ""}

    # getting generation target for training examples
    offensive = example["offensiveYN"] == 1.0
    if offensive:
        target = " ".join(
            [
                OFFENSIVE_TOKEN,
                GROUP_TOKEN,
                NA_TOKEN
                if example.get("targetMinority") is None
                else example["targetMinority"],
                STEREOTYPE_TOKEN,
                NA_TOKEN
                if example.get("targetStereotype") is None
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


def tokenize_entailment_func(tokenizer: AutoTokenizer, example: dict) -> dict:
    tokenized = tokenizer(example["post"], example["targetStereotype"])
    tokenized["labels"] = example["entailment_label"]
    return tokenized


def aggregate_post_group(group: pd.Series) -> pd.Series:
    # is the post offensive under any group?
    offensive = any(group["offensiveYN"])
    ref_groups = []
    ref_stereotypes = []

    for _, row in group.iterrows():
        if row["offensiveYN"]:
            if isinstance(row["targetMinority"], str):
                ref_groups.append(row["targetMinority"])
            if isinstance(row["targetStereotype"], str):
                ref_stereotypes.append(row["targetStereotype"])

    if ref_groups:
        ref_groups = sorted(set(ref_groups))
    else:
        ref_groups = [NA_TOKEN]

    if ref_stereotypes:
        ref_stereotypes = sorted(set(ref_stereotypes))
    else:
        ref_stereotypes = [NA_TOKEN]

    return pd.Series(
        {
            "offensiveYN": offensive,
            "referenceMinorityGroups": ref_groups,
            "referenceStereotypes": ref_stereotypes,
        }
    )


def summarize_dataset(df: pd.DataFrame) -> dict:
    if "targetMinority" in df.columns:
        groups = set(df["targetMinority"])
        stereotypes = set(df["targetStereotype"])
    else:
        groups = set(itertools.chain(*df["referenceMinorityGroups"]))
        stereotypes = set(itertools.chain(*df["referenceStereotypes"]))
    return {
        "examples": len(df),
        "label distribution": df.value_counts("offensiveYN", normalize=True).to_dict(),
        "distinct groups": len(set(groups)),
        "distinct stereotypes": len(set(stereotypes)),
    }


def prepare_dataframes(
    data_conf: DictConfig, splits: SPLIT
) -> dict[SPLIT, pd.DataFrame]:
    dataframes = {}

    # for each post, get all reference groups/stereotypes for BLEU/ROUGE/WMD evaluation
    for split in splits:
        df = filter_dataframe(
            pd.read_csv(join(data_conf.data_dir, f"{split}.csv"))
        ).drop_duplicates(subset=["post", "targetMinority", "targetStereotype"])

        if split == "train":
            if data_conf.subsample_common_stereotypes:
                stereotype_counts = df.value_counts("targetStereotype")
                inclusion_probs = 1 / np.sqrt(
                    df["targetStereotype"].apply(
                        lambda s: stereotype_counts[s] if s in stereotype_counts else 1
                    )
                )
                included = np.random.uniform(size=len(df)) <= inclusion_probs
                df = df[included]

        else:
            # when evaluating, we "aggregate" the data, so that post becomes a unique
            # key, and generating any among ref groups/stereotypes is acceptable
            df = df.groupby("post").apply(aggregate_post_group).reset_index()

        if split == "dev" and data_conf.get("dev_size", -1) > 0:
            df = df.sample(data_conf.dev_size)

        dataframes[split] = df

        print(f"{split} dataset:")
        print(json.dumps(summarize_dataset(df), indent=4))

    return dataframes


def prepare_data(
    data_conf: DictConfig, tokenizer: AutoTokenizer
) -> tuple[dict[SPLIT, pd.DataFrame], dict[SPLIT, DataLoader]]:
    disable_progress_bar()  # datasets progbars kind of annoying
    splits = ["train", "dev", "test"] + list(data_conf.get("additional_test", []))
    dataframes = prepare_dataframes(data_conf, splits)
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


def prepare_dataframes_entailment_classifier(
    data_conf: DictConfig,
) -> dict[SPLIT, pd.DataFrame]:
    splits = ["train", "dev", "test"]
    dataframes = {}

    for split in splits:
        df = filter_dataframe(
            pd.read_csv(join(data_conf.data_dir, f"{split}.csv"))
        ).drop_duplicates(subset=["post", "targetMinority", "targetStereotype"])

        stereotype_counts = df.value_counts("targetStereotype")
        inclusion_probs = 1 / np.sqrt(
            df["targetStereotype"].apply(
                lambda s: stereotype_counts[s] if s in stereotype_counts else 1
            )
        )
        included = np.random.uniform(size=len(df)) <= inclusion_probs
        df = df[included]

        positive = df.dropna(subset=("targetMinority", "targetStereotype")).copy()
        positive["entailment_label"] = 1

        group_to_stereotype = defaultdict(list)
        for group, stereotype in zip(
            positive["targetMinority"], positive["targetStereotype"]
        ):
            group_to_stereotype[group].append(stereotype)

        groups = sorted(group_to_stereotype.keys())
        for g in group_to_stereotype:
            group_to_stereotype[g] = sorted(set(group_to_stereotype[g]))

        easy_negative = []
        hard_negative = []

        for _, row in positive.iterrows():

            group = row["targetMinority"]
            stereotype = row["targetStereotype"]
            other_group = random.choice(groups)
            other_group_stereotype = random.choice(group_to_stereotype[other_group])

            row_easy_neg = row.copy()
            row_easy_neg["targetMinority"] = other_group
            row_easy_neg["targetStereotype"] = other_group_stereotype
            row_easy_neg["entailment_label"] = 0
            easy_negative.append(row_easy_neg)

            if len(group_to_stereotype[group]) >= 5:
                row_hard_neg = row.copy()
                row_hard_neg["targetMinority"] = group
                group_other_stereotype = random.choice(group_to_stereotype[group])
                while group_other_stereotype == stereotype:
                    group_other_stereotype = random.choice(group_to_stereotype[group])
                row_hard_neg["targetStereotype"] = group_other_stereotype
                row_hard_neg["entailment_label"] = 0
                hard_negative.append(row_hard_neg)

        easy_negative = pd.DataFrame(easy_negative)
        hard_negative = pd.DataFrame(hard_negative)

        dataset_size = min(
            len(positive) * 2, len(easy_negative) * 4, len(hard_negative) * 4
        )
        df = pd.concat(
            (
                positive.sample(dataset_size // 4 * 2),
                easy_negative.sample(dataset_size // 4),
                hard_negative.sample(dataset_size // 4),
            ),
            ignore_index=True,
        )

        dataframes[split] = df

    return dataframes


def prepare_data_entailment_classifier(
    data_conf: DictConfig, tokenizer: AutoTokenizer
) -> tuple[dict[SPLIT, pd.DataFrame], dict[SPLIT, DataLoader]]:
    disable_progress_bar()  # datasets progbars kind of annoying
    splits = ["train", "dev", "test"]
    dataframes = prepare_dataframes_entailment_classifier(data_conf)
    dataset_raw = DatasetDict(
        {split: Dataset.from_pandas(dataframes[split]) for split in splits}
    )

    tokenize_example = functools.partial(tokenize_entailment_func, tokenizer)
    dataset = dataset_raw.map(
        tokenize_example, num_proc=num_workers(), desc="tokenizing.."
    )
    dataset_torch = dataset.with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # this collator pads batches to the same lengths on the fly
    collator = DataCollatorWithPadding(tokenizer, padding=True)

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
