import sys, os
from os.path import join
import pandas as pd
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from openai_api_cache import Jurassic1APICache, OpenAIAPICache

from utils import seed_everything
from data import prepare_dataframes
from metrics import get_metrics

API = {
    "jurassic": Jurassic1APICache,
    "gpt-3": OpenAIAPICache,
}

instruction = "Classify the offensiveness of the given post, and decide whether it implies a stereotype targeting some identity-related group."

prompt_format = """post: {post}
offensive: {offensive}
group targeted: {group}
stereotype: {stereotype}
"""
test_prompt_format = """post: {post}
offensive:"""


def extract_fields(completion: str) -> pd.Series:
    fields = [f.strip() for f in completion.split("\n")]
    assert fields[0] in ("yes", "no")
    assert fields[1].startswith("group targeted: ")
    assert fields[2].startswith("stereotype: ")

    return pd.Series(
        {
            "offensivePrediction": 1 if fields[0] == "yes" else 0,
            "generatedMinorityGroup": fields[1][16:],
            "generatedStereotype": fields[2][12:],
        }
    )

def main():
    seed_everything(42)
    conf = OmegaConf.load(sys.argv[1])

    # dump config to output_dir
    os.makedirs(conf.output_dir, exist_ok=True)
    with open(join(conf.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    dataframes = prepare_dataframes(conf.data_config, splits=["train", "dev", "test"])

    # sample from training set
    train_samples_per_class = conf.get("train_samples", 8)
    train_seed = conf.get("train_seed", 42)
    test_samples = conf.get("test_samples", 100)
    test_seed = conf.get("test_seed", 42)

    train_df = dataframes["train"]
    train_items = [instruction]
    train_subset = pd.concat(
        (
            train_df[train_df["offensiveYN"] == 0.0]
            .sample(train_samples_per_class, random_state=train_seed)
            .fillna("None"),
            train_df[train_df["offensiveYN"] == 1.0]
            .sample(train_samples_per_class, random_state=train_seed)
            .fillna("None"),
        )
    ).sample(frac=1.0)
    test_subset = dataframes["test"].sample(test_samples, random_state=test_seed)

    for _, row in train_subset.iterrows():
        train_items.append(
            prompt_format.format(
                post=row["post"].replace("\n", " "),
                offensive="yes" if row["offensiveYN"] == 1.0 else "no",
                group=row["targetMinority"],
                stereotype=row["targetStereotype"],
            )
        )

    train_prompt = "\n\n".join(train_items)

    if conf.get("dry_run"):
        print("### BEGIN PROMPT ###")
        print(train_prompt)
        print("### END PROMPT ###")
        exit(0)

    with open(join(conf.output_dir, "train-prompt.txt"), "w") as f:
        f.write(train_prompt)

    api = API[conf["class"]](open(conf.api_key_file).read().strip())

    if conf["class"] == "gpt-3":
        generation_kwargs = {
            "max_tokens": 50,
            "stop": ("\n\n", "post:"),
        }
    elif conf["class"] == "jurassic":
        generation_kwargs = {
            "maxTokens": 50,
            "stopSequences": ("\n\n", "post:"),
        }
    else:
        raise Exception

    test_completions = []
    for _, row in tqdm(test_subset.iterrows()):
        test_item = test_prompt_format.format(post=row["post"])
        prompt = "\n\n".join(train_items + [test_item])
        resp = api.generate(
            model=conf.generate_config.model,
            prompt=prompt,
            temperature=conf.generate_config.get("temperature", 0.0),
            **generation_kwargs
        )
        
        if conf["class"] == "gpt-3":
            completion = resp["choices"][0]["text"]
        elif conf["class"] == "jurassic":
            completion = resp["completions"][0]["data"]["text"]
        else:
            raise Exception
        
        test_completions.append(completion)

    test_subset["generation"] = test_completions
    test_subset_generated = pd.concat(
        (test_subset, test_subset["generation"].apply(extract_fields)), axis="columns"
    )
    test_subset_generated.to_json(
        join(conf.output_dir, "test-df.jsonl"), lines=True, orient="records"
    )
    test_metrics = get_metrics(test_subset_generated)
    test_metrics.to_json(join(conf.output_dir, "test-metrics.json"))


if __name__ == "__main__":
    main()
