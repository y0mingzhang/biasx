import os
import sys
from os.path import join
import warnings

import pandas as pd
from omegaconf import OmegaConf
from openai_api_cache import Jurassic1APICache, OpenAIAPICache
from tqdm.auto import tqdm
import torch

from data import NA_TOKEN, prepare_dataframes
from metrics import get_metrics
from modeling import load_entail_model
from utils import get_device, seed_everything, to_device

warnings.filterwarnings("ignore")

API = {
    "jurassic": Jurassic1APICache,
    "gpt-3": OpenAIAPICache,
}

instruction = (
    "Classify the offensiveness of the given statement"
    " and give a reason for the decision."
)


def fetch_gpt3_completion(resp: dict) -> str:
    return resp["choices"][0]["text"]


def fetch_jurassic_completion(resp: dict) -> str:
    return resp["completions"][0]["data"]["text"]


prompt_format = """statement: {}
offensive: {}
targeted group: {}
reason: {}
"""
test_prompt_format = """statement: {post}
offensive:"""


def extract_fields(completion: str, partial: bool = False) -> pd.Series:
    try:
        fields = [f.strip() for f in completion.split("\n")]
        assert fields[0] in ("yes", "no")
        if partial:
            return {"offensivePrediction": 1 if fields[0] == "yes" else 0}

        assert fields[1].startswith("targeted group: ")
        assert fields[2].startswith("reason: ")
        return pd.Series(
            {
                "offensivePrediction": 1 if fields[0] == "yes" else 0,
                "generatedMinorityGroup": fields[1][16:],
                "generatedStereotype": fields[2][8:],
            }
        )
    except Exception:
        print("unable to extract from", completion)
        return pd.Series(
            {
                "offensivePrediction": 1 if fields[0] == "yes" else 0,
                "generatedMinorityGroup": NA_TOKEN,
                "generatedStereotype": NA_TOKEN,
            }
        )


def main():
    seed_everything(42)
    conf = OmegaConf.load(sys.argv[1])

    # dump config to output_dir
    os.makedirs(conf.output_dir, exist_ok=True)
    with open(join(conf.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    with open(conf["manual_prompt"]) as f:
        train_prompt = f.read()

    test_subset = pd.read_json(conf["test_file"], lines=True, orient="records")

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
    for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
        test_item = test_prompt_format.format(post=row["post"])
        prompt = "\n".join([train_prompt, test_item])

        # first generate a binary prediction on whether the post is offensive
        resp = api.generate(
            model=conf.generate_config.model,
            prompt=prompt,
            temperature=0.0,  # no sampling when predicting offensiveness
            max_tokens=4,
            stop=("\n",),
        )

        if conf["class"] == "gpt-3":
            completion_label = fetch_gpt3_completion(resp)
        elif conf["class"] == "jurassic":
            completion_label = fetch_jurassic_completion(resp)
        else:
            raise Exception

        prompt = prompt + completion_label

        resp = api.generate(
            model=conf.generate_config.model,
            prompt=prompt,
            temperature=conf.generate_config.get("temperature", 0.0),
            **generation_kwargs,
        )

        if conf["class"] == "gpt-3":
            completion = fetch_gpt3_completion(resp)
        elif conf["class"] == "jurassic":
            completion = fetch_jurassic_completion(resp)
        else:
            raise Exception

        completion = completion_label + completion
        test_completions.append(completion)

    test_subset["generation"] = test_completions
    test_subset_generated = pd.concat(
        (test_subset[["post", "offensiveYN", "referenceMinorityGroups", "referenceStereotypes"]], test_subset["generation"].apply(extract_fields)), axis="columns"
    )
    test_subset_generated.to_json(
        join(conf.output_dir, "test-df.jsonl"), lines=True, orient="records"
    )
    test_metrics = get_metrics(test_subset_generated)
    test_metrics.to_json(join(conf.output_dir, "test-metrics.json"))


if __name__ == "__main__":
    main()
