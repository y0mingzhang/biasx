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
    "Classify the offensiveness of the given post, and decide whether"
    " it implies a stereotype targeting some identity-related group."
)


def fetch_gpt3_completion(resp: dict) -> str:
    return resp["choices"][0]["text"]


def fetch_jurassic_completion(resp: dict) -> str:
    return resp["completions"][0]["data"]["text"]


prompt_format = """post: {post}
offensive: {offensive}
group targeted: {group}
stereotype: {stereotype}
"""
test_prompt_format = """post: {post}
offensive:"""


def extract_fields(completion: str, partial: bool = False) -> pd.Series:
    try:
        fields = [f.strip() for f in completion.split("\n")]
        assert fields[0] in ("yes", "no")
        if partial:
            return {"offensivePrediction": 1 if fields[0] == "yes" else 0}

        assert fields[1].startswith("group targeted: ")
        assert fields[2].startswith("stereotype: ")
        return pd.Series(
            {
                "offensivePrediction": 1 if fields[0] == "yes" else 0,
                "generatedMinorityGroup": fields[1][16:],
                "generatedStereotype": fields[2][12:],
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
    device = get_device()

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

    if "manual_prompt" in conf:
        with open(conf["manual_prompt"]) as f:
            train_prompt = f.read()
    else:
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

    if "test_file" in conf:
        test_subset = pd.read_json(conf["test_file"], lines=True, orient="records")
    else:
        test_subset = dataframes["test"].sample(test_samples, random_state=test_seed)

    if "entailment_config" in conf:
        entailment_mode = conf["entailment_config"]["mode"]
        entailment_threshold = conf["entailment_config"].get("threshold", 0.5)
    else:
        entailment_mode = None

    if entailment_mode:
        entail_model, entail_tokenizer = load_entail_model(conf["entailment_config"])
        entail_model.to(device)
        entail_model.eval()

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
        prompt = "\n\n".join([train_prompt, test_item])

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

        offensive = extract_fields(completion_label, partial=True)[
            "offensivePrediction"
        ]

        if offensive:
            # generate group/stereotype only if post is offensive
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
            stereotype = extract_fields(completion)["generatedStereotype"]

            if stereotype != NA_TOKEN and entailment_mode:
                with torch.no_grad():
                    for i in range(10):
                        tokenized = entail_tokenizer(
                            row["post"], stereotype, return_tensors="pt"
                        )
                        tokenized = to_device(tokenized, device)
                        outputs = entail_model(**tokenized)
                        entail_prob = outputs.logits.softmax(1)[0][1].item()
                        if entail_prob < entailment_threshold:
                            if entailment_mode == "suppress":
                                print(
                                    f"Suppress post={row['post']} stereotype={stereotype}"
                                )
                                completion = ""
                            elif entailment_mode == "resample" and i == 9:
                                print("exceeded max retries, skip generation for post", row["post"])
                                completion = ""
                                break
                        else:
                            break
                        resp = api.generate(
                            overwrite_cache=True,
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
                        stereotype = extract_fields(completion)["generatedStereotype"]


        else:
            completion = completion_label
        test_completions.append(completion)

    test_subset["generation"] = test_completions
    test_subset_generated = pd.concat(
        (test_subset, test_subset["generation"].apply(extract_fields)), axis="columns"
    )
    test_subset_generated.insert(
        loc=0, column="index", value=test_subset_generated.index
    )
    test_subset_generated.to_json(
        join(conf.output_dir, "test-df.jsonl"), lines=True, orient="records"
    )
    test_metrics = get_metrics(test_subset_generated)
    test_metrics.to_json(join(conf.output_dir, "test-metrics.json"))


if __name__ == "__main__":
    main()
