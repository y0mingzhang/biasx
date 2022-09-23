import sys, os
from os.path import join
from omegaconf import OmegaConf

from utils import seed_everything
from modeling import initialize_model_and_tokenizer, train, evaluate
from data import prepare_data


def main():
    seed_everything(42)
    conf = OmegaConf.load(sys.argv[1])

    # dump the config to output_dir
    os.makedirs(conf.output_dir, exist_ok=True)
    with open(join(conf.output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config=conf, f=f)

    model, tokenizer = initialize_model_and_tokenizer(conf)
    data, dataloaders = prepare_data(conf.data_config, tokenizer)

    train(
        conf,
        model,
        tokenizer,
        dataloaders["train"],
        data["train"],
        dataloaders["dev"],
        data["dev"],
    )
    evaluate(conf, model, tokenizer, "test", dataloaders["test"], data["test"])


if __name__ == "__main__":
    main()
