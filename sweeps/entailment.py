import os

config_template = """output_dir: sweeps/outputs/entailment/{data_split}/{model_name_dir}-{lr}
model_name: {model_name}
{data_split_config}
train_config:
  optimizer: adamw
  train_steps: 4000
  grad_accumulation_steps: 1
  lr: {lr}
  eval_every: 200
"""

data_split_configs = {
"implHate":
"""data_config:
  data_dir: data/implHate
  batch_size: 16""",
"SBIC":
"""data_config:
  data_dir: data/SBIC.v2
  batch_size: 16""",
}

LR = ["2e-4", "1e-4", "5e-5", "2e-5", "1e-5"]

for data_split, data_split_config in data_split_configs.items():
    for lr in LR:
            for model_name in ["bert-base-uncased", "microsoft/deberta-v3-large"]:
                model_name_dir = model_name.split('/')[-1]
                os.makedirs(f"sweeps/configs/entailment/{model_name_dir}", exist_ok=True)
                config = config_template.format(lr=lr, model_name=model_name, data_split_config=data_split_config, model_name_dir=model_name_dir, data_split=data_split)
                with open(f"sweeps/configs/entailment/{model_name_dir}/{data_split}-{lr}.yaml", "w") as f:
                    f.write(config)
