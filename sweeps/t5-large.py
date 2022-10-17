import os

config_template = """output_dir: sweeps/outputs/supervised/{data_split}/t5-large/{lr}-{acc_steps}
model_name: t5-large
{data_split_config}
  batch_size: 8

train_config:
  optimizer: adamw
  train_steps: 8000
  grad_accumulation_steps: {acc_steps}
  lr: {lr}
  eval_every: 800
"""

data_split_configs = {
"implHate":
"""data_config:
  subsample_common_stereotypes: True
  data_dir: data/implHate""",
"SBIC":
"""data_config:
  subsample_common_stereotypes: True
  dev_size: 1000
  data_dir: data/SBIC.v2
  additional_test:
    - test-implHate""",
"SBIC+implHate":
"""data_config:
  subsample_common_stereotypes: True
  dev_size: 1000
  data_dir: data/SBIC+implHate
  additional_test:
    - test-implHate""",
"SBIC+implHate-balanced":
"""data_config:
  subsample_common_stereotypes: True
  dev_size: 1000
  data_dir: data/SBIC+implHate-balanced
  additional_test:
  - test-implHate""",
}

LR = ["2e-4", "1e-4", "5e-5", "2e-5", "1e-5"]
ACC_STEPS = [4, 16]

os.makedirs("sweeps/configs/supervised/t5-large", exist_ok=True)
for data_split, data_split_config in data_split_configs.items():
    for lr in LR:
        for acc_steps in ACC_STEPS:
            config = config_template.format(lr=lr, acc_steps=acc_steps, data_split=data_split, data_split_config=data_split_config)
            with open(f"sweeps/configs/supervised/t5-large/{data_split}-{lr}-{acc_steps}.yaml", "w") as f:
                f.write(config)
