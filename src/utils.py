import random, torch, os, glob
import numpy as np
from os.path import join, basename


def seed_everything(seed: int) -> None:
    # seed to get deterministic experiment results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device() -> str:
    # figure out which device to use
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    # put a batch onto some device
    return {k: v.to(device) for k, v in batch.items()}


def load_last_model(output_dir, model):
    # load latest model
    ckpt_dir = join(output_dir, "ckpt")
    if not os.path.exists(ckpt_dir):
        return model, 0

    def step_from_ckpt(s):
        return int(basename(s)[:-3])

    ckpts = {step_from_ckpt(s): s for s in glob.glob(join(ckpt_dir, "*"))}
    if not ckpts:
        return model, 0
    last_step = max(ckpts)
    last_ckpt = ckpts[last_step]
    model.load_state_dict(torch.load(last_ckpt))
    return model, last_step


def save_model(output_dir, model, step):
    # save model
    ckpt_dir = join(output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = join(ckpt_dir, f"{step}.pt")
    torch.save(model.state_dict(), ckpt)


def num_workers() -> int:
    return min(os.cpu_count(), 16)
