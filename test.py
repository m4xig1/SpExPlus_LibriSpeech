import pandas as pd
from tqdm import tqdm
from loss.loss import SpexPlusLoss
from datasets.libri_dataset import get_test_dataloader

from datasets.config import config_dataloader
from model.spex_plus import SpEx_Plus

from logger.visualize import get_visualizer

from metrics.metrics import MetricTracker, SiSdr, Pesq, normalize_audio

import torch
import numpy as np
import os
import logging
import wandb
from run_configs.test_streaming import stream_config
from utils.stream import separate_sources

torch.manual_seed(42)
np.random.seed(42)


def move_batch_to_device(batch: dict, device: torch.device):
    for key in batch.keys():
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def main():
    dataloader = get_test_dataloader(config_dataloader)
    model = SpEx_Plus()
    device = stream_config["device"]
    checkpount = torch.load(stream_config["checkpoint_path"], device)
    model.load_state_dict(checkpount["state_dict"])
    model = model.to(device)

    logger = logging.getLogger("test streaming")
    writer = get_visualizer()
    metrics = {"SI-SDR": SiSdr(), "PesQ": Pesq()}
    log_step = stream_config.get("log step", 20)

    metric_tracker = MetricTracker("loss", *[key for key in metrics], writer=writer)

    model.eval()
    metric_tracker.reset()
    
    pred_table = {}

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="val",
            total=len(dataloader),
        ):
            batch = move_batch_to_device(batch, device)
            outputs = separate_sources(
                model, batch["mix"], batch["reference"], batch["ref_len"], device
            )
            batch.update(outputs)

            for key in metrics:
                writer.set_step(batch_idx, "val")
                res = metrics[key](batch["short"], batch["target"])
                writer.log_scalar(f"{key}", res)
                metric_tracker.update(key, res, n=batch["mix"].shape[0])

            if batch_idx % log_step == 0:
                pred_table[batch_idx // log_step] = {
                    "reference": writer.wandb.Audio(
                        batch["reference"].squeeze(0).detach().cpu().numpy(),
                        sample_rate=16000,
                    ),
                    "mix": writer.wandb.Audio(
                        batch["mix"].squeeze(0).detach().cpu().numpy(),
                        sample_rate=16000,
                    ),
                    "predicted_short": writer.wandb.Audio(
                        normalize_audio(batch["short"].squeeze(0).detach())
                        .cpu()
                        .numpy(),
                        sample_rate=16000,
                    ),
                    "target": writer.wandb.Audio(
                        batch["target"].squeeze(0).detach().cpu().numpy(),
                        sample_rate=16000,
                    ),
                }
                writer.log_table("Audio", pd.DataFrame.from_dict(pred_table, orient="index"))

        print("FINAL METRICS:")
        for metric_name in metric_tracker.keys():
            print(f"{metric_name}: {metric_tracker.avg(metric_name)}")


if __name__ == "__main__":
    # parse args or use config?
    main()
