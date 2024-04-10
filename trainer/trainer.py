import random
from random import shuffle

import PIL
import PIL.Image
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from .base import BaseTrainer
from .utils import inf_loop

from metrics.metrics import normalize_audio, MetricTracker

from .config import config_trainer


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        dataloaders,
        config=config_trainer,
        device="cuda" if torch.cuda.is_available else "cpu",
        lr_scheduler=None,
        len_epoch=None,
    ):
        super().__init__(model, loss, metrics, optimizer, device, config)
        # self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.test_dataloader = dataloaders["test"]

        if len_epoch is None:  # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:  # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler
        self.log_step = 5  # WARNING

        self.train_metrics = MetricTracker(
            "loss", "grad_norm", *[key for key in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[key for key in self.metrics], writer=self.writer
        )
        self.fine_tune = config.get("fine_tune", False)
        self.scheduler_config = config.get("lr_scheduler", None)
        self.grad_accum_iters = config.get("grad_accum_iters", 1)

        if config.get("resume_path", None) is not None:
            self._resume_checkpoint(config["resume_path"])

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"] != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in config file is different "
                "from that of checkpoint. Optimizer parameters not being resumed."
            )
        elif not self.fine_tune:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    @staticmethod
    def move_batch_to_device(batch: dict, device: torch.device):
        """
        move all necessary tensors to the HPU
        """
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        return batch

    def _clip_grad_norm(self):
        """
        gradient clipping to prevent exploding
        """

        if self.config.get("grad_norm_clip", None) is not None:
            clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"])

    def _process_epoch(self, epoch):
        """
        Training logic for an epoch
        return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.log_scalar("epoch", epoch)
        last_train_metrics = None

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e):
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    f"Train Epoch: {epoch} = Loss: {batch['loss'].item() :.6f}"
                )
                # self._log_predictions(**batch)
                self._log_scalars(self.train_metrics)

                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx >= self.len_epoch:  # end train process
                break

        # eval
        log = last_train_metrics
        if last_train_metrics is None:
            self.logger.warning("All batches with logging are skipped while train!")
            log = {}
        
        val_log = self._evaluation_epoch(epoch, self.test_dataloader)
        log.update(**{f"val_{name}": value for name, value in val_log.items()})

        if self.lr_scheduler is not None and self.scheduler_config.get(
            "epoch_based", False
        ):
            if self.scheduler_config.get("requires_loss", True):
                if "val_loss" in log: 
                    self.lr_scheduler.step(log["val_loss"]) 
            else:
                self.lr_scheduler.step()
        return log

    def process_batch(self, batch, batch_idx, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.model(batch["mix"], batch["reference"], batch["ref_len"])
        batch.update(outputs)
        batch["loss"] = (
            self.loss(outputs, batch["target"], batch["speaker_id"], is_train)
            / self.grad_accum_iters
        )
        if is_train:
            batch["loss"].backward()
            if (batch_idx + 1) % self.grad_accum_iters == 0 or (
                batch_idx + 1
            ) == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None and not self.scheduler_config.get(
                    "epoch_based", False
                ):
                    self.lr_scheduler.step()
                self.train_metrics.update("grad_norm", self.get_grad_norm())
                self.optimizer.zero_grad()

        metrics.update("loss", batch["loss"].item())

        for key in self.metrics:
            metrics.update(
                key,
                self.metrics[key](batch["short"], batch["target"]),  # always short?
                n=batch["mix"].shape[0],
            )

        return batch

    def _evaluation_epoch(self, epoch, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc="val",
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    batch_idx,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "val")
            self._log_scalars(self.evaluation_metrics)
            if batch_idx % self.log_step == 0:
                self._log_predictions(**batch)

        return self.evaluation_metrics.result()

    def _log_predictions(
        self,
        reference,
        mix,
        short,
        mid,
        long,
        target,
        examples_to_log=1,
        *args,
        **kwargs,
    ):
        if self.writer is None:
            self.logger.warning(
                "Warning: no visualizer found for logging predicted audios"
            )
            return
        rows = {}
        tuples = list(zip(reference, mix, short, target))
        shuffle(tuples)

        it = 0
        sr = 16000
        for ref, mix, pred, target in tuples[:examples_to_log]:
            rows[it] = {
                "reference": self.writer.wandb.Audio(
                    ref.squeeze(0).detach().cpu().numpy(), sample_rate=sr
                ),
                "mix": self.writer.wandb.Audio(
                    mix.squeeze(0).detach().cpu().numpy(), sample_rate=sr
                ),
                "predicted_short": self.writer.wandb.Audio(
                    normalize_audio(pred.squeeze(0).detach()).cpu().numpy(),
                    sample_rate=sr,
                ),
                "target": self.writer.wandb.Audio(
                    target.squeeze(0).detach().cpu().numpy(), sample_rate=sr
                ),
            }
            it += 1

        self.writer.log_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.log_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
