import torch
from torch.cuda import is_available
from metrics.metrics import MetricTracker
from trainer.config import config_stream
from trainer.trainer import Trainer


class StreamTrainer(Trainer):
    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        dataloaders,
        separate_sources,  # function to calculate predictions instead of model.forward()
        config=config_stream,
        device="cuda" if torch.cuda.is_available else "cpu",
        lr_scheduler=None,
        len_epoch=None,
        skip_oom=True,
        log_step=50,
    ):
        super().__init__(
            model,
            loss,
            metrics,
            optimizer,
            dataloaders,
            config,
            device,
            lr_scheduler,
            len_epoch,
            skip_oom,
            log_step,
        )
        self.separate_sources = separate_sources

    def process_batch(self, batch, batch_idx, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        outputs = self.separate_sources(
            model=self.model,
            mix=batch["mix"],
            ref=batch["reference"],
            ref_len=batch["ref_len"],
            device=self.device,
            **self.config["stream"]
        )
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
