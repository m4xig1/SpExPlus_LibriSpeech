import logging
import os

cur_dir = os.getcwd()


config_trainer = {
    "logger": {"level": logging.INFO},
    "optimizer": {"lr": 1e-3, "weight_decay": 1e-5},
    "lr_scheduler": {  # Reduce on plateau
        "mode": "min",
        "factor": 0.5,
        "patience": 2,
        "min_lr": 1e-6,
        "verbose": False,
        "requires_loss": True,
        "epoch_based": True,
    },
    "epoch_len": 5000,
    # "checkpoint_path": "/home/m4xig1/speaker_extraction_SpEx/checkpoints/",
    # "checkpoint_path": "/kaggle/working/SpExPlus_LibriSpeech/model/",
    "checkpoint_path": cur_dir + "/model/",
    "device": "cuda",
    "no_improvment": 20,  # epochs before stopping the training if val_loss is not decending
    "epochs": 100,
    "save_period": 2,
    "monitor": "min val_loss",
    "max_grad_norm": 256,
}
