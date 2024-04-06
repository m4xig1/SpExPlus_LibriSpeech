import logging

config_trainer = {
    "logger": {"level": logging.INFO},
    "optimizer": {"lr": 0.0025, "weight_decay": 1e-5},
    "lrScheduler": {  # Reduce on plateau
        "mode": "min",
        "factor": 0.5,
        "patience": 2,
        "min_lr": 0,
        "verbose": True,
    },
    "epoch_len": 1000,
    "nCheckpoints": 10,
    # "checkpoint_path": "/home/m4xig1/speaker_extraction_SpEx/checkpoints/",
    "checkpoint_path": "/kaggle/working/SpExPlus_LibriSpeech/model/",
    "device": "cuda", 
    "no_improvment": 6,  # steps before stopping the training
}
