config_dataloader = {
    "logging_name": "dataloader",
    "path_to_train": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/train/",
    "path_to_val": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/val/",
    "train": {"batch_size": 1, "num_workers": 4}, # batch size = 1?
    "val": {"batch_size": 1, "num_workers": 4},
    "test": {"batch_size": 1, "num_workers": 4},
    "sample_rate": 16000,
}
