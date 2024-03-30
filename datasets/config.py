config_dataloader = {
    "logging_name": "dataloader",
    "path_to_train": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/train/",
    "path_to_val": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/val/",
    "train": {
        "batch_size": 4,
        "num_workers": 4,
        "create_index": False,
        "index_path": "/home/m4xig1/speaker_extraction_SpEx/datasets/",
        "logging_name": "train dataloader",
    },  # batch size = 16?
    "test": {
        "batch_size": 4,
        "num_workers": 4,
        "create_index": False,
        "index_path": "/home/m4xig1/speaker_extraction_SpEx/datasets/",
        "logging_name": "test dataloader",
    },
    "sample_rate": 16000,
}
