config_dataloader = {
    "logging_name": "dataloader",
    "path_to_train": "/kaggle/input/librispeech-mixes/train-clean-100-mixed/",
    "path_to_val": "/kaggle/input/librispeech-mixes/test-clean-mixed/",
    # "path_to_train": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/train/",
    # "path_to_val": "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/val/",
    "train": {
        "batch_size": 4,
        "num_workers": 4,
        "create_index": True,
        # "index_path": "/home/m4xig1/speaker_extraction_SpEx/datasets/",
        "index_path": "/kaggle/working/SpExPlus_LibriSpeech/datasets/",
        "logging_name": "train dataloader",
    },  # batch size = 16?
    "test": {
        "batch_size": 4,
        "num_workers": 4,
        "create_index": True,
        # "index_path": "/home/m4xig1/speaker_extraction_SpEx/datasets/",
        "index_path": "/kaggle/working/SpExPlus_LibriSpeech/datasets/",
        "logging_name": "test dataloader",
    },
    "sample_rate": 16000,
}
