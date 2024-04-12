import os

cur_dir = os.getcwd()


config_dataloader = {
    "logging_name": "dataloader",
    "path_to_train": "/kaggle/input/librispeech-mixes/train-clean-100-mixed/",
    "path_to_val": "/kaggle/input/librispeech-mixes/test-clean-mixed/",
    # "path_to_train": cur_dir + "/librispeech-mixes/train-clean-100-mixed/",
    # "path_to_val": cur_dir + "/librispeech-mixes/test-clean-mixed/",
    "train": {
        "batch_size": 2,
        "num_workers": 4,
        "create_index": True,
        # "index_path": cur_dir + "/datasets/",
        "index_path": "/kaggle/working/SpExPlus_LibriSpeech/datasets/",
        "logging_name": "train dataloader",
        "max_len": None,  # number of samples in dataset
    },
    "test": {
        "batch_size": 1,
        "num_workers": 4,
        "create_index": True,
        # "index_path": cur_dir + "/datasets/",
        "index_path": "/kaggle/working/SpExPlus_LibriSpeech/datasets/",
        "logging_name": "test dataloader",
        "max_len": 500,
    },
    "sample_rate": 16000,
}
