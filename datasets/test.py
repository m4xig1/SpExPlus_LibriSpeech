if __name__ == "__main__":
    import os
    # from libri_dataset import LibriDataset, get_test_dataloader, get_train_dataloader
    from libri_dataset import LibriDataset, get_test_dataloader, get_train_dataloader
    import config
    import soundfile as sf

    dataset = LibriDataset(config.config_dataloader, config.config_dataloader["path_to_train"])
    # print(dataset[0])
    # path = "/home/m4xig1/speaker_extraction_SpEx/libri_dataset/mix/val/84_6295_000123-mixed.wav"
    # audio, sr = sf.read(path)
    # print(audio, sr)
    dataloader = get_train_dataloader(config.config_dataloader)
    for i, batch in enumerate(dataloader):
        print(batch)
        break 
        # if (i > 200):
        #     break
        