#!/bin/bash

git clone https://github.com/m4xig1/SpExPlus_LibriSpeech.git

mkdir SpExPlus_LibriSpeech/libri_dataset

cd SpExPlus_LibriSpeech/libri_dataset

# Download the dataset from Kaggle (assuming you have Kaggle CLI installed and configured)
kaggle datasets download -d lizakonstantinova/librispeech-mixes

sudo apt-get install unzip
unzip librispeech-mixes.zip

rm librispeech-mixes.zip

cd ../..

echo "Repository cloned and dataset downloaded successfully!"