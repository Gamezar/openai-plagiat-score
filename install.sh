#!/bin/bash
#
pip install gdown

FILE_ID="16Be7V62ew3PVf5pbxNOr_uLB1CTGpYg-"
OUTPUT="roberta-base-openai-detector.zip"

gdown "https://drive.google.com/uc?id=${FILE_ID}" -O ${OUTPUT}

unzip -o ${OUTPUT}

rm ${OUTPUT}

echo "File downloaded and unzipped successfully."

pip install -r requirements.txt
