##main.py 

import matplotlib
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import kagglehub
from   PIL import Image

# TODO: switch out PIL for something else

PENUMONIA = kagglehub.dataset_download("iamtapendu/rsna-pneumonia-processed-dataset")
PENUMONIA_TRAIN_CSV    = PENUMONIA + "/stage2_train_metadata.csv"
PNEUMONIA_TRAIN_IMAGES = PENUMONIA + "/Training/Images"
PNEUMONIA_TRAIN_MASKS  = PENUMONIA + "/Training/Masks"

train_metadata = pd.read_csv(PENUMONIA_TRAIN_CSV)

for patient_id in train_metadata.get("patientId"):

    for folder in (PNEUMONIA_TRAIN_IMAGES, PNEUMONIA_TRAIN_MASKS):

        filepath = f"{folder}/{patient_id}.png"

        try:
            with open(filepath) as file:
                pass
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")
            exit(1)

print("Training data files present.")

for key in tuple(train_metadata)[0:]:
    print()
    print(train_metadata[key].value_counts())

print("\nNaNs:")
NaNs = train_metadata.isnull().sum()
print(NaNs[NaNs > 0])

train_y = tuple(train_metadata["Target"])[:100]
train_y = np.array(
    tuple(
        map(
            lambda out: [out],
            train_y
        )
    )
)

def load_training_data(load_len: int, img_size: tuple[int, int]) -> np.array:

    train_x = []

    for i, patient in train_metadata.iterrows():

        if (i >= load_len):
            break

        img = Image.open(f"{PNEUMONIA_TRAIN_IMAGES}/{patient.patientId}.png")
        img = img.resize(img_size)
        train_x.append(np.array(img).flatten())

    return np.array(train_x)

try:
    train_x = np.load(".cache/train_x.npy")
except FileNotFoundError:
    train_x = load_training_data(100, (200, 200))
    np.save(".cache/train_x", train_x)

print(train_x.shape)
print(train_y.shape)
