##main.py 

import matplotlib
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import kagglehub
import os
from   PIL                     import Image
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler
from   sklearn.decomposition   import PCA
from   sklearn.linear_model    import LogisticRegression
from   sklearn.ensemble        import RandomForestClassifier

# TODO: switch out PIL for something else
# TODO: a logger

PENUMONIA = kagglehub.dataset_download("iamtapendu/rsna-pneumonia-processed-dataset")
PENUMONIA_TRAIN_CSV    = PENUMONIA + "/stage2_train_metadata.csv"
PNEUMONIA_TRAIN_IMAGES = PENUMONIA + "/Training/Images"
PNEUMONIA_TRAIN_MASKS  = PENUMONIA + "/Training/Masks"

IMAGE_COUNT = 4000
IMAGE_SIZE  = (400, 400)

PCA_DIMENTION_FACT = 0.1
REDUCED_DIMENTION  = round(IMAGE_COUNT * PCA_DIMENTION_FACT)

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
print(NaNs[NaNs > 0], end = "\n" * 2)

dataset_y = np.array(tuple(map(lambda out: out, train_metadata["Target"]))[:IMAGE_COUNT])

def load_training_data(
        load_len: int,
        img_size: tuple[int, int],
        reduced_dimention: int
) -> np.array:

    dataset_x = []
    assert (load_len >= reduced_dimention)

    print("Loading dataset from source...")

    for i, patient in train_metadata.iterrows():

        if (i >= load_len):
            break

        img = Image.open(f"{PNEUMONIA_TRAIN_IMAGES}/{patient.patientId}.png")
        img = img.resize(img_size)

        dataset_x.append(np.array(img).flatten())

    print("Done.\n")

    print("Standardizing dataset...")
    dataset_x = StandardScaler().fit_transform(dataset_x)
    print("Done.\n")

    print("Reducing dataset dimensionality...")
    dataset_x = PCA(n_components = REDUCED_DIMENTION).fit_transform(dataset_x)
    print("Done.\n")

    return np.array(dataset_x)

class OutdatedCacheError(Exception):
    pass

try:

    print("Loading dataset from cache...")
    dataset_x = np.load(".cache/dataset_x.npy")
    print("Done.\n")

    if (
        dataset_x.shape[0] != IMAGE_COUNT or
        dataset_x.shape[1] != REDUCED_DIMENTION
    ):
        raise OutdatedCacheError("Loading dataset from source due to outdated cache.")

except (FileNotFoundError, OutdatedCacheError) as e:

    if isinstance(e, OutdatedCacheError):
        print(e)
    else:
        print(f"{e.strerror}: '{e.filename}'.")

    dataset_x = load_training_data(
        IMAGE_COUNT,
        IMAGE_SIZE,
        REDUCED_DIMENTION,
    )

    if not os.path.exists(".cache"):
        os.makedirs(".cache")

    print("Saving dataset to '.cache/dataset_x.npy'...")
    np.save(".cache/dataset_x", dataset_x)
    print("Done.\n")

train_x, test_x, train_y, test_y = train_test_split(
    dataset_x,
    dataset_y,
    train_size = 0.8,
    stratify   = dataset_y
)

print("Dataset successfully loaded.")

model = LogisticRegression(max_iter = 5_000)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)

print(model.score(train_x, train_y))
print(model.score(test_x,  test_y ))

pass

model = RandomForestClassifier(500)
model.fit(train_x, train_y)

pred_y = model.predict(test_x)

print(model.score(train_x, train_y))
print(model.score(test_x,  test_y ))
