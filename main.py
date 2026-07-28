##main.py 

import matplotlib
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import kagglehub
import os
from   typing                  import Callable
from   PIL                     import Image
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler
from   sklearn.decomposition   import PCA
from   sklearn.linear_model    import LogisticRegression
from   sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection   import cross_val_score

# TODO: switch out PIL for something else
# TODO: a logger

# V The dataset only contains images and their associated labels. You may trim additional information.
# V The overall dataset is representative of a classification issue.
# ? You are able to justify your choice.( Choosing the same dataset as another group without justification is forbidden.)

# V Your data could need to be normalised. Images of different sizes, color palettes or resolution might need  some pre-processing.
# V Your data could be unbalanced. Are positive and negative cases both present and equally represented in your data?
# V (X) Are there outliers that skew the data in specific directions or extremes?

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
    assert (load_len >= reduced_dimention or load_len == -1)

    print("Loading dataset from source...")

    for i, patient in train_metadata.iterrows():

        if (i >= load_len and load_len != -1):
            break

        dataset_x.append(
            np.array(
                Image.open(f"{PNEUMONIA_TRAIN_IMAGES}/{patient.patientId}.png").convert("L").resize(img_size)
            ).flatten()
        )

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

def get_call_str(function_like: Callable, *args, **kwargs) -> str:

    call_str = f"{function_like.__qualname__}("

    if args:
        call_str += "".join([str(arg) + ", " for arg in args])

    if kwargs:
        call_str += "".join(
            [f"{key} = {value}, " for key, value in kwargs.items()]
        )

    if args or kwargs:
        call_str = call_str[:-2]

    call_str += ")"
    return call_str

def test_model(model_class, *args, **kwargs) -> tuple[float, float]:

    model = model_class(*args, **kwargs)
    model.fit(train_x, train_y)

    scores = cross_val_score(
        model,
        dataset_x, dataset_y.ravel(),
        cv = 5, scoring = "f1"
    )

    return (scores.mean(), scores.std(), model)

best_mean  = {"value": None, "model": None}
best_std   = {"value": None, "model": None}

def save_best(mean, std, model) -> None:

    global best_mean
    global best_std

    if  best_mean["value"] is None or mean > best_mean["value"]:
        best_mean["value"] = mean
        best_mean["model"] = model

    if  best_std["value"] is None or std < best_std["value"]:
        best_std["value"] = std
        best_std["model"] = model

    return None

#for i in range(10):
#    mean, std, model = test_model(LogisticRegression, max_iter = 1_000 + i * 500)
#    save_best(mean, std, model)
#    print()
#    break

#for i in range(1, 11):
#    for depth in list(range(1, 7)) + [None]:
#
#        mean, std, model = test_model(
#            RandomForestClassifier,
#            n_estimators = i * 50,
#            max_depth = depth
#        )
#
#        save_best(mean, std, model)
#        print()

print(test_model(RandomForestClassifier, n_estimators = 500, max_depth = 6))
