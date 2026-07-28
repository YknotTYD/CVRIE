## main.py 

import constants
import numpy  as np
import pandas as pd
import DatasetLoader
from   PIL                     import Image
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler
from   sklearn.decomposition   import PCA
from   sklearn.linear_model    import LogisticRegression
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.model_selection import cross_val_score

# TODO: switch out PIL for something else
# TODO: a logger

#  V The dataset only contains images and their associated labels. You may trim additional information.
#  V The overall dataset is representative of a classification issue.
#  ? You are able to justify your choice.( Choosing the same dataset as another group without justification is forbidden.)

#  V Your data could need to be normalised. Images of different sizes, color palettes or resolution might need  some pre-processing.
#  V Your data could be unbalanced. Are positive and negative cases both present and equally represented in your data?
#  V (X) Are there outliers that skew the data in specific directions or extremes?

# (V) How do you split training data from evaluation data?
#  X Each time, your model should try to adapt itself to reduce the value of its loss function. This concept should be part of your notebook

def load_training_data(
        train_metadata:    pd.DataFrame,
        load_len:          int,
        img_size:          tuple[int, int],
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
                Image.open(f"{constants.PNEUMONIA_TRAIN_IMAGES}/{patient.patientId}.png").convert("L").resize(img_size)
            ).flatten()
        )

    print("Done.\n")

    print("Standardizing dataset...")
    dataset_x = StandardScaler().fit_transform(dataset_x)
    print("Done.\n")

    print("Reducing dataset dimensionality...")
    dataset_x = PCA(n_components = constants.REDUCED_DIMENTION).fit_transform(dataset_x)
    print("Done.\n")

    return np.array(dataset_x)

dataset_x, dataset_y = DatasetLoader.load_or_build_cache(load_training_data)

train_x, test_x, train_y, test_y = train_test_split(
    dataset_x,
    dataset_y,
    train_size = 0.8,
    stratify   = dataset_y
)

def test_model(model_class, *args, **kwargs) -> tuple[float, float]:

    model = model_class(*args, **kwargs)
    model.fit(train_x, train_y)

    scores = cross_val_score(
        model,
        dataset_x, dataset_y.ravel(),
        cv = 5, scoring = "f1"
    )

    return (scores.mean(), scores.std(), model)

print(test_model(RandomForestClassifier, n_estimators = 500, max_depth = 6))

# best_mean = {"value": None, "model": None}
# best_std  = {"value": None, "model": None}

#for i in range(10):
#    mean, std, model = test_model(LogisticRegression, max_iter = 1_000 + i * 500)
#    save_best(best_mean, best_std, mean, std, model)
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
#        save_best(best_mean, best_std, mean, std, model)
#        print()
