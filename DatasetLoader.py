## DatasetLoader.py

import constants
import os
import numpy  as np
import pandas as pd
from   typing import Callable

class OutdatedCacheError(Exception):
    pass

def load_or_build_cache(dataset_loader: Callable) -> tuple[np.array, np.array]:

    train_metadata = pd.read_csv(constants.PENUMONIA_TRAIN_CSV)
    dataset_y = np.array(tuple(map(lambda out: out, train_metadata["Target"]))[:constants.IMAGE_COUNT])

    try:

        print("Loading dataset from cache...")
        dataset_x = np.load(".cache/dataset_x.npy")
        print("Done.\n")

        if (
            dataset_x.shape[0] != constants.IMAGE_COUNT or
            dataset_x.shape[1] != constants.REDUCED_DIMENTION
        ):
            raise OutdatedCacheError("Loading dataset from source due to outdated cache.")

    except (FileNotFoundError, OutdatedCacheError) as e:

        if isinstance(e, OutdatedCacheError):
            print(e)
        else:
            print(f"{e.strerror}: '{e.filename}'.")

        dataset_x = dataset_loader(
            train_metadata,
            constants.IMAGE_COUNT,
            constants.IMAGE_SIZE,
            constants.REDUCED_DIMENTION,
        )

        if not os.path.exists(".cache"):
            os.makedirs(".cache")

        print("Saving dataset to '.cache/dataset_x.npy'...")
        np.save(".cache/dataset_x", dataset_x)
        print("Done.\n")


    print("Dataset successfully loaded.")
    return (dataset_x, dataset_y)
