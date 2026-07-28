##constants.py

import kagglehub

PENUMONIA = kagglehub.dataset_download("iamtapendu/rsna-pneumonia-processed-dataset")
PENUMONIA_TRAIN_CSV    = PENUMONIA + "/stage2_train_metadata.csv"
PNEUMONIA_TRAIN_IMAGES = PENUMONIA + "/Training/Images"
PNEUMONIA_TRAIN_MASKS  = PENUMONIA + "/Training/Masks"

IMAGE_COUNT = 4000
IMAGE_SIZE  = (400, 400)

PCA_DIMENTION_FACT = 0.1
REDUCED_DIMENTION  = round(IMAGE_COUNT * PCA_DIMENTION_FACT)
