import kagglehub
from pathlib import Path

# Télécharger la dernière version du dataset
path = kagglehub.dataset_download("farzadnekouei/trash-type-image-dataset")

print("Path to dataset files:", path)
