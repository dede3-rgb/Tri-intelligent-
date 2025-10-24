"""
Objectif du Code : Gestion des Données et Visualisation

Ce fichier est le point de connexion entre le modèle et les images de déchets.
Il sert à :
1. Charger les images du dataset de tri des déchets (`get_data`) et les préparer 
   pour le modèle ResNet18 (redimensionnement, normalisation).
2. Fournir une structure (`save_plots`) pour enregistrer les graphiques de 
   précision et de perte, ce qui est crucial pour la phase d'évaluation 
   du projet.

ATTENTION : La fonction get_data() est ici adaptée pour lire les images de vos 
5 catégories de déchets depuis une structure de dossiers standard (ImageFolder). 
Assurez-vous que vos données sont structurées en dossiers 'data/train/' et 'data/validation/', 
avec les 5 classes comme sous-dossiers.
"""

import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms # Utilisé pour charger les dossiers d'images

plt.style.use("ggplot")


# Fonction pour charger les données (adaptée pour le dataset de déchets)
def get_data(batch_size=64):
    print("[INFO]: Chargement du dataset personnalisé de déchets...")
    
    # Chemins des données (Assurez-vous que le dossier 'data' existe à la racine du projet)
    data_dir = 'data' 
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'validation')
    
    # -------------------------------------------------------------------------
    # Transformations pour ResNet18 (standard pour les modèles pré-entraînés)
    # -------------------------------------------------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),      # 1. Redimensionner à 256 pixels
            transforms.CenterCrop(224),  # 2. Rognage central à 224x224 (taille d'entrée ResNet)
            transforms.ToTensor(),       # 3. Convertir l'image en Tenseur PyTorch
            # 4. Normalisation avec les moyennes et écarts types d'ImageNet (Transfer Learning)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Créer les datasets à partir des dossiers (ImageFolder attribue les labels 0 à 4 
    # automatiquement en fonction des sous-dossiers de classes)
    dataset_train = datasets.ImageFolder(
        root=train_dir,
        transform=data_transforms['train']
    )

    dataset_valid = datasets.ImageFolder(
        root=valid_dir,
        transform=data_transforms['validation']
    )

    # Créer les DataLoaders (permet de charger les images par lots (batches))
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    
    print(f"[INFO]: {len(dataset_train)} images d'entraînement chargées.")
    print(f"[INFO]: {len(dataset_valid)} images de validation chargées.")

    return train_loader, valid_loader


# Fonction pour sauvegarder les graphiques de perte et de précision.
def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    """
    Sauvegarde la précision et la perte (loss) au fil des époques sur le disque.
    """
    # Graphique de Précision (Accuracy)
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="tab:red", linestyle="-", label="validataion accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    # Sauvegarde dans le dossier 'outputs'
    plt.savefig(os.path.join("outputs", name + "_accuracy.png"))

    # Graphique de Perte (Loss)
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="tab:red", linestyle="-", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("outputs", name + "_loss.png"))
