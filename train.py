"""
But du Code : Lancer et Gérer l'Entraînement du Modèle de Tri

Ce script est le point d'entrée principal pour entraîner le réseau ResNet18.
Son rôle est de :
1. Configurer les paramètres (epochs, learning rate, GPU/CPU).
2. Définir le nombre de classes à 5 (Emballages, Verre, Plastique, Métal, Non recyclable).
3. Charger les données.
4. Gérer la boucle d'entraînement (forward/backward pass) et la validation du modèle.

NOTE : La section 'torchvision' (Transfer Learning) est optimisée pour utiliser 5 classes
et geler les couches de base, ce qui est la meilleure stratégie pour ce projet.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random

# Importation des modules locaux (le modèle construit de zéro et le modèle torchvision adapté)
from resnet18 import ResNet, BasicBlock
from resnet18_torchvision import build_model
from training_utils import train, validate # Fonctions de boucle d'entraînement et de validation
from utils import save_plots, get_data # Fonctions de chargement des données et de sauvegarde des graphiques

# -------------------------------------------------------------------
# 1. Configuration des Arguments
# -------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default="torchvision", # CHANGEMENT CLÉ : Utiliser torchvision (Transfer Learning) par défaut
    help="choose model built from scratch or the Torchvision model",
    choices=["scratch", "torchvision"],
)
args = vars(parser.parse_args())

# -------------------------------------------------------------------
# 2. Définition des Constantes
# -------------------------------------------------------------------
# Fixer la graine aléatoire pour garantir la reproductibilité des résultats
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# Paramètres d'apprentissage et d'entraînement.
epochs = 20 # Nombre de cycles d'entraînement complets
batch_size = 64 # Nombre d'images traitées à la fois
learning_rate = 0.01 # Vitesse à laquelle le modèle apprend
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Utiliser le GPU si disponible

# -------------------------------------------------------------------
# 3. Chargement des Données
# -------------------------------------------------------------------
# get_data doit être adapté dans utils.py pour charger les images de vos déchets (5 classes)
train_loader, valid_loader = get_data(batch_size=batch_size)

# -------------------------------------------------------------------
# 4. Définition du Modèle
# -------------------------------------------------------------------
# Définir le modèle en fonction du choix dans les arguments (-m scratch ou -m torchvision)
NUM_CLASSES = 5 # VOTRE NOMBRE DE CLASSES (5 poubelles)

if args["model"] == "scratch":
    print("[INFO]: Training ResNet18 built from scratch...")
    # Entraînement de zéro (lent mais nécessaire si le dataset est très différent d'ImageNet)
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=NUM_CLASSES).to(
        device
    )
    plot_name = "resnet_scratch"

if args["model"] == "torchvision":
    print("[INFO]: Training the Torchvision ResNet18 model (Transfer Learning)...")
    # Transfer Learning : on utilise le modèle pré-entraîné (pretrained=True)
    # et on gèle la base (fine_tune=False, voir resnet18_torchvision.py)
    model = build_model(pretrained=True, fine_tune=False, num_classes=NUM_CLASSES).to(device)
    plot_name = "resnet_torchvision"

print(model)

# Affichage des paramètres totaux et entraînables.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.") # Doit être faible en Transfer Learning

# Optimiseur (ajuste les poids du modèle).
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
# Fonction de perte (mesure l'erreur). CrossEntropyLoss est standard pour la classification.
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------------------
# 5. Boucle d'Entraînement
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Listes pour suivre les pertes et précisions (pour les graphiques).
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    
    # Démarrer la boucle des époques.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        
        # Phase d'entraînement (le modèle apprend, les poids sont mis à jour)
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        
        # Phase de validation (on teste la performance, pas de mise à jour des poids)
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, device
        )
        
        # Enregistrement des résultats pour les graphiques
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        
        # Affichage des résultats de l'époque
        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)

    # Sauvegarder les graphiques de perte et de précision une fois l'entraînement terminé.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, name=plot_name)
    print("TRAINING COMPLETE")
