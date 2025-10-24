"""
But du Code : Implémentation du Transfer Learning pour le Tri Intelligent

Ce fichier définit la fonction pour charger et configurer le modèle ResNet18 pré-entraîné
(Transfer Learning). Il adapte le modèle pour le tri de vos 5 catégories de déchets :
Emballages & papiers, Verre, Plastique, Métal, Non recyclable.

La stratégie recommandée est d'utiliser le modèle pré-entraîné (pretrained=True)
et de geler les couches de base (fine_tune=False) pour entraîner rapidement 
uniquement la tête de classification de 5 sorties.
"""

import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=True, fine_tune=True, num_classes=5):  # 5 classes pour le tri des déchets
    
    # ----------------------------------------------------
    # 1. Chargement du modèle
    # ----------------------------------------------------
    if pretrained:
        print("[INFO]: Loading pre-trained weights (Transfer Learning)...")
        # Charge le modèle ResNet18 avec les poids appris sur ImageNet
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif not pretrained:
        print("[INFO]: Not loading pre-trained weights (Training from scratch)...")
        # Charge l'architecture sans poids pré-entraînés
        model = models.resnet18(weights=None)
    
    # ----------------------------------------------------
    # 2. Logique de Fine-Tuning (Gel des couches de base)
    # ----------------------------------------------------
    if not fine_tune:
        print("[INFO]: Freezing base layers (Training only the final head)...")
        # Geler toutes les couches du réseau de base (transfer learning standard)
        for params in model.parameters():
            params.requires_grad = False
    elif fine_tune:
        print("[INFO]: Fine-tuning all layers (Training the whole network)...")
        # Laisser toutes les couches du réseau de base entraînables
        pass 

    # ----------------------------------------------------
    # 3. Adaptation de la Tête de Classification (Couche Finale)
    # ----------------------------------------------------
    
    # Récupérer le nombre de features en entrée de la couche finale (512 pour ResNet18)
    # Ceci est la définition de la variable num_ftrs
    num_ftrs = model.fc.in_features 

    # Remplacer la couche finale pour qu'elle ait le bon nombre de classes (5 sorties)
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Assurer que la nouvelle couche finale est TOUJOURS entraînable 
    # (même si le reste du réseau est gelé)
    for params in model.fc.parameters():
        params.requires_grad = True

    return model


if __name__ == "__main__":
    # --- Bloc de Test ---
    # Teste le modèle avec la configuration de Transfer Learning recommandée (précis et rapide)
    model = build_model(pretrained=True, fine_tune=False, num_classes=5) 
    
    print("\n--- Configuration du Modèle Tri Intelligent ---")
    
    # Calcul des paramètres pour vérification
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Trainable parameters (New Head Only): {total_trainable_params:,}") 
    # Le nombre de paramètres entraînés doit être très faible (~1025) car la base est gelée.
