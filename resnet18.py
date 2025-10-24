"""
Objectif du Code : Définir l'Architecture ResNet18 (à partir de zéro)

Ce fichier a pour but de définir l'architecture du réseau de neurones convolutionnel 
ResNet18 (Residual Network avec 18 couches) en le construisant entièrement à partir 
des modules de base de PyTorch. Il crée la structure complète du réseau. Son rôle 
est d'extraire des caractéristiques complexes de l'image pour la classer dans 
l'une des 5 catégories de votre projet "Tri Intelligent".

Ce modèle est l'alternative "scratch" (entraîné de zéro) à la méthode du 
Transfer Learning (pré-entraîné sur ImageNet).

Basé sur l'article : Deep Residual Learning for Image Recognition.
Lien : https://arxiv.org/pdf/1512.03385v1.pdf
"""

import torch.nn as nn
import torch

from torch import Tensor
from typing import Type


class BasicBlock(nn.Module):
    """
    Définit le bloc de construction fondamental de ResNet18 (BasicBlock), 
    caractérisé par une connexion résiduelle (identité).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1, # Facteur multiplicatif (toujours 1 pour ResNet18)
        downsample: nn.Module = None, # Module de sous-échantillonnage optionnel
    ) -> None:
        super(BasicBlock, self).__init__()
        
        self.expansion = expansion
        self.downsample = downsample
        
        # 1ère Couche Conv : 3x3, avec le stride pour la réduction spatiale si nécessaire
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) # Normalisation de batch
        self.relu = nn.ReLU(inplace=True) 
        
        # 2ème Couche Conv : 3x3
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) # Normalisation de batch

    def forward(self, x: Tensor) -> Tensor:
        identity = x # Sauvegarde de l'entrée pour la connexion résiduelle

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # Si les dimensions (taille spatiale) ont été réduites, applique le downsample
            # à l'entrée 'identity' pour la faire correspondre à 'out'.
            identity = self.downsample(x)

        # Connexion Résiduelle : Ajoute l'entrée originale à la sortie du bloc.
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    Classe principale pour construire le réseau ResNet18.
    """
    def __init__(
        self,
        img_channels: int, # Nombre de canaux de l'image (3 pour RGB)
        num_layers: int, # Doit être 18
        block: Type[BasicBlock], # Le type de bloc à utiliser
        num_classes: int = 5, # CHANGEMENT : 5 classes pour le tri des déchets
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # ResNet18 utilise 4 groupes de 2 BasicBlocks
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64
        # 1ère étape : Couche de convolution initiale (7x7) pour l'extraction de features
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Max Pooling pour une première réduction de la taille (réduction du bruit)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Construction des 4 groupes de couches résiduelles (layer1 à layer4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # Stride=2 réduit la dimension
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Couche finale pour la classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Réduit le feature map final à 1x1
        self.fc = nn.Linear(512 * self.expansion, num_classes) # Couche linéaire (dense) pour 5 sorties

    def _make_layer(
        self, block: Type[BasicBlock], out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """
        Fonction utilitaire pour créer une séquence de BasicBlocks.
        """
        downsample = None
        if stride != 1:
            # Crée la convolution 1x1 pour le sous-échantillonnage si nécessaire
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        
        layers = []
        # Ajoute le premier bloc du groupe (avec downsample s'il y a lieu)
        layers.append(
            block(self.in_channels, out_channels, stride, self.expansion, downsample)
        )
        self.in_channels = out_channels * self.expansion

        # Ajoute les blocs suivants (sans downsample)
        for i in range(1, blocks):
            layers.append(
                block(self.in_channels, out_channels, expansion=self.expansion)
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # Passage avant (forward pass) à travers le réseau
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Aplatissement (Flattening) des features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x) # Couche de classification finale

        return x


if __name__ == "__main__":
    # --- Bloc de Test ---
    # Crée un tenseur d'entrée aléatoire (une image 3 canaux, 224x224 pixels)
    tensor = torch.rand([1, 3, 224, 224])
    # Instancie le modèle pour 5 classes
    model = ResNet(img_channels=3, num_layers=18, block=BasicBlock, num_classes=5)
    print(model)

    # Calcul des paramètres pour vérification
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

    output = model(tensor)
