import torch

from tqdm import tqdm # Importe tqdm pour la barre de progression

"""
Objectif du Code : Le Moteur de l'Apprentissage (Training Utilities)

Ce fichier définit les deux fonctions fondamentales qui contrôlent l'apprentissage du modèle :
1. train() : Met à jour les poids du modèle en se basant sur la perte (erreurs) et l'optimiseur.
2. validate() : Évalue la performance du modèle sur des données de test/validation sans modifier ses poids.

Ce code est générique et parfaitement adapté pour la tâche de classification de votre projet 
Tri Intelligent (5 classes). Il n'y a rien à y changer.
"""


# Fonction d'entraînement : met à jour les poids du modèle.
def train(model, trainloader, optimizer, criterion, device):
    model.train() # Mode entraînement (active le Dropout, etc.)
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    
    # Boucle sur les données d'entraînement, avec barre de progression de tqdm
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device) # Envoi des images sur le GPU/CPU
        labels = labels.to(device) # Envoi des étiquettes (classes 0 à 4) sur le GPU/CPU
        
        optimizer.zero_grad() # Remet les gradients précédents à zéro (étape obligatoire)
        
        # Forward pass : le modèle fait une prédiction
        outputs = model(image) 
        
        # Calcul de la perte (mesure l'erreur)
        loss = criterion(outputs, labels) 
        train_running_loss += loss.item()
        
        # Calcul de la précision : trouve la classe prédite avec la probabilité maximale
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        
        # Backpropagation : calcule le gradient (l'influence de chaque poids sur l'erreur)
        loss.backward()
        
        # Update the weights : ajuste les poids du modèle via l'optimiseur
        optimizer.step()

    # Calcul de la perte et de la précision pour l'époque complète
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Fonction de validation : évalue le modèle sans modifier ses poids.
def validate(model, testloader, criterion, device):
    model.eval() # Mode évaluation (désactive le Dropout, etc.)
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad(): # TRES IMPORTANT : désactive le calcul de gradient (gain de temps)
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            # Forward pass (pas de différence avec l'entraînement)
            outputs = model(image)
            
            # Calcul de la perte et de la précision
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Calcul de la perte et de la précision pour l'époque complète
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
