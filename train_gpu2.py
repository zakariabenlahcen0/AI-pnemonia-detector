"""
🫁 PNEUMONIA DETECTION - V2 PERFORMANCE EDITION (Target: 93%+)
Optimisé pour RTX 5070 avec Cosine Annealing et Poids Dynamiques
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION AVANCÉE ---
DATA_DIR = 'data/chest_xray'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Hyperparamètres affinés
LR_PHASE1 = 0.0005     # Plus doux pour commencer
LR_PHASE2 = 0.0001     # Très fin pour la phase 2
EPOCHS_PHASE1 = 10     # Warmup plus court
EPOCHS_PHASE2 = 30     # Fine-tuning plus long (C'est là qu'on gagne les %)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Performance Mode sur: {torch.cuda.get_device_name(0)}")

# --- 1. DATA PREPARATION (AUGMENTATION BOOSTÉE) ---
print("\n--- Préparation des Données & Augmentation ---")

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # NOUVEAU: Jeu sur la lumière et le contraste (Crucial pour les rayons X)
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                  for x in ['train', 'val', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                             shuffle=(x == 'train'), num_workers=NUM_WORKERS, pin_memory=True)
               for x in ['train', 'val', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes

# --- CALCUL AUTOMATIQUE DES POIDS (ÉQUILIBRE PARFAIT) ---
# On compte les images pour ne pas biaiser le modèle
n_normal = len(os.listdir(os.path.join(DATA_DIR, 'train', 'NORMAL')))
n_pneumonia = len(os.listdir(os.path.join(DATA_DIR, 'train', 'PNEUMONIA')))
total = n_normal + n_pneumonia

# Formule statistique pour équilibrer les classes
# --- MODIFICATION V3 : ON FORCE L'ÉQUILIBRE ---
# On met 1.0 pour traiter les deux classes à égalité
# (Ou 1.2 si on veut une toute petite sécurité, mais 1.0 est mieux pour le score global)
pos_weight_value = 1.0
# Note : C'est une astuce mathématique. Au lieu de [1, 2], on aura peut-être [1, 1.34]
pos_weight_tensor = torch.tensor([pos_weight_value]).to(device)

print(f"📊 Statistiques: Normal={n_normal}, Pneumonia={n_pneumonia}")
print(f"⚖️ Poids calculé automatiquement: {pos_weight_value:.4f} (Optimisé pour l'équilibre)")


# --- 2. MODEL BUILDING (Regularisation) ---
def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    
    # On ajoute un peu plus de Dropout pour éviter le par-coeur
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.ReLU(),
        nn.Dropout(0.4), # Augmenté à 0.4
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3), # Augmenté à 0.3
        nn.Linear(512, 1),
    )
    return model.to(device)

model = build_model()

# --- 3. TRAINING FUNCTION (Avec Cosine Scheduler) ---
def train_model(model, optimizer, scheduler, num_epochs=25, phase_name="Training"):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    scaler = torch.amp.GradScaler('cuda') # RTX 5070 Power

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        # Utilisation du poids calculé dynamiquement
                        loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                        loss = loss_func(outputs, labels)

                    preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step du scheduler uniquement en phase train
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'{phase_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# --- PHASE 1: WARMUP ---
print("\n🔒 PHASE 1: WARMUP")
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.classifier.parameters(), lr=LR_PHASE1, weight_decay=1e-4)
# Pas de scheduler complexe pour le warmup
model = train_model(model, optimizer, None, num_epochs=EPOCHS_PHASE1, phase_name="Phase 1")

# --- PHASE 2: FINE TUNING (COSINE ANNEALING) ---
print("\n🔓 PHASE 2: FINE TUNING (High Performance)")
for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=1e-4)
# Le CosineAnnealing est l'arme secrète pour les derniers %
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2, eta_min=1e-6)

model = train_model(model, optimizer, scheduler, num_epochs=EPOCHS_PHASE2, phase_name="Phase 2")

# --- EVALUATION ---
print("\n--- Evaluation Finale ---")
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        outputs = model(inputs)
        preds = torch.sigmoid(outputs) > 0.5
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

# Sauvegarde Pro
if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/pneumonia_pro_1933.pth')
print("✅ Modèle Pro sauvegardé: models/pneumonia_pro_1933.pth")