"""
🫁 PNEUMONIA DETECTION - V4 BALANCED EDITION
Stratégie : WeightedRandomSampler + Early Stopping + EfficientNet-B3
Objectif : Équilibrer Précision et Rappel (Moins d'hallucinations)
"""
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report
import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONFIGURATION ---
DATA_DIR = 'data/chest_xray'
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
PATIENCE = 5  # Early Stopping (Arrêt si pas d'amélioration après 5 époques)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Balanced Mode sur: {torch.cuda.get_device_name(0)}")

# --- 1. DATA PREPARATION ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # Augmentation modérée pour éviter de trop déformer les poumons sains
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
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

# --- 2. THE SECRET WEAPON: WEIGHTED SAMPLER ---
# On calcule le poids de chaque image pour forcer l'équilibre 50/50
print("⚖️ Calcul de l'échantillonnage équilibré...")
targets = [label for _, label in image_datasets['train'].samples]
class_count = np.bincount(targets)
class_weights = 1. / class_count
sample_weights = torch.tensor([class_weights[t] for t in targets])

# Ce sampler va forcer le modèle à voir autant de NORMAL que de PNEUMONIA
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True),
    'val': DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
    'test': DataLoader(image_datasets['test'], batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
class_names = image_datasets['train'].classes
print(f"📊 Données prêtes. Classes: {class_names}")

# --- 3. MODEL BUILDING ---
def build_model():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    
    # Dropout ajusté pour moins de paranoïa
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512), # Stabilise l'apprentissage
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
    )
    return model.to(device)

model = build_model()

# --- 4. EARLY STOPPING CLASS ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- 5. TRAINING LOOP (Optimisée) ---
criterion = nn.BCEWithLogitsLoss() # Plus besoin de pos_weight car on a le Sampler !
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3) # Weight decay augmenté pour réduire la paranoïa
# Scheduler qui réduit le LR si la loss stagne (Plateau)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
early_stopper = EarlyStopping(patience=PATIENCE)

def train_model(model, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    
    scaler = torch.amp.GradScaler('cuda')

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
                        loss = criterion(outputs, labels)

                    preds = torch.sigmoid(outputs) > 0.5

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Sauvegarde du meilleur modèle basé sur la Validation Loss (plus fiable que l'accuracy)
            if phase == 'val':
                scheduler.step(epoch_loss) # Mise à jour du scheduler
                early_stopper(epoch_loss)  # Vérification Early Stopping
                
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        if early_stopper.early_stop:
            print("🛑 Early stopping triggered!")
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Loss: {best_loss:.4f} Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model

# Lancement unique (Pas de phase 1/2, on fait tout d'un coup avec le sampler)
model = train_model(model, num_epochs=25)

# --- EVALUATION ---
print("\n--- Evaluation Finale ---")
model.eval()
y_true = []
y_pred_probs = []

with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        y_true.extend(labels.cpu().numpy())
        y_pred_probs.extend(probs.cpu().numpy())

# Test avec un seuil plus équilibré (0.5 car le sampler a équilibré les données)
# Tu pourras ajuster ce seuil (0.65) plus tard avec optimize_model.py
y_pred = np.array(y_pred_probs) > 0.5 

print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))

if not os.path.exists('models'):
    os.makedirs('models')
torch.save(model.state_dict(), 'models/pneumonia_v4_balanced.pth')
print("✅ Modèle V4 Balanced sauvegardé: models/pneumonia_v4_balanced.pth")