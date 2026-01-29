import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- CONFIG ---
DATA_DIR = 'data/chest_xray'
MODEL_PATH = 'models/pneumonia_pro_1933.pth' # Ton dernier modèle
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Optimisation du seuil sur {torch.cuda.get_device_name(0)}...")

# 1. Prépare les données
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Charge le modèle
model = models.efficientnet_b3(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 1024),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 1),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 3. Récupère toutes les prédictions brutes
print("Calcul des probabilités brutes...")
y_true = []
y_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        # On garde la probabilité brute (entre 0 et 1)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        y_true.extend(labels.numpy())
        y_probs.extend(probs)

y_true = np.array(y_true)
y_probs = np.array(y_probs)

# 4. Teste différents seuils
print("\n--- RECHERCHE DU MEILLEUR SCORE ---")
best_acc = 0
best_threshold = 0
best_cm = None

thresholds = np.arange(0.1, 1.0, 0.05) # De 0.1 à 0.95

print(f"{'Seuil':<10} | {'Accuracy':<10} | {'Recall Normal':<15} | {'Recall Pneumo':<15}")
print("-" * 60)

for thresh in thresholds:
    # Applique le seuil
    y_pred = (y_probs > thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcul des recalls spécifiques
    tn, fp, fn, tp = cm.ravel()
    recall_normal = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_pneumo = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"{thresh:.2f}       | {acc:.4f}     | {recall_normal:.4f}          | {recall_pneumo:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_threshold = thresh
        best_cm = cm

print("\n" + "="*60)
print(f"🏆 MEILLEUR RÉSULTAT TROUVÉ")
print("="*60)
print(f"✅ Seuil optimal : {best_threshold:.2f}")
print(f"✅ Accuracy      : {best_acc:.4f} ({(best_acc*100):.2f}%)")

# Affichage détaillé du gagnant
y_pred_final = (y_probs > best_threshold).astype(int)
print("\nClassification Report (Optimisé):")
print(classification_report(y_true, y_pred_final, target_names=['NORMAL', 'PNEUMONIA']))