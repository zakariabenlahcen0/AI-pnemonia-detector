import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- CONFIGURATION ---
BASE_DIR = r'C:\Users\msi\Desktop\PNEUMONIE COPIE'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_pro_193.pth')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs') # Le dossier cible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.95 

# 1. Création automatique du dossier outputs
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"📂 Dossier créé : {OUTPUT_DIR}")
else:
    print(f"📂 Dossier trouvé : {OUTPUT_DIR}")

print(f"🎨 Génération des graphiques en cours...")

# 2. Charger le modèle
def load_model():
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
    # Chargement sur CPU pour être sûr (Windows)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# 3. Prétraitement
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Calculs
model = load_model()
y_true = []
y_probs = []
y_pred = []

normal_files = glob.glob(os.path.join(TEST_DIR, 'NORMAL', '*.*'))
pneumonia_files = glob.glob(os.path.join(TEST_DIR, 'PNEUMONIA', '*.*'))
all_files = [(f, 0) for f in normal_files] + [(f, 1) for f in pneumonia_files]

print(f"📊 Analyse de {len(all_files)} images...")

with torch.no_grad():
    for filepath, label in all_files:
        try:
            img = Image.open(filepath).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            
            output = model(img_tensor)
            probability = torch.sigmoid(output).item()
            
            y_true.append(label)
            y_probs.append(probability)
            y_pred.append(1 if probability > THRESHOLD else 0)
        except:
            pass

# --- GRAPHIQUE 1 : MATRICE DE CONFUSION ---
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Prédit NORMAL', 'Prédit PNEUMONIE'],
            yticklabels=['Vrai NORMAL', 'Vrai PNEUMONIE'],
            annot_kws={"size": 16, "weight": "bold"})

plt.title(f'Matrice de Confusion (Seuil = {THRESHOLD})', fontsize=14)
plt.ylabel('Réalité', fontsize=12)
plt.xlabel('Prédiction IA', fontsize=12)
plt.tight_layout()

# Sauvegarde dans le dossier outputs
save_path_cm = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(save_path_cm, dpi=300)
print(f"✅ Image sauvegardée : {save_path_cm}")

# --- GRAPHIQUE 2 : COURBE ROC / AUC ---
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taux de Faux Positifs', fontsize=11)
plt.ylabel('Taux de Vrais Positifs', fontsize=11)
plt.title('Performance ROC', fontsize=13)
plt.legend(loc="lower right", fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()

# Sauvegarde dans le dossier outputs
save_path_roc = os.path.join(OUTPUT_DIR, 'auc_roc_curve.png')
plt.savefig(save_path_roc, dpi=300)
print(f"✅ Image sauvegardée : {save_path_roc}")