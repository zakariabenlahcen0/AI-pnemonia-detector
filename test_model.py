import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION WINDOWS ---
# Mets ici le chemin vers ton fichier .pth (tu peux copier le fichier depuis WSL vers ton bureau)
MODEL_PATH = 'models\pneumonia_pro_193.pth' 
# Mets ici le chemin vers ton dossier de test contenant les sous-dossiers 'NORMAL' et 'PNEUMONIA'
TEST_DIR = r'C:\Users\msi\Desktop\PNEUMONIE COPIE\data\chest_xray\test' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Démarrage du test de masse sur {DEVICE}...")

# 1. Recharger l'architecture EXACTE (V3)
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
    # Chargement des poids (map_location gère le passage GPU->CPU si besoin)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("✅ Modèle chargé avec succès.")
        return model
    except Exception as e:
        print(f"❌ Erreur de chargement : {e}")
        return None

# 2. Prétraitement des images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Fonction de test
def run_batch_test():
    model = load_model()
    if not model: return

    y_true = []
    y_pred = []
    
    # Récupérer les images
    normal_images = glob.glob(os.path.join(TEST_DIR, 'NORMAL', '*.*'))
    pneumonia_images = glob.glob(os.path.join(TEST_DIR, 'PNEUMONIA', '*.*'))
    
    all_files = [(f, 0) for f in normal_images] + [(f, 1) for f in pneumonia_images]
    
    print(f"📂 Images trouvées : {len(all_files)} images à analyser.")
    print("⏳ Analyse en cours... (Cela peut prendre quelques secondes)")

    with torch.no_grad():
        for filepath, label in all_files:
            try:
                img = Image.open(filepath).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                output = model(img_tensor)
                probability = torch.sigmoid(output).item()
                
                # SEUIL OPTIMISÉ (Celui qu'on a trouvé)
                prediction = 1 if probability > 0.95 else 0
                
                y_true.append(label)
                y_pred.append(prediction)
            except Exception as e:
                print(f"Erreur sur {filepath}: {e}")

    # 4. Rapport
    print("\n" + "="*50)
    print("📊 RÉSULTATS DU TEST WINDOWS")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['NORMAL', 'PNEUMONIA']))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice de Confusion :")
    print(f"Vrais Sains (TN): {cm[0][0]} | Faux Malades (FP): {cm[0][1]}")
    print(f"Malades Ratés (FN): {cm[1][0]} | Vrais Malades (TP): {cm[1][1]}")

if __name__ == "__main__":
    run_batch_test()