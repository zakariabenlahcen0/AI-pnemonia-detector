import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
BASE_DIR = r'C:\Users\msi\Desktop\PNEUMONIE COPIE'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'pneumonia_pro_1933.pth')
TEST_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🕵️ Démarrage du STRESS TEST sur {DEVICE}...")

# 1. Charger le modèle
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
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Prétraitement standard
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_single(image_pil):
    img_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
    return prob

# --- TEST 1: HALLUCINATION (Bruit et Noir) ---
print("\n--- TEST 1: HALLUCINATIONS (Images artificielles) ---")

# Image Noire
img_black = Image.new('RGB', (224, 224), color='black')
prob_black = predict_single(img_black)
print(f"⚫ Image Noire (Vide) : {prob_black:.4f} ({prob_black*100:.2f}%)")

# Image Blanche
img_white = Image.new('RGB', (224, 224), color='white')
prob_white = predict_single(img_white)
print(f"⚪ Image Blanche (Vide) : {prob_white:.4f} ({prob_white*100:.2f}%)")

# Bruit Aléatoire (Neige TV)
img_noise = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
prob_noise = predict_single(img_noise)
print(f"📺 Bruit Aléatoire (Noise): {prob_noise:.4f} ({prob_noise*100:.2f}%)")


# --- TEST 2 & 3: VALIDATION SUR UNE VRAIE PNEUMONIE ---
# On prend une vraie image de pneumonie pour voir comment il réagit aux altérations
pneumonia_sample = glob.glob(os.path.join(TEST_DIR, 'PNEUMONIA', '*.jpeg'))[0]
img_real = Image.open(pneumonia_sample).convert('RGB').resize((224, 224))
prob_real = predict_single(img_real)

print(f"\n--- REFERENCE (Vrai Malade) ---")
print(f"🫁 Image originale : {prob_real:.4f} ({prob_real*100:.2f}%)")

# Création de l'image masquée (BORDS)
img_masked_borders = img_real.copy()
draw = ImageDraw.Draw(img_masked_borders)
# On dessine des rectangles noirs sur les coins (là où il y a les étiquettes)
draw.rectangle([(0, 0), (50, 224)], fill="black") # Gauche
draw.rectangle([(174, 0), (224, 224)], fill="black") # Droite
prob_border = predict_single(img_masked_borders)

# Création de l'image masquée (CENTRE)
img_masked_center = img_real.copy()
draw = ImageDraw.Draw(img_masked_center)
# On cache les poumons (le centre)
draw.rectangle([(50, 50), (174, 174)], fill="black")
prob_center = predict_single(img_masked_center)

print("\n--- TEST 2: BIAIS (Cache-t-il les étiquettes ?) ---")
print(f"🙈 Bords masqués : {prob_border:.4f} ({prob_border*100:.2f}%)")
if abs(prob_real - prob_border) < 0.2:
    print("✅ BON SIGNE : Le modèle regarde bien le centre, pas les bords.")
else:
    print("⚠️ ALERTE : Le modèle perd confiance sans les bords (tricherie possible).")

print("\n--- TEST 3: ROBUSTESSE (Regarde-t-il les poumons ?) ---")
print(f"🕳️ Poumons masqués (Centre noir) : {prob_center:.4f} ({prob_center*100:.2f}%)")
if prob_center < 0.5:
    print("✅ BON SIGNE : Le modèle ne voit plus la maladie quand on cache les poumons.")
else:
    print("⚠️ ALERTE : Le modèle devine encore la maladie sans voir les poumons (Biais de fond).")

# --- GENERATION VISUELLE ---
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
titles = ["Originale", "Bruit (Hallucination?)", "Sans Bords (Triche?)", "Sans Poumons (Biais?)"]
images = [img_real, img_noise, img_masked_borders, img_masked_center]
scores = [prob_real, prob_noise, prob_border, prob_center]

for i in range(4):
    ax[i].imshow(images[i])
    ax[i].set_title(f"{titles[i]}\nPred: {scores[i]:.2%}")
    ax[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'stress_test_analysis.png'))
print(f"\n📸 Image d'analyse générée : {os.path.join(OUTPUT_DIR, 'stress_test_analysis.png')}")