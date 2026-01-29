import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Pneumonia AI Detector", page_icon="🫁", layout="wide")
DEVICE = torch.device("cpu")
MODEL_PATH = 'models/pneumonia_pro_1933.pth' # Ton meilleur modèle

# --- SIDEBAR (MÉTRIQUES) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Performance IA")
    st.markdown("Basé sur **EfficientNet-B3** optimisé sur RTX 5070.")
    
    st.divider()
    
    # Métriques issues de ton dernier test optimize_model.py
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Précision", "92%", delta="High")
    with col2:
        st.metric("Sécurité", "99%", help="Taux de détection des malades (Recall)")
    
    st.markdown("### ⚙️ Réglages")
    threshold = st.slider("Sensibilité (Seuil)", 0.0, 1.0, 0.95, 
                          help="Seuil de confiance nécessaire pour déclarer une maladie.")
    
    st.info("ℹ️ **Note :** Ce modèle est calibré pour être ultra-sensible. Un seuil de 0.95 est recommandé pour éviter les fausses alertes.")

# --- FONCTIONS ---
@st.cache_resource
def load_model():
    try:
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        # Architecture V3 (Correspond à ton entraînement)
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            return model
        return None
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- INTERFACE PRINCIPALE ---
st.title("🫁 Détection Automatique de Pneumonie")
st.markdown("Importez une radiographie thoracique (X-Ray). L'intelligence artificielle l'analysera **instantanément**.")

# Chargement du modèle au démarrage
model = load_model()

if model:
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mise en page : Image à gauche, Résultats à droite
        col_img, col_res = st.columns([1, 1.5])

        with col_img:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Radio importée', use_column_width=True)

        with col_res:
            st.markdown("### 🔍 Analyse en cours...")
            
            # Barre de progression fictive pour l'effet "Calcul" (très rapide)
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.005) # Petit délai pour l'effet visuel
                progress_bar.progress(i + 1)
            
            # Prédiction réelle
            img_tensor = process_image(image)
            with torch.no_grad():
                output = model(img_tensor)
                probability = torch.sigmoid(output).item()

            is_pneumonia = probability > threshold
            
            # Affichage du résultat
            st.divider()
            if is_pneumonia:
                st.error("## ⚠️ RÉSULTAT : PNEUMONIE")
                st.markdown(f"L'IA est sûre à **{probability:.2%}** qu'il y a une infection.")
                
                # Jauge rouge
                st.progress(probability)
                st.warning("🚨 **Attention :** Ce résultat indique une forte probabilité pathologique. Veuillez consulter un médecin.")
            else:
                st.success("## ✅ RÉSULTAT : NORMAL")
                prob_sain = 1 - probability
                st.markdown(f"L'IA pense que les poumons sont sains à **{prob_sain:.2%}**.")
                
                # Jauge verte
                st.progress(prob_sain)
                st.caption("Aucune anomalie détectée avec le seuil actuel.")

else:
    st.error("Impossible de charger le modèle 'pneumonia_pro_193.pth'. Vérifiez qu'il est bien dans le dossier models/.")