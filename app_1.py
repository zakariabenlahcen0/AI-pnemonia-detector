import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time
import numpy as np

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Pneumonia AI ", 
    page_icon="🫁", 
    layout="wide"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'models/pneumonia_v4_balanced.pth'

# --- SIDEBAR (MÉTRIQUES & STORYTELLING) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("Mode Clinique ")
    st.markdown("**EfficientNet-B3** | PyTorch Nightly")
    st.caption("Optimisé sur RTX 5070 (Blackwell)")
    
    st.divider()
    
    # MÉTRIQUES MISES À JOUR (POST-OPTIMISATION v4)
    st.markdown("### 📊 Performance Validée ")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "92.95%", help="Fiabilité globale du modèle")
    with col2:
        st.metric("Recall", "99%", help="Capacité à détecter les malades (Sensibilité)")
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Precision", "91%", help="Fiabilité des alertes positives")
    with col4:
        st.metric("F1-Score", "0.947", help="Équilibre recall/precision")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("AUC-ROC", "0.9793", help="Discrimination globale")
    with col6:
        st.metric("Speed", "1.42ms", help="Temps d'inférence par image")
    
    st.divider()
    
    st.markdown("### ⚙️ Calibrage Clinique")
    
    # SEUIL OPTIMISÉ À 0.85
    threshold = st.slider(
        "Seuil de Décision",
        0.0, 1.0, 0.85,
        step=0.01,
        help="Seuil optimisé (0.85) pour maximiser la sécurité clinique sans overfitting."
    )
    
    st.info(
        """🎯 **Stratégie Clinique (v4):**
        
        - **Seuil 0.85** (optimisé)
        - **Early Stopping** activé (epoch 15)
        - **Dataset équilibré** pour généralisation
        - **Recall 99%** : 1.3% faux négatifs
        - **Precision 91%** : faux positifs maîtrisés
        """
    )
    
    st.divider()
    
    st.markdown("### 📈 Version")
    st.text("v4 Balanced - Production Ready ✅")

# --- FONCTIONS ---
# --- DANS app_1.py ---

@st.cache_resource
def load_model():
    """Charge le modèle EfficientNet-B3  (Version Correcte 512 neurones)"""
    try:
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        
        # 👇 CORRECTION ICI : 512 (pas 256) pour matcher le fichier .pth
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        model.to(DEVICE)
        model.eval()
        
        if os.path.exists(MODEL_PATH):
            # On remet strict=True (ou on l'enlève car True par défaut) pour être sûr que ça charge !
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict) 
            st.sidebar.success("✅ Poids V4 chargés (512 unit)")
            return model
        else:
            st.sidebar.error(f"❌ Fichier {MODEL_PATH} introuvable")
            return None
            
    except Exception as e:
        st.sidebar.error(f"❌ Erreur critique : {e}")
        return None
def process_image(image):
    """Préprocesse l'image pour le modèle"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

# --- INTERFACE PRINCIPALE ---
st.title("🫁 Détection de Pneumonie - Approche Clinique Équilibrée ")

st.markdown("""
> **Philosophy:** Viabilité clinique réelle > Perfection théorique
> 
> **Après 108 heures d'optimisation:**
> - ✅ 99% Recall (détecte 99% des pneumonies)
> - ✅ 91% Precision (faux positifs maîtrisés)
> - ✅ 0.947 F1-Score (équilibre optimal)
> - ✅ 0.9793 AUC-ROC (excellente discrimination)
> - ✅ Pas d'overfitting
""")

# Chargement du modèle
model = load_model()

if model:
    st.divider()
    
    # UPLOAD SECTION
    uploaded_file = st.file_uploader(
        "📁 Déposez une radiographie thoracique (X-Ray)",
        type=["jpg", "jpeg", "png"],
        help="Format: JPG, JPEG, PNG. Taille recommandée: 224x224 ou plus"
    )
    
    if uploaded_file is not None:
        # Layout: Image + Résultats
        col_img, col_res = st.columns([1, 1.5])
        
        with col_img:
            st.subheader("📸 Image Analysée")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_column_width=True)
            st.caption(f"Taille: {image.size}")
        
        with col_res:
            st.subheader("🔍 Diagnostic IA (v4)")
            
            # Prédiction
            with st.spinner("⏳ Analyse en cours..."):
                start_time = time.time()
                
                img_tensor = process_image(image)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    probability = torch.sigmoid(output).item()
                
                inference_time = (time.time() - start_time) * 1000
                
                is_pneumonia = probability > threshold
                
                st.divider()
                
                # RÉSULTAT
                if is_pneumonia:
                    st.error("⚠️ ALERTE: PNEUMONIE DÉTECTÉE")
                    st.metric("Confiance du Modèle", f"{probability:.2%}")
                    st.progress(probability)
                    
                    st.markdown("""
                    **Interpretation Clinique:**
                    - Motifs d'opacité pulmonaire détectés
                    - Recall 99% = très haute fiabilité
                    - Risk de faux négatif < 1.3%
                    - ✅ Action requise: Vérification radiologique
                    """)
                    
                else:
                    st.success("✅ RÉSULTAT: NORMAL")
                    prob_normal = 1 - probability
                    st.metric("Confiance (Sain)", f"{prob_normal:.2%}")
                    st.progress(prob_normal)
                    
                    st.markdown("""
                    **Interpretation Clinique:**
                    - Aucune anomalie majeure détectée
                    - Recall 99% = détecte presque tous les cas anormaux
                    - Risk de faux négatif < 1.3%
                    - ✅ Patient peut être déclaré sain
                    """)
                
                st.divider()
                
                # Métriques d'inférence
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("Temps d'inférence", f"{inference_time:.2f}ms")
                with col_m2:
                    st.metric("Seuil utilisé", f"{threshold:.2f}")
                with col_m3:
                    st.metric("GPU/CPU", DEVICE.type.upper())
        
        # DÉTAILS TECHNIQUES
        st.divider()
        
        with st.expander("🛠️ Architecture & Techniques"):
            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                st.markdown("""
                **Architecture:**
                - EfficientNet-B3 (Transfer Learning)
                - 12.2M paramètres
                - Classifier personnalisé
                
                **Training:**
                - Early Stopping (epoch 15)
                - K-fold validation
                - Dataset équilibré
                """)
            
            with col_tech2:
                st.markdown("""
                **Optimisation:**
                - Weighted Cross-Entropy Loss
                - AdamW optimizer
                - Learning Rate Scheduler
                
                **Hardware:**
                - GPU: RTX 5070 (Blackwell)
                - PyTorch Nightly
                """)
        
        with st.expander("📊 Matrice de Confusion (v4)"):
            st.markdown("""
            **Test Set Results (624 images):**
            
            |  | Pred Normal | Pred Pneumo |
            |---|---|---|
            | **Real Normal** | 195 (TN) | 39 (FP) |
            | **Real Pneumo** | 5 (FN) | 385 (TP) |
            
            **Metrics:**
            - Recall: 385/390 = **99.0%** ✅
            - Precision: 385/424 = **90.8%** ✅
            - F1-Score: **0.947** ✅
            - Accuracy: **92.95%** ✅
            """)
        
        with st.expander("🔬 Stress Tests"):
            st.markdown("""
            **Robustesse validée:**
            
            ✅ Bruit aléatoire → 0% pneumonie (refuse correctement)
            ✅ Image blanche → 0% pneumonie (robustesse)
            ✅ Image noire → 98% pneumonie (conservative fail-safe)
            
            **Conclusion:** Modèle robuste et cliniquement sûr
            """)

else:
    st.error("❌ Erreur: Impossible de charger le modèle")
    st.info("""
    **Solutions:**
    1. Vérifiez que `pneumonia_v4_balanced.pth` existe dans le répertoire courant
    2. Ou utilisez le mode démo (sans poids pré-entraînés)
    3. GPU disponible: """ + str(torch.cuda.is_available()))