# 🚗 ADAS v6 — Advanced Driver Assistance System

![Version](https://img.shields.io/badge/version-6.0-blue)
![Team](https://img.shields.io/badge/team-3%20students-green)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Approche: **Notebook-Centric**

```
📔 notebooks/ADAS_v6_kaggle.ipynb
      ↓ (Kaggle Notebook + GPU T4 x2 + Internet)
      ├─ Télécharge datasets (LISA, GTSDB, DS1-3)
      ├─ Crée data/processed/dataset/
      ├─ Lance training 120 epochs
      └─ Génère models/best.pt
```

**Important:** Les données `data/raw/` et `data/processed/` ne sont **PAS committées** au Git.  
Seul le notebook est la source.

---

```
Collecte Dataset       Training YOLO v8s          Déploiement Web
(LISA, GTSDB, DS1-3)  (120 epochs, mAP50>0.75)   (Flask + SSE + Dashboard)
      ↓                        ↓                           ↓
  Étudiant 1        →     Étudiant 2         →     Étudiant 3
```

---

## 🎯 Fonctionnalités Principales

### ✅ Détection Temps Réel
- **15 classes** panneaux français + feux tricolores
- Modèle **YOLOv8s** optimisé (mAP50 > 0.75)
- **SSE streaming** vidéo frame-par-frame

### 🚗 Tableau de Bord HUD
- Interface type **dashcam automobile**
- Affichage vitesse calibrée (estimée)
- Alertes dynamiques (danger/warning/ok)
- Légende 15 classes interactive

### 📍 Calibration Vitesse
- **CalibratedSpeedEstimator** (estimation robuste)
- Focale caméra + largeur panneau = vitesse km/h
- Filtrage outliers + lissage EMA
- Retour `None` si calibration insuffisante

### ⚠️ Système Alertes ADAS
- **Critère:** Feu rouge, STOP, Entrée interdite
- **Warning:** Dépassement interdit, excès vitesse
- **Safe:** Feu vert, vitesse correcte
- **Info:** Vitesse non calibrée

---

## 📊 Architecture

```
Frontend (HTML/CSS/JS)
    ↓↑ SSE streaming
    
Backend (Flask API)
    ├─ POST /upload     (traite fichier)
    ├─ GET /stream     (SSE frame-by-frame)
    └─ GET /          (dashboard HUD)
    ↓
Modules Détection
    ├─ SignDetector     (YOLO v8s + best.pt)
    ├─ CalibratedSpeedEstimator (calcul vitesse)
    └─ AlertEngine      (logique alertes)
```

---

## 🚀 Démarrage Rapide

### **1. Installation**
```bash
git clone <repo_url>
cd adas_v3
pip install -r requirements.txt
```

### **2. Copier Modèle**
```bash
# Mettre best.pt dans:
cp /chemin/to/best.pt models/
```

### **3. Lancer API**
```bash
python run.py
# → http://localhost:5000
```

### **4. Upload Vidéo/Image**
- Glisser fichier sur la zone drop
- Cliquer "Analyser"
- Observer détections + alerte en live

---

## 📁 Structure Projet

```
adas_v3/
├── 📄 README.md                    ← Vous êtes ici
├── 📄 COORDINATION.md              (coordination équipe)
├── 📄 PROJECT_STRUCTURE.md         (détail arborescence)
├── 📄 CONTRIBUTING.md              (règles Git)
│
├── 📁 docs/
│   ├── ETUDIANT_1_README.md        ← Dataset & Prep
│   ├── ETUDIANT_2_README.md        ← Model Training
│   └── ETUDIANT_3_README.md        ← Backend & Deploy
│
├── 📁 data/
│   ├── raw/                        (données brutes)
│   ├── processed/dataset/          (YOLO-ready) ★
│   └── statistics/                 (rapports audit)
│
├── 📁 notebooks/
│   └── ADAS_v6_kaggle.ipynb       (dataset + training complet)
│
├── 📁 models/
│   ├── best.pt                    (modèle optimisé) ★
│   └── training_logs/             (logs training)
│
├── 📁 modules/
│   └── detector.py                (détection + vitesse + alerte)
│
├── 📁 templates/
│   └── index.html                 (dashboard HUD)
│
├── 📁 tests/
│   ├── test_detector.py
│   ├── test_speed_estimator.py
│   └── test_api.py
│
├── 🐍 app.py                       (API Flask)
├── 🐍 run.py                       (launcher)
└── 📄 requirements.txt
```

**Voir [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) pour détail complet.**

---

## 👥 Équipe & Responsabilités

### **Étudiant 1 — Dataset & Preprocessing**
**Tâches:** Collecte (LISA, GTSDB, DS1-3), normalisation, audit

- ✅ T1.1 Collecte & intégration données
- ✅ T1.2 Nettoyage annotations YOLO
- ✅ T1.3 Augmentation & balance dataset
- ✅ T1.4 Audit qualité

📍 **Voir [ETUDIANT_1_README.md](docs/ETUDIANT_1_README.md)**

---

### **Étudiant 2 — Model Training & Optimization**
**Tâches:** Config YOLO, 120 epochs, best.pt

- ✅ T2.1 Pipeline training (config + hyperparams)
- ✅ T2.2 Entraînement & monitoring
- ✅ T2.3 Évaluation & sélection best.pt
- ✅ T2.4 Tuning optimisation

📍 **Voir [ETUDIANT_2_README.md](docs/ETUDIANT_2_README.md)**

---

### **Étudiant 3 — Backend Web & Real-time Detection**
**Tâches:** API Flask, calibration vitesse, dashboard HUD

- ✅ T3.1 API Flask & SSE streaming
- ✅ T3.2 Calibration vitesse & AlertEngine
- ✅ T3.3 Dashboard UI (tableau de bord HUD)
- ✅ T3.4 Tests intégration & déploiement

📍 **Voir [ETUDIANT_3_README.md](docs/ETUDIANT_3_README.md)**

---

**Coordination complète:** [COORDINATION.md](COORDINATION.md)

---

## 📊 Performances

| Métrique | Valeur |
|---|---|
| **Modèle** | YOLOv8s (45 MB) |
| **mAP50** | > 0.75 ✅ |
| **FPS Détection** | ~30 FPS (GPU T4) |
| **Latence SSE** | < 100ms |
| **Classes** | 15 (panneaux français) |
| **Dataset** | 7381 train + 1000 val |

---

## 🛠️ Technologies

```
Backend:
  - Flask 2.3+        (API REST)
  - Ultralytics YOLO  (détection)
  - OpenCV 4.8+       (vidéo/image)
  - NumPy + Pandas    (data processing)

Frontend:
  - HTML5 + CSS3      (responsive HUD)
  - JavaScript ES6+   (SSE client)
  - Chart.js          (graphiques vitesse)

Infrastructure:
  - Python 3.9+
  - GPU: CUDA/PyTorch
  - Docker (optional)
```

---

## 🧪 Tests

```bash
# Lancer suite complète
pytest tests/ -v

# Test détecteur
pytest tests/test_detector.py

# Test vitesse calibrée
pytest tests/test_speed_estimator.py

# Test alertes
pytest tests/test_alert_engine.py

# Test API
pytest tests/test_api.py
```

---

## 🤝 Collaboration & Git

**Règles strictes:**
1. Branch par étudiant: `etudiant_X_task`
2. PR format: `[T1.1] Description`
3. Code review obligatoire (≥1 peer)
4. Commit message: `feat/fix/docs(scope): subject`

**Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour détails complets.**

---

## 📝 Dépendances

```bash
# Installer
pip install -r requirements.txt

# Ou manuellement:
pip install ultralytics flask opencv-python torch torchvision
pip install roboflow pyyaml pandas matplotlib tensorboard pytest
```

**Voir [requirements.txt](requirements.txt)**

---

## ⚙️ Configuration

### **app.py — Modèle Priorité**
```python
MODEL_USER = "models/best_adas_v5.pt"      # v5 (si existe)
MODEL_REAL = "models/best_real.pt"         # v6 alternative
MODEL_FALL = "models/best.pt"              # ✅ Défaut v6
```

### **modules/detector.py — Classes**
```python
CLASS_NAMES = [
    'vitesse_20', 'vitesse_30', 'vitesse_50', 'vitesse_60',
    'vitesse_70', 'vitesse_80', 'vitesse_100', 'vitesse_120',
    'depassement_interdit', 'stop', 'sens_interdit', 'entree_interdite',
    'feu_vert', 'feu_rouge', 'feu_orange'
]
```

### **templates/index.html — Calibration Vitesse**
```javascript
// UI Controls:
#spd-enable       (toggle activation)
#focal           (focale caméra px — défaut 900)
#signw           (largeur panneau m — défaut 0.60)
```

---

## 📺 Utilisation Dashboard

### **Interface**
```
┌─────────────────────────────────────────────────────────┐
│  ADAS v6                                    [🟢 ACTIF]  │
├─────────────────────────────────────────────────────────┤
│
│ [📁 UPLOAD]    │  [🎯 VIDEO + CANVAS + ALERT]  │ [📊 ACTIVE]
│                │                                │ [DETS]
│ Conf: ▓▓▓░░   │                                │
│ Skip: ▓░░░░   │   [HUD Panel]                  │ # Obj: 5
│                │   ├─ KM/H: 45                 │
│ ⚙️ CALIB:      │   ├─ DIST: 12.5m              │ vitesse_30
│ ☑ Activated   │   ├─ LIMITE: 30               │ 92%
│ Focal: 900    │   └─ [ALERTE MSG]              │
│ Signw: 0.60   │                                │ stop
│                │   [SPEED GRAPH]                │ 78%
│ ▶ ANALYSER    │   [progress]                   │
│                │                                │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Déploiement Production (Optional)

```bash
# Gunicorn + SSL
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Docker
docker build -t adas_v6 .
docker run -p 5000:5000 adas_v6

# Nginx reverse proxy (si besoin)
server {
    listen 80;
    location / {
        proxy_pass http://localhost:5000;
    }
}
```

---

## 📖 Documentation Complète

- **Dataset & Preprocessing:** [ETUDIANT_1_README.md](docs/ETUDIANT_1_README.md)
- **Model Training:** [ETUDIANT_2_README.md](docs/ETUDIANT_2_README.md)
- **Backend & Deploy:** [ETUDIANT_3_README.md](docs/ETUDIANT_3_README.md)
- **Coordination Équipe:** [COORDINATION.md](COORDINATION.md)
- **Structure Détaillée:** [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Git Workflow:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Routes:** [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) (soon)

---

## 🎓 Détails Académiques

**Master:** Analyse de Données & IA  
**Équipe:** 3 Étudiants  
**Durée:** [Indiquer durée du projet]  
**Professeur:** [Nom Professeur]  

---

## 📊 Métriques Dataset

```
LISA (Feux):           Y images → 12, 13, 14
GTSDB (Panneaux):      Z images → 0-11
DS1/DS2/DS3 (Mix):     W images → 15 classes

Total:
├─ Train:  7381 images | 20271 bboxes
├─ Val:    1000 images | 2500 bboxes
└─ 15 classes normalisées
```

---

## ⚠️ Notes Importantes

1. **best.pt obligatoire** — À placer dans `models/` pour déploiement
2. **Vitesse calibrée** — Optionnelle, retourne `None` si non calibrée
3. **SSE Throttling** — Limité à ~2× vitesse réelle vidéo
4. **GPU Memory** — ~4.7GB détection + streaming
5. **Cleanup** — Fichiers uploads > 1h supprimés automatiquement

---

## 🔗 Liens Ressources

- [YOLO v8 Docs](https://docs.ultralytics.com/models/yolov8/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Python](https://opencv-python-tutroals.readthedocs.io/)
- [GitHub SSH Setup](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

---

## 📄 License

MIT License — See LICENSE file

---

## ✅ Status

- ✅ Dataset pipeline complete
- ✅ Model training ready
- ✅ Backend API deployed
- ✅ Dashboard UI live
- ✅ Tests passing
- 🟢 **Production Ready**

---

**Dernière mise à jour:** 2026-04-30  
**Version:** 6.0  
**Status:** 🟢 Active & Maintainable

---

## 📞 Questions?

- 📍 GitHub Issues
- 💬 Team Discord/Slack
- 📧 Contact Professeur

**Bonne collaboration équipe! 🚀**
