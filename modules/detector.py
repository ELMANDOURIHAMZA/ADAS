"""
detector.py — ADAS Final
=========================
Mapping robuste des classes a partir des noms du modele.
Compatible modele de classification ET detection.
"""

import os, time, collections
import numpy as np

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

# Map robuste nom_modele -> id brut utilise par les regles ADAS
MODEL_NAME_TO_RAW = {
    # Vitesse
    "0": "0", "vitesse_20": "0", "speed_20": "0",
    "1": "1", "vitesse_30": "1", "speed_30": "1",
    "2": "2", "vitesse_50": "2", "speed_50": "2",
    "3": "3", "vitesse_60": "3", "speed_60": "3",
    "4": "4", "vitesse_70": "4", "speed_70": "4",
    "5": "5", "vitesse_80": "5", "speed_80": "5",
    "7": "7", "vitesse_100": "7", "speed_100": "7",
    "8": "8", "vitesse_120": "8", "speed_120": "8",
    # Interdictions / priorites
    "9": "9", "depassement_interdit": "9", "no_overtaking": "9",
    "10": "10", "camion_depassement_interdit": "10", "truck_no_overtaking": "10",
    "14": "14", "stop": "14", "stop_sign": "14", "stop_panneau": "14",
    "15": "15", "sens_interdit": "15", "no_entry_direction": "15",
    "16": "16", "camion_interdit": "16", "truck_forbidden": "16",
    "17": "17", "entree_interdite": "17", "entry_forbidden": "17",
    "26": "26", "feux_signalisation": "26", "traffic_lights": "26",
    # Feux
    "go": "go", "feu_vert": "go", "green_light": "go",
    "feu_rouge": "stop", "red_light": "stop",
    "warning": "warning", "feu_orange": "warning", "yellow_light": "warning",
}

CLASS_FR = {
    "0":"Limite 20 km/h",   "1":"Limite 30 km/h",   "2":"Limite 50 km/h",
    "3":"Limite 60 km/h",   "4":"Limite 70 km/h",   "5":"Limite 80 km/h",
    "7":"Limite 100 km/h",  "8":"Limite 120 km/h",
    "9":"Dépassement interdit", "10":"Camion: dép. interdit",
    "14":"STOP",            "15":"Sens interdit",
    "16":"Camion interdit", "17":"Entrée interdite",
    "26":"Feux signalisation",
    "go":"Feu Vert",  "stop":"Feu Rouge",  "warning":"Feu Orange",
}
CLASS_ICON = {
    "0":"🔵","1":"🔵","2":"🔵","3":"🔵","4":"🔵",
    "5":"🔵","7":"🔵","8":"🔵",
    "9":"🚫","10":"🚫",
    "14":"🛑","15":"⛔","16":"🚫","17":"⛔",
    "26":"🚦","go":"🟢","stop":"🔴","warning":"🟠",
}
ALERT_LEVEL = {
    "stop":"critical","14":"critical","15":"critical","17":"critical",
    "warning":"warning","26":"warning","9":"warning","10":"warning",
    "go":"safe",
}
ALERT_MSG = {
    "critical": "⛔ DANGER — Arrêt/Accès interdit",
    "warning":  "⚠️ Attention — Ralentir",
    "safe":     "✅ Voie libre — Avancez",
    "info":     "ℹ️ Panneau détecté",
}
SPEED_LIMITS = {
    "0":20,"1":30,"2":50,"3":60,"4":70,"5":80,"7":100,"8":120,
}
# Couleurs hex pour le frontend canvas
CLASS_COLOR = {
    "go":"#00e87a","stop":"#ff1e3c","warning":"#ff8800",
    "14":"#ff1e3c","15":"#ff3355","17":"#ff3355",
    "9":"#ff6600","10":"#ff6600","16":"#ff6600","26":"#ffd600",
}
SPEED_COLOR = "#00d4ff"


# ══════════════════════════════════════════════════════════════
# SPEED ESTIMATOR  (sténopé, frame-par-frame)
# ══════════════════════════════════════════════════════════════
class SpeedEstimator:
    FOCAL   = 700.0    # px
    REAL_W  = 0.60     # m (largeur réelle d'un panneau standard)
    SMOOTH  = 7        # fenêtre de lissage

    def __init__(self):
        self._buf    = collections.deque(maxlen=self.SMOOTH)
        self._prev_z = None
        self._prev_t = None

    def update(self, bbox_w_px: float, timestamp: float) -> dict:
        Z = (self.FOCAL * self.REAL_W / bbox_w_px) if bbox_w_px > 1 else float("inf")
        out = {"distance_m": round(Z, 2) if Z < 9999 else None,
               "speed_smooth": 0.0}
        if self._prev_z is not None and self._prev_t is not None:
            dt = timestamp - self._prev_t
            if 0 < dt < 5 and Z < 9999 and self._prev_z < 9999:
                v = min(abs(self._prev_z - Z) / dt * 3.6, 300.0)
                self._buf.append(v)
                out["speed_smooth"] = round(sum(self._buf) / len(self._buf), 1)
        self._prev_z = Z
        self._prev_t = timestamp
        return out

    def reset(self):
        self._prev_z = None; self._prev_t = None; self._buf.clear()


# ══════════════════════════════════════════════════════════════
# ALERT ENGINE
# ══════════════════════════════════════════════════════════════
class AlertEngine:
    def __init__(self):
        self._limit = 50

    @property
    def current_limit(self): return self._limit

    def update_limit(self, raw: str):
        if raw in SPEED_LIMITS:
            self._limit = SPEED_LIMITS[raw]

    def evaluate(self, speed: float, sign: str = None) -> dict:
        if sign in ("stop","14","15","17"):
            return {"state":"danger",    "msg":"⛔ ARRÊT OBLIGATOIRE !"}
        ex = speed - self._limit
        if ex > 25: return {"state":"overspeed","msg":f"🚨 EXCÈS +{ex:.0f} km/h !"}
        if ex >  5: return {"state":"warning",  "msg":f"⚠️ Ralentir +{ex:.0f} km/h"}
        return         {"state":"ok",        "msg":f"✅ Vitesse correcte"}


# ══════════════════════════════════════════════════════════════
# SIGN DETECTOR
# ══════════════════════════════════════════════════════════════
class SignDetector:
    def __init__(self, model_path: str, conf: float = 0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.model    = YOLO(model_path)
        self.conf_thr = conf
        self.names    = self.model.names

        task = getattr(self.model, "task", "classify")
        self._detect_mode = task in ("detect", "segment")

        mode = "DÉTECTION" if self._detect_mode else "CLASSIFICATION"
        print(f"[ADAS] Modèle prêt — {len(self.names)} classes — mode={mode}")

    def set_conf(self, v: float):
        self.conf_thr = max(0.05, min(0.95, float(v)))

    def _resolve_conf(self, conf: float = None) -> float:
        if conf is None:
            return self.conf_thr
        return max(0.05, min(0.95, float(conf)))

    def _raw_from_class_idx(self, cls_idx: int) -> str:
        label = self.names.get(cls_idx) if isinstance(self.names, dict) else self.names[cls_idx]
        key = str(label).strip().lower()
        return MODEL_NAME_TO_RAW.get(key, key)

    # ── Entrée principale ──────────────────────────────────────
    def detect(self, frame: np.ndarray, conf: float = None) -> list:
        try:
            conf_thr = self._resolve_conf(conf)
            res = self.model(frame, verbose=False, conf=conf_thr)[0]
            if self._detect_mode:
                return self._from_boxes(res, frame, conf_thr)
            else:
                return self._from_probs(res, frame, conf_thr)
        except Exception as exc:
            print(f"[ADAS][WARN] {exc}")
            return []

    # ── Mode détection (vraies bboxes) ────────────────────────
    def _from_boxes(self, res, frame, conf_thr: float) -> list:
        if res.boxes is None or not len(res.boxes):
            return []
        h, w = frame.shape[:2]
        out  = []
        for i in sorted(range(len(res.boxes)),
                        key=lambda i: float(res.boxes.conf[i]), reverse=True):
            conf = float(res.boxes.conf[i])
            if conf < conf_thr: continue
            cls  = int(res.boxes.cls[i])
            raw  = self._raw_from_class_idx(cls)
            x1,y1,x2,y2 = [float(v) for v in res.boxes.xyxy[i]]
            out.append(self._build(raw, conf,
                                   max(0,x1),max(0,y1),min(w,x2),min(h,y2),w,h))
        return out

    # ── Mode classification (top-k avec bbox simulée) ─────────
    def _from_probs(self, res, frame, conf_thr: float) -> list:
        if res.probs is None:
            return []
        h, w = frame.shape[:2]
        out  = []

        top5_idx  = res.probs.top5          # liste de 5 indices
        top5_conf = res.probs.top5conf.tolist()

        for rank, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            if float(conf) < conf_thr:
                break
            # N'afficher que top-1 (ou top-2 si conf > 0.4)
            if rank > 0 and float(conf) < 0.40:
                break

            raw = self._raw_from_class_idx(int(idx))

            # Bbox simulée : plus petite pour les rangs suivants
            scale  = 0.70 - rank * 0.12
            margin = (1.0 - scale) / 2
            offset = rank * 0.06

            x1 = w * (margin + offset)
            y1 = h * (margin + offset)
            x2 = x1 + w * scale
            y2 = y1 + h * scale
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            out.append(self._build(raw, float(conf), x1, y1, x2, y2, w, h))

        return out

    # ── Constructeur de dictionnaire de détection ─────────────
    def _build(self, raw, conf, x1, y1, x2, y2, W, H) -> dict:
        alert = ALERT_LEVEL.get(raw, "info")
        bw    = x2 - x1
        return {
            "raw":        raw,
            "label_fr":   CLASS_FR.get(raw, raw),
            "icon":       CLASS_ICON.get(raw, "📍"),
            "confidence": round(conf * 100, 1),
            "alert":      alert,
            "alert_msg":  ALERT_MSG[alert],
            "speed_limit":SPEED_LIMITS.get(raw),
            "bbox":       [round(x1), round(y1), round(x2), round(y2)],
            "bbox_w_px":  bw,
            "cx":         (x1 + x2) / 2,
            "cy":         (y1 + y2) / 2,
            "color":      CLASS_COLOR.get(raw, SPEED_COLOR),
            "simulated":  not self._detect_mode,
        }