"""
detection.py — Détecteur de panneaux ADAS avec bounding boxes
Gère la détection YOLO, le filtrage des classes et la construction des résultats.
"""

import os
import numpy as np

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from .constants import MODEL_NAME_TO_RAW, ALLOWED_RAW_CLASSES, CLASS_FR, CLASS_ICON, ALERT_LEVEL, ALERT_MSG, SPEED_LIMITS, CLASS_COLOR, SPEED_COLOR


class SignDetector:
    """Détecteur YOLO pour panneaux de signalisation routière."""
    
    def __init__(self, model_path: str, conf: float = 0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle introuvable : {model_path}")

        self.model    = YOLO(model_path)
        self.conf_thr = conf
        self.names    = self.model.names   # dict {idx: name}

        # ── Déterminer le mode (détection vs classification) ────────────────
        task = getattr(self.model, "task", None)
        if task is None:
            inner = getattr(self.model, "model", None)
            cname = type(inner).__name__ if inner else ""
            task  = "detect" if "Detection" in cname else "classify"

        self._detect_mode = task in ("detect", "segment")

        mode = "DÉTECTION" if self._detect_mode else "CLASSIFICATION"
        print(f"[ADAS v6] Modèle prêt — {len(self.names)} classes — task={task} — mode={mode}")
        if not self._detect_mode:
            print("[ADAS v6][WARN] Modèle en mode CLASSIFICATION : bboxes simulées !")

    def set_conf(self, v: float):
        """Change le seuil de confiance."""
        self.conf_thr = max(0.05, min(0.95, float(v)))

    def _resolve_conf(self, conf: float = None) -> float:
        """Valide le seuil de confiance."""
        if conf is None:
            return self.conf_thr
        return max(0.05, min(0.95, float(conf)))

    def _raw_from_class_idx(self, cls_idx: int) -> str:
        """Résout l'ID brut ADAS à partir de l'indice de classe YOLO."""
        label = (self.names.get(cls_idx)
                 if isinstance(self.names, dict)
                 else self.names[cls_idx])
        key = str(label).strip().lower()
        return MODEL_NAME_TO_RAW.get(key, key)

    def detect(self, frame: np.ndarray, conf: float = None) -> list:
        """
        Détecte les panneaux dans une image.
        Retourne une liste de dictionnaires avec les détections.
        """
        try:
            conf_thr = self._resolve_conf(conf)
            res = self.model(frame, verbose=False, conf=conf_thr)[0]
            if self._detect_mode:
                return self._from_boxes(res, frame, conf_thr)
            else:
                return self._from_probs(res, frame, conf_thr)
        except Exception as exc:
            print(f"[ADAS][WARN] Erreur détection : {exc}")
            return []

    def _from_boxes(self, res, frame, conf_thr: float) -> list:
        """Traite les résultats en mode détection (vraies bboxes)."""
        if res.boxes is None or not len(res.boxes):
            return []
        h, w = frame.shape[:2]
        out  = []
        # Trier par confiance décroissante
        indices = sorted(range(len(res.boxes)),
                         key=lambda i: float(res.boxes.conf[i]), reverse=True)
        for i in indices:
            conf = float(res.boxes.conf[i])
            if conf < conf_thr:
                continue
            cls = int(res.boxes.cls[i])
            raw = self._raw_from_class_idx(cls)
            if raw not in ALLOWED_RAW_CLASSES:
                continue
            x1, y1, x2, y2 = [float(v) for v in res.boxes.xyxy[i]]
            out.append(self._build(raw, conf,
                                   max(0, x1), max(0, y1),
                                   min(w, x2), min(h, y2), w, h))
        return out

    def _from_probs(self, res, frame, conf_thr: float) -> list:
        """Traite les résultats en mode classification (fallback avec bboxes simulées)."""
        if res.probs is None:
            return []
        h, w = frame.shape[:2]
        out  = []
        top5_idx  = res.probs.top5
        top5_conf = res.probs.top5conf.tolist()
        for rank, (idx, conf) in enumerate(zip(top5_idx, top5_conf)):
            if float(conf) < conf_thr:
                break
            if rank > 0 and float(conf) < 0.40:
                break
            raw    = self._raw_from_class_idx(int(idx))
            if raw not in ALLOWED_RAW_CLASSES:
                continue
            scale  = 0.70 - rank * 0.12
            margin = (1.0 - scale) / 2
            offset = rank * 0.06
            x1 = w * (margin + offset)
            y1 = h * (margin + offset)
            x2 = min(w - 1, x1 + w * scale)
            y2 = min(h - 1, y1 + h * scale)
            out.append(self._build(raw, float(conf), x1, y1, x2, y2, w, h))
        return out

    def _build(self, raw, conf, x1, y1, x2, y2, W, H) -> dict:
        """Construit un dictionnaire de détection."""
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
