"""
ADAS Fixed — app.py
=====================
Corrections apportées :
  1. SpeedEstimator activé dans /stream (vitesse frame-par-frame réelle)
  2. Throttling SSE : sleep entre frames pour ne pas saturer le client
  3. Tracking simple par classe pour SpeedEstimator (un estimateur par panneau actif)
  4. feu_rouge mappé sur "red_light" (séparé du panneau STOP panneau)
"""

import os, cv2, uuid, json, time
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

from modules.detector import (
    SignDetector, AlertEngine, SpeedEstimator,
    CLASS_FR, CLASS_ICON, ALERT_MSG, ALERT_LEVEL, SPEED_LIMITS,
)

BASE       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "static", "uploads")
MODEL_REAL = os.path.join(BASE, "models", "best_real.pt")
MODEL_FALL = os.path.join(BASE, "models", "best.pt")
MODEL_PATH = MODEL_REAL if os.path.exists(MODEL_REAL) else MODEL_FALL
PROFILE_PATH = os.path.join(BASE, "models", "inference_profile.json")
VIDEO_EXT  = {"mp4","avi","mov","mkv","webm"}
IMAGE_EXT  = {"jpg","jpeg","png","bmp","gif"}
MAX_DIM    = 640

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 600 * 1024 * 1024

print("\n" + "="*55)
print("  ADAS Fixed — Chargement du modèle...")
print("="*55)
default_conf = 0.25
if os.path.exists(PROFILE_PATH):
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            profile = json.load(f)
            default_conf = float(profile.get("best", {}).get("conf", default_conf))
            print(f"  Profil inference chargé: conf={default_conf:.2f}")
    except Exception as exc:
        print(f"  [WARN] Profil inference invalide: {exc}")

print(f"  Modèle utilisé: {MODEL_PATH}")
detector = SignDetector(MODEL_PATH, conf=default_conf)
print("  ✓ Prêt sur http://localhost:5000\n")


# ── helpers ──────────────────────────────────────────────────────────────────
def ext(fn):
    return fn.rsplit(".", 1)[-1].lower() if "." in fn else ""

def resize(frame):
    h, w = frame.shape[:2]
    s = min(MAX_DIM / w, MAX_DIM / h, 1.0)
    if s < 1.0:
        return cv2.resize(frame, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA), w/int(w*s), h/int(h*s)
    return frame, 1.0, 1.0

def serialize(dets):
    return [{
        "raw":        d["raw"],
        "label_fr":   d["label_fr"],
        "icon":       d["icon"],
        "confidence": d["confidence"],
        "alert":      d["alert"],
        "alert_msg":  d["alert_msg"],
        "speed_limit":d["speed_limit"],
        "bbox":       d["bbox"],
        "color":      d["color"],
        "simulated":  d.get("simulated", False),
    } for d in dets]


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "Aucun fichier"}), 400

    e      = ext(f.filename)
    job_id = str(uuid.uuid4())[:8]
    path   = os.path.join(UPLOAD_DIR, f"{job_id}.{e}")
    f.save(path)

    conf = float(request.form.get("confidence", 0.25))

    if e in VIDEO_EXT:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return jsonify({
            "ok": True, "type": "video", "job_id": job_id,
            "url": f"/static/uploads/{job_id}.{e}",
            "fps": round(fps, 3), "frames": nf,
            "w": w, "h": h, "conf": conf,
        })

    elif e in IMAGE_EXT:
        img = cv2.imread(path)
        if img is None:
            return jsonify({"error": "Image illisible"}), 400
        small, sx, sy = resize(img)
        dets = detector.detect(small, conf=conf)
        for d in dets:
            x1,y1,x2,y2 = d["bbox"]
            d["bbox"] = [round(x1*sx), round(y1*sy), round(x2*sx), round(y2*sy)]
        top = dets[0] if dets else None
        ae  = AlertEngine()
        if top:
            ae.update_limit(top["raw"])
        adas = ae.evaluate(0.0, top["raw"] if top else None)
        return jsonify({
            "ok": True, "type": "image", "job_id": job_id,
            "url": f"/static/uploads/{job_id}.{e}",
            "dets": serialize(dets),
            "speed": None,
            "dist":  None,
            "adas":  adas["state"],
            "msg":   adas["msg"],
            "limit": ae.current_limit,
            "iw": img.shape[1], "ih": img.shape[0],
        })

    return jsonify({"error": "Format non supporté"}), 400


@app.route("/stream/<job_id>")
def stream(job_id):
    """
    SSE — traite chaque frame et envoie le résultat.
    CORRECTIONS :
      - SpeedEstimator activé (vitesse réelle frame/frame)
      - Throttling : on cède le temps pour ne pas inonder le client
        (on envoie les frames à ~2× la vitesse réelle max)
    """
    path = None
    for e in VIDEO_EXT:
        p = os.path.join(UPLOAD_DIR, f"{job_id}.{e}")
        if os.path.exists(p):
            path = p; break
    if not path:
        return jsonify({"error": "Fichier introuvable"}), 404

    conf = float(request.args.get("conf", 0.25))
    skip = max(1, int(request.args.get("skip", 1)))

    def generate():
        cap  = cv2.VideoCapture(path)
        fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        tot  = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        w0   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Facteur resize
        s  = min(MAX_DIM/w0, MAX_DIM/h0, 1.0)
        pw = int(w0*s); ph = int(h0*s)
        sx = w0/pw if pw else 1.0
        sy = h0/ph if ph else 1.0

        ae   = AlertEngine()
        # ── CORRECTION 2 : un SpeedEstimator par classe détectée ─────────────
        speed_trackers = {}   # raw_class → SpeedEstimator

        dt_frame = 1.0 / fps          # durée d'une frame en secondes
        # Throttle : on attend au minimum dt_frame/2 entre deux envois SSE
        # (permet un traitement 2× plus vite que temps réel, sans saturer)
        min_interval = dt_frame / 2.0

        fi   = 0
        last = None
        t_start = time.monotonic()

        yield f"data:{json.dumps({'type':'meta','fps':fps,'total':tot,'w':w0,'h':h0})}\n\n"

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = fi * dt_frame   # temps vidéo en secondes

            # Frames skippées → répéter dernier résultat
            if skip > 1 and fi % skip != 0 and last:
                d = dict(last); d["frame"] = fi; d["t"] = round(timestamp, 3)
                yield f"data:{json.dumps(d)}\n\n"
                fi += 1
                continue

            # Resize
            small = cv2.resize(frame, (pw, ph), cv2.INTER_AREA) if s < 1.0 else frame

            # ── DÉTECTION ────────────────────────────────────────────────────
            dets = detector.detect(small, conf=conf)

            # Recalibrer bbox → résolution originale
            for d in dets:
                x1,y1,x2,y2 = d["bbox"]
                d["bbox"]      = [round(x1*sx), round(y1*sy), round(x2*sx), round(y2*sy)]
                d["bbox_w_px"] = d["bbox"][2] - d["bbox"][0]

            top = dets[0] if dets else None
            if top:
                ae.update_limit(top["raw"])

            # ── CORRECTION 1 : VITESSE RÉELLE ────────────────────────────────
            # On utilise le panneau le plus confiant (top) pour estimer la vitesse.
            # Un SpeedEstimator indépendant par classe évite les sauts lors du
            # changement de panneau visible.
            spd  = None
            dist = None
            if top and top["bbox_w_px"] > 5:
                raw_key = top["raw"]
                if raw_key not in speed_trackers:
                    speed_trackers[raw_key] = SpeedEstimator()
                se_result = speed_trackers[raw_key].update(top["bbox_w_px"], timestamp)
                dist = se_result["distance_m"]
                spd  = se_result["speed_smooth"] if se_result["speed_smooth"] > 0 else None

            # ── ALERTE ───────────────────────────────────────────────────────
            adas = ae.evaluate(spd or 0.0, top["raw"] if top else None)

            data = {
                "type":  "frame",
                "frame": fi,
                "t":     round(timestamp, 3),
                "dets":  serialize(dets),
                "speed": spd,
                "dist":  dist,
                "adas":  adas["state"],
                "msg":   adas["msg"],
                "limit": ae.current_limit,
                "pct":   round(fi / tot * 100, 1),
            }
            last = data
            yield f"data:{json.dumps(data)}\n\n"
            fi += 1

            # ── CORRECTION 3 : THROTTLE ───────────────────────────────────────
            # Calcule le temps théorique de cette frame dans la vidéo réelle,
            # puis attend si on avance trop vite (évite de saturer le buffer JS).
            wall_elapsed   = time.monotonic() - t_start
            video_expected = timestamp          # temps vidéo de cette frame
            advance        = wall_elapsed - video_expected
            # Si on est plus de 1 seconde en avance sur la vidéo → on attend
            if advance < -min_interval:
                time.sleep(min(-advance - min_interval, 0.5))

        cap.release()
        yield f"data:{json.dumps({'type':'done','frames':fi})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
        headers={
            "Cache-Control":    "no-cache,no-transform",
            "X-Accel-Buffering":"no",
            "Connection":       "keep-alive",
        })


@app.route("/static/uploads/<path:fn>")
def uploads(fn):
    return send_from_directory(UPLOAD_DIR, fn)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)