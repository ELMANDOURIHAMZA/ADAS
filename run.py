from app import app
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  ADAS v3 — Detection Reelle avec Bounding Boxes")
    print("  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
