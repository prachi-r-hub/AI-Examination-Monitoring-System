import os
import json
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "scratch", "db.json")
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

def load_db():
    if not os.path.exists(DB_PATH):
        return {"history": [], "evidence": []}
    try:
        with open(DB_PATH, "r") as f:
            return json.load(f)
    except:
        return {"history": [], "evidence": []}

def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)

def add_history(source_type, source_details, restricted_count, status):
    data = load_db()
    data["history"].insert(0, {
        "timestamp": time.time(),
        "source_type": source_type,
        "source_details": source_details,
        "restricted_count": restricted_count,
        "status": status
    })
    save_db(data)

def add_evidence(source_type, object_name, category, confidence, image_filename):
    data = load_db()
    data["evidence"].insert(0, {
        "timestamp": time.time(),
        "source_type": source_type,
        "object_name": object_name,
        "category": category,
        "confidence": confidence,
        "image_url": f"/evidence_files/{image_filename}"
    })
    save_db(data)
