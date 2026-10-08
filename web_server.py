import os
import sys
import uvicorn

# Ensure the backend directory is in the path
proj_root = os.path.dirname(os.path.abspath(__file__))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

if __name__ == "__main__":
    print("==================================================")
    print("EXAMINATION MONITORING SYSTEM STARTING...")
    print("==================================================")
    
    # Launch the FastAPI app from backend/server.py
    # Using host 0.0.0.0 and port 8000
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=False)
