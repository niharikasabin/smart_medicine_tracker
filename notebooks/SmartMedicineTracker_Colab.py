# ============================================================
# SmartMedicineTracker_Colab.ipynb
# ============================================================
# Open this as a .ipynb in Google Colab.
# Set Runtime → Change runtime type → T4 GPU
# Then run cells top-to-bottom.
# ============================================================
# To convert this to .ipynb:
#   pip install jupytext
#   jupytext --to notebook notebooks/colab_setup.py
# ============================================================

# %% [markdown]
# # 💊 Smart Medicine Adherence Tracker
# ### Google Colab Setup & Training Notebook
# **Runtime: GPU (T4 recommended)**  
# Go to: Runtime → Change runtime type → Hardware accelerator: GPU

# %% [markdown]
# ## Step 1: Install Dependencies

# %%
# Install all required packages
get_ipython().system('pip install ultralytics torch torchvision opencv-python-headless pillow numpy pandas scikit-learn matplotlib seaborn plotly streamlit tqdm PyYAML -q')
print("✅ Dependencies installed")

# %%
# Check GPU availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Memory:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## Step 2: Mount Google Drive & Set Up Project

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import os, shutil
from pathlib import Path

# Create project directory
PROJECT_DIR = "/content/smart_medicine_tracker"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)

# Google Drive paths
DRIVE_BASE    = "/content/drive/MyDrive/MedicineTracker"
DRIVE_DATASET = f"{DRIVE_BASE}/dataset"
DRIVE_MODELS  = f"{DRIVE_BASE}/models"

# Create Drive directories
for d in [DRIVE_BASE, DRIVE_DATASET, DRIVE_MODELS]:
    os.makedirs(d, exist_ok=True)

print(f"✅ Project dir: {PROJECT_DIR}")
print(f"✅ Drive dir:   {DRIVE_BASE}")

# %% [markdown]
# ## Step 3: Upload / Download Dataset
#
# **Option A: Upload from local machine**
# ```python
# from google.colab import files
# uploaded = files.upload()  # Upload your zip file
# ```
#
# **Option B: Download from Roboflow (recommended)**
# Replace API_KEY with your Roboflow API key from roboflow.com

# %%
# OPTION B: Download from Roboflow
# Uncomment and fill in your API key + dataset details

# !pip install roboflow -q
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY_HERE")
# project = rf.workspace("roboflow-universe").project("pill-detection")
# dataset = project.version(1).download("yolov8", location=f"{PROJECT_DIR}/dataset")
# print("✅ Dataset downloaded from Roboflow")

# %%
# OPTION C: Generate a simple synthetic dataset for testing
# (not real medicine images — only for verifying pipeline works)

import cv2
import numpy as np
import yaml
from pathlib import Path

def create_synthetic_dataset(base_dir, n_train=50, n_val=15, n_test=10):
    """
    Create a minimal synthetic dataset (colored rectangles = 'medicines').
    FOR TESTING PIPELINE ONLY — use real dataset for actual training.
    """
    classes = ["pill_strip", "pill_bottle", "tablet", "medicine_box"]
    colors  = [(0,200,100), (255,140,0), (0,150,255), (200,50,200)]
    
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        img_dir = Path(base_dir) / split / "images"
        lbl_dir = Path(base_dir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(n):
            # Random background
            img = np.random.randint(200, 230, (640, 640, 3), dtype=np.uint8)
            
            # Place 1-3 random "medicine" rectangles
            labels = []
            for _ in range(np.random.randint(1, 4)):
                cls_id = np.random.randint(0, 4)
                color  = colors[cls_id]
                w, h   = np.random.randint(80, 200), np.random.randint(50, 120)
                x1     = np.random.randint(0, 640 - w)
                y1     = np.random.randint(0, 640 - h)
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), color, -1)
                cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0,0,0), 2)
                
                # YOLO label (normalized)
                cx = (x1 + w/2) / 640
                cy = (y1 + h/2) / 640
                nw = w / 640
                nh = h / 640
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            
            cv2.imwrite(str(img_dir / f"img_{i:04d}.jpg"), img)
            with open(lbl_dir / f"img_{i:04d}.txt", "w") as f:
                f.write("\n".join(labels))
        
        print(f"  ✅ {split}: {n} images")
    
    # Write YAML
    yaml_content = {
        "path":  base_dir,
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    4,
        "names": {0:"pill_strip", 1:"pill_bottle", 2:"tablet", 3:"medicine_box"},
    }
    with open(f"{base_dir}/medicine.yaml", "w") as f:
        yaml.dump(yaml_content, f)
    
    print(f"\n✅ Synthetic dataset at: {base_dir}/medicine.yaml")

dataset_path = f"{PROJECT_DIR}/dataset"
create_synthetic_dataset(dataset_path)

# %% [markdown]
# ## Step 4: Train YOLOv8

# %%
from ultralytics import YOLO

# Load pretrained YOLOv8 nano (fastest) — change to yolov8s.pt for better accuracy
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data    = f"{PROJECT_DIR}/dataset/medicine.yaml",
    epochs  = 30,          # Increase to 100 for real dataset
    imgsz   = 640,
    batch   = 16,
    device  = 0 if torch.cuda.is_available() else "cpu",
    project = f"{PROJECT_DIR}/models/runs",
    name    = "medicine_detect",
    patience= 10,
    augment = True,
    plots   = True,
    verbose = True,
)

print(f"\n✅ Training complete!")
print(f"mAP50:    {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
print(f"mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

# %%
# Save best model to Google Drive
best_pt = f"{PROJECT_DIR}/models/runs/medicine_detect/weights/best.pt"
if Path(best_pt).exists():
    shutil.copy(best_pt, f"{DRIVE_MODELS}/medicine_yolo.pt")
    shutil.copy(best_pt, f"{PROJECT_DIR}/models/medicine_yolo.pt")
    print(f"✅ Model saved to Drive: {DRIVE_MODELS}/medicine_yolo.pt")
else:
    print("⚠️ best.pt not found — check training output")

# %% [markdown]
# ## Step 5: Test Inference

# %%
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
trained_model = YOLO(f"{PROJECT_DIR}/models/medicine_yolo.pt")

# Test on a sample image from test set
test_imgs = list(Path(f"{PROJECT_DIR}/dataset/test/images").glob("*.jpg"))
if test_imgs:
    img_path = str(test_imgs[0])
    results = trained_model(img_path, conf=0.45)
    
    # Display result
    annotated = results[0].plot()
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated[:,:,::-1])   # BGR → RGB
    plt.axis("off")
    plt.title("YOLOv8 Medicine Detection Result")
    plt.show()
    
    print(f"Detections: {len(results[0].boxes)}")
    for box in results[0].boxes:
        cls_name = trained_model.names[int(box.cls)]
        conf     = float(box.conf)
        print(f"  {cls_name}: {conf:.2%}")

# %% [markdown]
# ## Step 6: Set Up Database & Seed Data

# %%
# Upload project files from Drive or recreate database module inline
# For Colab: create database inline

import sqlite3, json, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = f"{PROJECT_DIR}/data/adherence.db"
Path(f"{PROJECT_DIR}/data").mkdir(exist_ok=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Create schema
with get_conn() as conn:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, email TEXT UNIQUE
        );
        CREATE TABLE IF NOT EXISTS medicines (
            medicine_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, name TEXT, type TEXT DEFAULT 'tablet',
            dose_times TEXT DEFAULT '["08:00","20:00"]', is_active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS adherence_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER, medicine_id INTEGER, medicine_name TEXT,
            timestamp TEXT, taken INTEGER, confidence REAL DEFAULT 1.0, notes TEXT
        );
    """)

# Seed realistic data
np.random.seed(42)

with get_conn() as conn:
    conn.execute("INSERT OR IGNORE INTO users (name, email) VALUES ('Alice', 'alice@test.com')")
    uid = conn.execute("SELECT user_id FROM users WHERE email='alice@test.com'").fetchone()[0]
    conn.execute("INSERT INTO medicines (user_id, name, type) VALUES (?,?,?)", (uid,"Metformin 500mg","tablet"))
    mid = conn.lastrowid

    base = datetime.now() - timedelta(days=60)
    for day in range(60):
        date = base + timedelta(days=day)
        weekday = date.weekday()
        prob = 0.90 if weekday < 5 else 0.72
        for hour in [8, 20]:
            ts     = date.replace(hour=hour, minute=np.random.randint(0, 20))
            taken  = 1 if np.random.random() < prob else 0
            conn.execute(
                "INSERT INTO adherence_logs (user_id, medicine_id, medicine_name, timestamp, taken) VALUES (?,?,?,?,?)",
                (uid, mid, "Metformin 500mg", ts.isoformat(sep=' ', timespec='seconds'), taken)
            )

print(f"✅ Database seeded at: {DB_PATH}")

# %%
# View summary
with get_conn() as conn:
    df = pd.read_sql_query(
        "SELECT date(timestamp) as date, SUM(taken) as taken, COUNT(*)-SUM(taken) as missed "
        "FROM adherence_logs WHERE user_id=? GROUP BY date ORDER BY date DESC LIMIT 10",
        conn, params=[uid]
    )
print(df)
print(f"\nOverall adherence: {df['taken'].sum() / (df['taken']+df['missed']).sum():.1%}")

# %% [markdown]
# ## Step 7: Build Training Dataset & Train LSTM

# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ── Build sliding-window dataset ──────────────────────────
SEQ_LEN = 14

def get_sequence_from_db():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT date(timestamp) as date, MAX(taken) as taken "
            "FROM adherence_logs WHERE user_id=? AND medicine_id=? "
            "GROUP BY date ORDER BY date",
            (uid, mid)
        ).fetchall()
    
    sequence = [r["taken"] for r in rows]
    print(f"Raw sequence ({len(sequence)} days): {sequence[-20:]}")
    return sequence

def make_sequences(sequence, seq_len=14):
    X, y = [], []
    for i in range(len(sequence) - seq_len):
        X.append(sequence[i : i + seq_len])
        y.append(sequence[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

raw_seq = get_sequence_from_db()

# Augment with synthetic patterns for better training
np.random.seed(42)
for _ in range(300):
    base_rate = np.random.uniform(0.6, 0.95)
    syn = []
    for j in range(SEQ_LEN + 1):
        p = min(base_rate + 0.1, 0.98) if j > 0 and syn and syn[-1] == 1 else max(base_rate - 0.15, 0.4)
        syn.append(1 if np.random.random() < p else 0)
    raw_seq.extend(syn)

X, y = make_sequences(raw_seq, SEQ_LEN)
X = X.reshape(-1, SEQ_LEN, 1)
print(f"Dataset: X={X.shape}, y={y.shape}")
print(f"Miss rate: {1 - y.mean():.1%}")

# %%
# ── Define LSTM model ─────────────────────────────────────
class AdherenceLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.classifier(out).squeeze(-1)

# ── Training setup ────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.tensor(X_val),   torch.tensor(y_val)),   batch_size=32)

model_lstm = AdherenceLSTM().to(device)
optimizer  = torch.optim.Adam(model_lstm.parameters(), lr=1e-3, weight_decay=1e-4)
criterion  = nn.BCELoss()
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# %%
# ── Training loop ─────────────────────────────────────────
EPOCHS = 60
history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
best_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    # Train
    model_lstm.train()
    t_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model_lstm(xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_lstm.parameters(), 1.0)
        optimizer.step()
        t_losses.append(loss.item())
    
    # Validate
    model_lstm.eval()
    v_losses, all_p, all_l = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = model_lstm(xb.to(device))
            v_losses.append(criterion(p, yb.to(device)).item())
            all_p.extend(p.cpu().numpy())
            all_l.extend(yb.numpy())
    
    t_loss = np.mean(t_losses)
    v_loss = np.mean(v_losses)
    acc    = accuracy_score(np.array(all_l), (np.array(all_p) > 0.5).astype(int))
    f1     = f1_score(np.array(all_l), (np.array(all_p) > 0.5).astype(int), zero_division=0)
    
    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["val_acc"].append(acc)
    history["val_f1"].append(f1)
    
    scheduler.step(v_loss)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | t_loss={t_loss:.4f} val_loss={v_loss:.4f} acc={acc:.3f} f1={f1:.3f}")
    
    if v_loss < best_loss:
        best_loss = v_loss
        torch.save(model_lstm.state_dict(), f"{PROJECT_DIR}/models/adherence_lstm.pt")

print("\n✅ Training complete! Best model saved.")

# %%
# ── Evaluation ────────────────────────────────────────────
model_lstm.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        preds = model_lstm(xb.to(device))
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.numpy())

preds_bin = (np.array(all_preds) > 0.5).astype(int)
labels    = np.array(all_labels)

print("📊 Final Evaluation:")
print(f"  Accuracy: {accuracy_score(labels, preds_bin):.4f}")
print(f"  F1-Score: {f1_score(labels, preds_bin, zero_division=0):.4f}")
print()
print(classification_report(labels, preds_bin, target_names=["Will Take", "Will Miss"], zero_division=0))

# %%
# ── Plot training curves ──────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")

axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
axes[0].plot(history["val_loss"],   label="Val Loss",   color="#F44336")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history["val_acc"], label="Accuracy", color="#4CAF50")
axes[1].plot(history["val_f1"],  label="F1-Score", color="#FF9800")
axes[1].set_title("Validation Metrics"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PROJECT_DIR}/models/training_history.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Save LSTM to Drive ────────────────────────────────────
shutil.copy(f"{PROJECT_DIR}/models/adherence_lstm.pt", f"{DRIVE_MODELS}/adherence_lstm.pt")
print(f"✅ LSTM saved to Drive: {DRIVE_MODELS}/adherence_lstm.pt")

# %% [markdown]
# ## Step 8: Demo Prediction

# %%
# ── Load model and predict ────────────────────────────────
model_lstm.eval()

def predict_miss_probability(sequence):
    """Predict probability of missing next dose given 14-day history."""
    seq = list(sequence)
    if len(seq) < SEQ_LEN:
        seq = [0] * (SEQ_LEN - len(seq)) + seq
    elif len(seq) > SEQ_LEN:
        seq = seq[-SEQ_LEN:]
    
    x = torch.tensor(seq, dtype=torch.float32).reshape(1, SEQ_LEN, 1).to(device)
    with torch.no_grad():
        prob = model_lstm(x).item()
    return round(prob, 4)

# Test predictions
print("🔮 Sample Predictions:")
test_cases = [
    ([1]*14,                                    "Perfect adherence (14 days)"),
    ([1,1,0,1,1,0,1,1,0,1,1,0,1,1],            "Occasional misses"),
    ([1,1,1,1,1,1,1,0,0,0,0,0,0,0],            "Recent decline"),
    ([0,0,0,0,0,0,0,0,0,1,0,0,0,0],            "Poor adherence"),
]

for seq, label in test_cases:
    prob = predict_miss_probability(seq)
    risk = "🟢 Low" if prob < 0.3 else "🟡 Medium" if prob < 0.6 else "🔴 HIGH"
    print(f"  {label}")
    print(f"    Sequence:        {seq}")
    print(f"    Miss probability: {prob:.1%}  →  {risk}")
    print()

# %% [markdown]
# ## Step 9: Run Streamlit App (via ngrok tunnel)

# %%
# Install ngrok for public URL in Colab
get_ipython().system('pip install pyngrok -q')
get_ipython().system('pip install streamlit -q')

# %%
# Option: Run app via subprocess + ngrok
import subprocess, time, threading

# Copy app files (or recreate them in PROJECT_DIR)
# For demo, create a minimal streamlit app inline

MINIMAL_APP = f"""
import streamlit as st
import torch
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="💊 Medicine Tracker", layout="wide")
st.title("💊 Smart Medicine Tracker — Colab Demo")

SEQ_LEN = {SEQ_LEN}
DB_PATH  = "{DB_PATH}"

class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, 64, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 32), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(32, 1), torch.nn.Sigmoid()
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)

@st.cache_resource
def load_model():
    m = LSTM()
    try:
        m.load_state_dict(torch.load("{PROJECT_DIR}/models/adherence_lstm.pt", map_location="cpu"))
    except: pass
    m.eval()
    return m

tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Prediction"])

with tab1:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT date(timestamp) as date, SUM(taken) as taken, COUNT(*)-SUM(taken) as missed FROM adherence_logs GROUP BY date ORDER BY date DESC LIMIT 30", conn)
    conn.close()
    
    if not df.empty:
        pct = df['taken'].sum() / (df['taken']+df['missed']).sum() * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("30-Day Adherence", f"{{pct:.1f}}%")
        c2.metric("Doses Taken", int(df['taken'].sum()))
        c3.metric("Doses Missed", int(df['missed'].sum()))
        st.bar_chart(df.set_index('date')['taken'])
    else:
        st.info("No data yet.")

with tab2:
    model = load_model()
    st.subheader("Enter recent 14-day adherence (1=taken, 0=missed)")
    seq_input = st.text_input("Sequence", value="1,1,0,1,1,1,0,1,1,1,1,0,1,1")
    if st.button("Predict"):
        seq = [float(x.strip()) for x in seq_input.split(",")]
        seq = (seq[-SEQ_LEN:] if len(seq) >= SEQ_LEN else [0]*(SEQ_LEN-len(seq))+seq)
        x = torch.tensor(seq, dtype=torch.float32).reshape(1, SEQ_LEN, 1)
        with torch.no_grad():
            prob = model(x).item()
        color = "🟢" if prob < 0.3 else "🟡" if prob < 0.6 else "🔴"
        st.metric("Miss Probability", f"{{prob:.1%}}", delta=f"{{color}}")
        st.progress(prob)
"""

with open("/content/app_colab.py", "w") as f:
    f.write(MINIMAL_APP)

# Start Streamlit in background
proc = subprocess.Popen(
    ["streamlit", "run", "/content/app_colab.py", "--server.port=8501", "--server.headless=true"]
)
time.sleep(4)

# Create public URL
from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"\n🌐 App is running!")
print(f"   Open: {public_url}")
print(f"   Share this URL with anyone!")

# %% [markdown]
# ## Summary
#
# ✅ What we built:
# - YOLOv8 medicine detection model
# - SQLite adherence database  
# - LSTM miss-probability predictor
# - Streamlit dashboard
#
# 📁 Files saved to Google Drive:
# - `MyDrive/MedicineTracker/models/medicine_yolo.pt`
# - `MyDrive/MedicineTracker/models/adherence_lstm.pt`
