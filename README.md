# 💊 Smart Medicine Adherence Tracker
### AI-Powered Medicine Detection + LSTM Adherence Prediction + Streamlit Dashboard

---

## 🧠 Project Overview

An end-to-end AI system that:
- Detects medicine (pill strips, bottles, blister packs) using **YOLOv8**
- Logs whether the user has taken medicine with timestamped records
- Stores all data in **SQLite** for persistence
- Predicts probability of **missing future doses** using **LSTM**
- Provides an interactive **Streamlit dashboard** with alerts

---

## 📁 Project Structure

```
smart_medicine_tracker/
├── README.md
├── requirements.txt
├── dataset/
│   ├── medicine.yaml           # YOLO dataset config
│   ├── train/images/           # Training images
│   ├── train/labels/           # YOLO format labels
│   ├── val/images/
│   ├── val/labels/
│   └── test/images/
├── models/
│   ├── train_yolo.py           # YOLOv8 training script
│   ├── inference.py            # Detection inference
│   └── lstm_model.py           # LSTM training + prediction
├── app/
│   ├── database.py             # SQLite database layer
│   ├── reminder.py             # Email/notification system
│   └── streamlit_app.py        # Main GUI application
├── utils/
│   ├── preprocess.py           # Image preprocessing
│   └── visualize.py            # Chart generation
├── notebooks/
│   └── SmartMedicineTracker_Colab.ipynb   # Full Colab notebook
└── data/
    └── adherence.db            # SQLite database (auto-created)
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app
streamlit run app/streamlit_app.py
```

## 🌐 Google Colab

Open `notebooks/SmartMedicineTracker_Colab.ipynb` in Google Colab with GPU enabled.

---

## 🎓 Viva Explanation

**Q: What does this project do?**
> It uses YOLOv8 computer vision to detect medicine objects from camera/image input, logs whether doses were taken into SQLite, then uses an LSTM neural network trained on that time-series data to predict the probability of missing the next dose. If risk exceeds 70%, the system triggers an alert.

**Q: Why LSTM?**
> Medicine adherence is a time-series problem — whether you missed a dose today depends on your recent pattern. LSTMs capture temporal dependencies in sequential binary data (taken=1, missed=0), making them ideal here.

**Q: Why YOLOv8?**
> YOLOv8 is state-of-the-art for real-time object detection, supports custom training easily via Ultralytics, and runs efficiently even on CPU for inference.
