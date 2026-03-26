"""
=============================================================
models/lstm_model.py — LSTM Adherence Prediction Model
=============================================================
Predicts the probability of MISSING the next dose based on
recent adherence history (time-series of 0/1 values).

Architecture:
    Input: (batch, sequence_len, 1)  — past N days: taken=1, missed=0
    LSTM1 → Dropout → LSTM2 → Dropout → Dense(1, sigmoid)
    Output: probability of MISSING next dose (0 = will take, 1 = will miss)

Usage:
    # Train
    python models/lstm_model.py --train

    # Predict from sequence
    python models/lstm_model.py --predict "1,1,0,1,1,1,0,1,1,1,1,0,1,1"
=============================================================
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# ── PyTorch imports ───────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    os.system("pip install torch -q")
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

# ── sklearn for metrics ───────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt


# ── Config ────────────────────────────────────────────────
LSTM_CONFIG = {
    "sequence_length": 14,      # 14 days of history as input
    "hidden_size":     64,      # LSTM hidden units
    "num_layers":      2,       # Stacked LSTM layers
    "dropout":         0.3,
    "epochs":          60,
    "batch_size":      32,
    "learning_rate":   1e-3,
    "model_path":      "models/adherence_lstm.pt",
    "meta_path":       "models/adherence_lstm_meta.json",
}


# ═══════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════

class AdherenceLSTM(nn.Module):
    """
    Bidirectional LSTM for binary dose-adherence prediction.
    
    Input shape:  (batch_size, sequence_length, 1)
    Output shape: (batch_size, 1)  — miss probability
    """

    def __init__(
        self,
        input_size:  int = 1,
        hidden_size: int = 64,
        num_layers:  int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Bidirectional LSTM — captures both recent and older patterns
        self.lstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
            bidirectional= True,
        )

        self.dropout = nn.Dropout(dropout)

        # Output: bidirectional doubles hidden size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),             # Output: miss probability [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)                # (batch, seq_len, hidden*2)
        last_hidden  = lstm_out[:, -1, :]         # Take last time step
        out          = self.dropout(last_hidden)
        return self.classifier(out).squeeze(-1)    # (batch,)


# ═══════════════════════════════════════════════════════════
# TRAINER
# ═══════════════════════════════════════════════════════════

class LSTMTrainer:
    """Train and evaluate the AdherenceLSTM model."""

    def __init__(self, config: dict = None):
        self.cfg    = {**LSTM_CONFIG, **(config or {})}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  Using device: {self.device}")

        self.model = AdherenceLSTM(
            hidden_size = self.cfg["hidden_size"],
            num_layers  = self.cfg["num_layers"],
            dropout     = self.cfg["dropout"],
        ).to(self.device)

    # ── Data Loading ──────────────────────────────────────

    def load_data_from_db(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load training sequences directly from the SQLite database."""
        from app.database import Database
        db = Database()
        return db.get_all_sequences_for_training(self.cfg["sequence_length"])

    def prepare_loaders(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader]:
        """Split data and build PyTorch DataLoaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=(y > 0.5).astype(int)
        )

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val,   dtype=torch.float32),
            torch.tensor(y_val,   dtype=torch.float32),
        )

        train_loader = DataLoader(train_ds, batch_size=self.cfg["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.cfg["batch_size"])

        return train_loader, val_loader

    # ── Training Loop ─────────────────────────────────────

    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> dict:
        """
        Train the LSTM model.
        
        Args:
            X, y: Optional. If None, loads from database.
        
        Returns:
            Training history dict with train_loss and val_loss per epoch.
        """
        if X is None or y is None:
            X, y = self.load_data_from_db()

        # ── Class weighting (handle imbalance) ────────────
        pos_rate = y.mean()
        neg_rate = 1 - pos_rate
        pos_weight = torch.tensor(neg_rate / (pos_rate + 1e-6), dtype=torch.float32)
        print(f"📊 Class balance: taken={pos_rate:.1%}, missed={neg_rate:.1%}")
        print(f"   pos_weight={pos_weight:.2f}")

        criterion = nn.BCELoss(weight=None)           # Simple BCE
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg["learning_rate"], weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=True
        )

        train_loader, val_loader = self.prepare_loaders(X, y)
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        best_val_loss = float("inf")
        patience_count = 0
        EARLY_STOP = 10

        print(f"\n🚀 Training LSTM | epochs={self.cfg['epochs']} | device={self.device}")
        print("─" * 60)

        for epoch in range(1, self.cfg["epochs"] + 1):
            # ── Train phase ───────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()
                train_losses.append(loss.item())

            # ── Validation phase ──────────────────────────
            self.model.eval()
            val_losses, all_preds, all_labels = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds   = self.model(X_batch)
                    val_losses.append(criterion(preds, y_batch).item())
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(y_batch.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            val_preds_bin = (np.array(all_preds) > 0.5).astype(int)
            val_labels    = np.array(all_labels)
            val_acc  = accuracy_score(val_labels, val_preds_bin)
            val_f1   = f1_score(val_labels, val_preds_bin, zero_division=0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            scheduler.step(val_loss)

            if epoch % 5 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{self.cfg['epochs']} | "
                    f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.3f} | val_f1={val_f1:.3f}"
                )

            # ── Save best model ───────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                self._save_model()
            else:
                patience_count += 1
                if patience_count >= EARLY_STOP:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break

        print("\n✅ Training complete!")
        self._evaluate(val_loader)
        return history

    # ── Evaluation ────────────────────────────────────────

    def _evaluate(self, val_loader: DataLoader):
        """Print full evaluation metrics."""
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = self.model(X_batch.to(self.device))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        preds_bin = (np.array(all_preds) > 0.5).astype(int)
        labels    = np.array(all_labels)

        print("\n📊 Evaluation Metrics:")
        print(f"   Accuracy:   {accuracy_score(labels, preds_bin):.4f}")
        print(f"   F1-Score:   {f1_score(labels, preds_bin, zero_division=0):.4f}")
        try:
            print(f"   ROC-AUC:    {roc_auc_score(labels, all_preds):.4f}")
        except ValueError:
            pass  # Only one class present
        print("\n   Classification Report:")
        print(classification_report(labels, preds_bin,
                                     target_names=["Will Take", "Will Miss"],
                                     zero_division=0))

    def plot_history(self, history: dict, save_path: str = "models/training_history.png"):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LSTM Training History", fontsize=14, fontweight="bold")

        # Loss curves
        axes[0].plot(history["train_loss"], label="Train Loss", color="#2196F3")
        axes[0].plot(history["val_loss"],   label="Val Loss",   color="#F44336")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy & F1
        axes[1].plot(history["val_acc"], label="Accuracy", color="#4CAF50")
        axes[1].plot(history["val_f1"],  label="F1-Score", color="#FF9800")
        axes[1].set_title("Validation Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📈 Training plot saved: {save_path}")

    # ── Save / Load ───────────────────────────────────────

    def _save_model(self):
        """Save model weights and config."""
        Path(self.cfg["model_path"]).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.cfg["model_path"])

        meta = {
            "sequence_length": self.cfg["sequence_length"],
            "hidden_size":     self.cfg["hidden_size"],
            "num_layers":      self.cfg["num_layers"],
            "dropout":         self.cfg["dropout"],
        }
        with open(self.cfg["meta_path"], "w") as f:
            json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════════════
# PREDICTOR (used in the GUI)
# ═══════════════════════════════════════════════════════════

class AdherencePredictor:
    """
    Load trained LSTM and make predictions.
    
    Usage:
        predictor = AdherencePredictor()
        prob = predictor.predict_miss_probability([1,1,0,1,1,1,0,1,1,1,1,0,1,1])
        # Returns: 0.23 → 23% chance of missing next dose
    """

    def __init__(self, model_path: str = None, meta_path: str = None):
        self.model_path = model_path or LSTM_CONFIG["model_path"]
        self.meta_path  = meta_path  or LSTM_CONFIG["meta_path"]
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model      = self._load_model()
        self.seq_len    = LSTM_CONFIG["sequence_length"]

    def _load_model(self) -> AdherenceLSTM:
        """Load model with weights."""
        # Load metadata if available
        if Path(self.meta_path).exists():
            with open(self.meta_path) as f:
                meta = json.load(f)
        else:
            meta = LSTM_CONFIG

        model = AdherenceLSTM(
            hidden_size = meta.get("hidden_size", 64),
            num_layers  = meta.get("num_layers", 2),
            dropout     = 0.0,   # No dropout at inference
        ).to(self.device)

        if Path(self.model_path).exists():
            model.load_state_dict(
                torch.load(self.model_path, map_location=self.device, weights_only=True)
            )
            print(f"✅ LSTM model loaded: {self.model_path}")
        else:
            print(f"⚠️  LSTM model not found at {self.model_path}")
            print("   Train first: python models/lstm_model.py --train")
            print("   Using untrained model — predictions will be random")

        model.eval()
        return model

    def predict_miss_probability(self, sequence: list) -> float:
        """
        Predict probability of MISSING the next dose.
        
        Args:
            sequence: List of 0/1 values (recent adherence history)
                      Length should match sequence_length (14 by default)
                      If shorter, it will be zero-padded at the start.
                      If longer, the last seq_len values are used.
        
        Returns:
            float in [0, 1]: probability of MISSING next dose
            0.0 = very likely to take the medicine
            1.0 = very likely to miss the medicine
        """
        seq = list(sequence)
        
        # Pad or trim to required length
        if len(seq) < self.seq_len:
            seq = [0] * (self.seq_len - len(seq)) + seq
        elif len(seq) > self.seq_len:
            seq = seq[-self.seq_len:]
        
        # Convert to tensor
        x = torch.tensor(seq, dtype=torch.float32).reshape(1, self.seq_len, 1)
        x = x.to(self.device)
        
        with torch.no_grad():
            prob = self.model(x).item()
        
        return round(prob, 4)

    def predict_from_db(self, user_id: int, medicine_id: int) -> dict:
        """
        Predict from most recent DB history.
        
        Returns:
            {
                miss_probability: float,
                risk_level: str,        # "Low" | "Medium" | "High"
                recent_streak: int,     # consecutive days taken
                recommendation: str,
            }
        """
        from app.database import Database
        db = Database()
        
        sequence = db.get_daily_sequence(user_id, medicine_id, days=self.seq_len)
        miss_prob = self.predict_miss_probability(sequence.tolist())
        
        # ── Risk level ────────────────────────────────────
        if miss_prob < 0.30:
            risk_level     = "Low"
            recommendation = "You're on track! Keep it up 💪"
        elif miss_prob < 0.60:
            risk_level     = "Medium"
            recommendation = "Consider setting an extra reminder today."
        else:
            risk_level     = "High"
            recommendation = "⚠️ High risk of missing dose! Please take your medicine now."
        
        # ── Streak calculation ────────────────────────────
        streak = 0
        for val in reversed(sequence.tolist()):
            if val == 1:
                streak += 1
            else:
                break
        
        return {
            "miss_probability": miss_prob,
            "take_probability": round(1 - miss_prob, 4),
            "risk_level":       risk_level,
            "recent_streak":    streak,
            "recommendation":   recommendation,
            "sequence":         sequence.tolist(),
        }


# ── CLI ───────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM Adherence Prediction")
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Predict from comma-separated sequence")
    parser.add_argument("--epochs",  type=int, default=LSTM_CONFIG["epochs"])
    parser.add_argument("--colab",   action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.train:
        LSTM_CONFIG["epochs"] = args.epochs
        trainer = LSTMTrainer()
        X, y = trainer.load_data_from_db()
        history = trainer.train(X, y)
        trainer.plot_history(history)

    elif args.predict:
        seq = [float(x) for x in args.predict.split(",")]
        predictor = AdherencePredictor()
        prob = predictor.predict_miss_probability(seq)
        print(f"\n🔮 Miss probability: {prob:.1%}")
        print(f"   Risk level: {'Low' if prob < 0.3 else 'Medium' if prob < 0.6 else 'High'}")

    else:
        # Quick demo
        print("Running quick demo...")
        trainer = LSTMTrainer()
        X, y = trainer.load_data_from_db()
        history = trainer.train(X, y)

        predictor = AdherencePredictor()
        
        print("\n🔮 Sample predictions:")
        examples = [
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "Perfect adherence"),
            ([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], "Occasional misses"),
            ([0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], "Poor adherence"),
        ]
        for seq, label in examples:
            prob = predictor.predict_miss_probability(seq)
            print(f"   {label}: {prob:.1%} miss chance")
