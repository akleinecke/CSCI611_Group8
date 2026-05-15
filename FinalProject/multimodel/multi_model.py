"""
Multimodal Speaker Identification - Fusion Model (V2)
======================================================
Combines:
  - Austin's  MouthVideoCNN V6  -> RGB input, .pt checkpoint
  - Alexander's AudioResNet     -> 512-dim features, .pth state_dict

Strategy: Late Fusion
  video frames    -> VideoModel.extract_features() -> [B, 512]  ──┐
                                                                    ├─ cat -> [B, 1024] -> classifier
  mel-spectrogram -> AudioModel.extract_features() -> [B, 512]  ──┘

Dataset: 60 speakers, 20 samples each (1200 total), 80/20 split

File layout expected:
    multimodal_fusion.py
    spectrogram_labels.csv
    spectrograms/
    vox_celeb_subset/
        subset.csv
        roi_cache/mediapipe_image_frames-32_size-64_seed-611/
    saved_models/
        video_cnn_V6_<ts>.pt
        audioresnet.pth
"""

from pathlib import Path
from datetime import datetime
import csv
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
)


# ==============================================================================
# Config
# ==============================================================================
SEED        = 611
NUM_CLASSES = 60
BATCH_SIZE  = 8
EPOCHS      = 30
LR          = 1e-3
LR_PAT      = 3
EARLY_STOP  = 8
MIN_EPOCHS  = 8

# Austin's files (V6 — RGB)
VIDEO_CHECKPOINT = Path("saved_models/video_cnn_V6_2026-05-05_11-36-47.pt")
VIDEO_CSV        = Path("./vox_celeb_subset/subset.csv")
VIDEO_DATA_ROOT  = Path("./vox_celeb_subset/")
ROI_CACHE_DIR    = Path("./vox_celeb_subset/roi_cache/mediapipe_image_frames-32_size-64_seed-611")

# Alexander's files (AudioResNet)
AUDIO_CHECKPOINT = Path("saved_models/audioresnet.pth")
AUDIO_CSV        = Path("spectrogram_labels.csv")

# Output
SAVE_DIR    = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)
timestamp   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FUSION_SAVE = SAVE_DIR / f"fusion_model_v2_{timestamp}.pt"
FUSION_TXT  = SAVE_DIR / f"fusion_model_v2_{timestamp}.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# 1. AUSTIN'S VIDEO MODEL V6 — RGB input, bidirectional GRU, 512-dim output
# ==============================================================================
class MouthVideoCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            # Input = [3, 64, 64]  <- RGB now
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
        # bidirectional -> 256*2 = 512
        self.gru     = nn.GRU(input_size=128, hidden_size=256,
                               batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = x.view(b, t, 128)
        gru_out, _ = self.gru(x)
        x = gru_out.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        """Returns 512-dim feature vector before classifier."""
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = x.view(b, t, 128)
        gru_out, _ = self.gru(x)
        x = gru_out.mean(dim=1)   # [B, 512]
        x = self.dropout(x)
        return x


# ==============================================================================
# 2. ALEXANDER'S AUDIO MODEL — ResNet with residual blocks, 512-dim output
# ==============================================================================
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class AudioResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ResBlock(1,   64,  stride=2)
        self.layer2 = ResBlock(64,  128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        """Returns 512-dim feature vector before classifier."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)   # [B, 512]
        x = self.dropout(x)
        return x


# 3. FUSION MODEL 

VIDEO_FEAT_DIM = 512
AUDIO_FEAT_DIM = 512
FUSED_DIM      = VIDEO_FEAT_DIM + AUDIO_FEAT_DIM 

class MultimodalFusionModel(nn.Module):
    def __init__(self, video_model, audio_model, num_classes,
                 freeze_backbones=True):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model

        if freeze_backbones:
            for p in self.video_model.parameters():
                p.requires_grad = False
            for p in self.audio_model.parameters():
                p.requires_grad = False

        self.fusion_head = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.BatchNorm1d(512), 
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), 
            nn.LeakyReLU(0.1), 
            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, frames, spectrogram):
        v     = self.video_model.extract_features(frames)       # [B, 512]
        a     = self.audio_model.extract_features(spectrogram)  # [B, 512]
        fused = torch.cat([v, a], dim=1)                        # [B, 1024]
        return self.fusion_head(fused)

# LOAD PRETRAIN WEIGHTS
def load_video_model(checkpoint_path, num_classes, device):
    model = MouthVideoCNN(num_classes=num_classes).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[VIDEO]  Loaded | epoch {ckpt.get('epoch','?')} | "
          f"best_val_acc={ckpt.get('best_val_acc', 0)*100:.2f}%")
    return model


def load_audio_model(checkpoint_path, num_classes, device):
    model = AudioResNet(num_classes=num_classes).to(device)
    # Alexander saved plain state_dict
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[AUDIO]  Loaded weights from {checkpoint_path.name}")
    return model


# DATASET
def fix_path(s):
    return Path(s.replace("\\", "/"))

def build_cache_name(row, cache_dir):
    raw       = row["mp4_path"].replace("\\", "/")
    safe_name = raw.replace("/", "_")
    return cache_dir / f"{safe_name}.pt"


class MultimodalDataset(Dataset):
    """
    Each sample returns:
        frames      : [T, 3, H, W]   RGB — Austin V6 format
        spectrogram : [1, F, T_spec] Alexander's .npy
        label       : int (0-59)
    """
    def __init__(self, paired_rows, label_to_idx, training=False):
        self.rows         = paired_rows
        self.label_to_idx = label_to_idx
        self.training     = training

    def __len__(self):
        return len(self.rows)

    def apply_augmentation(self, frames):
        frames = frames.clone()
        if random.random() < 0.5:
            frames = torch.flip(frames, dims=[3])
        brightness = 0.8 + 0.4 * random.random()
        frames = torch.clamp(frames * brightness, 0.0, 1.0)
        return frames

    def __getitem__(self, idx):
        row   = self.rows[idx]
        label = self.label_to_idx[row["label"]]

        # Video — RGB tensor [T, 3, H, W]
        frames = torch.load(row["mp4_cache_path"], weights_only=True)
        if self.training:
            frames = self.apply_augmentation(frames)

        # Audio — spectrogram [1, F, T]
        spec = np.load(row["spec_path"])
        spec = torch.FloatTensor(spec).unsqueeze(0)

        return frames, spec, torch.tensor(label, dtype=torch.long)


def build_paired_rows(video_csv, roi_cache_dir, audio_csv, seed=611):
    video_rows = []
    with open(video_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            video_rows.append(r)

    audio_df     = pd.read_csv(audio_csv)
    spk_to_specs = {}
    for _, row in audio_df.iterrows():
        spk_to_specs.setdefault(row["speaker_id"], []).append(row["spectrogram_path"])

    all_labels   = sorted({int(r["label"]) for r in video_rows})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
    spk_to_label = {r["speaker_id"]: int(r["label"]) for r in video_rows}

    spk_spec_cursor = {sid: 0 for sid in spk_to_specs}
    paired = []

    for r in video_rows:
        sid        = r["speaker_id"]
        cache_path = build_cache_name(r, roi_cache_dir)
        if not cache_path.exists():
            print(f"[WARNING] Missing ROI cache: {cache_path}, skipping.")
            continue
        specs = spk_to_specs.get(sid, [])
        if not specs:
            continue
        spec_path = specs[spk_spec_cursor[sid] % len(specs)]
        spk_spec_cursor[sid] += 1
        paired.append({
            "mp4_cache_path": cache_path,
            "spec_path":      spec_path,
            "speaker_id":     sid,
            "label":          spk_to_label[sid],
        })

    print(f"Paired samples ready: {len(paired)} across {len(all_labels)} speakers")
    return paired, label_to_idx


# ==============================================================================
# 6. TRAINING LOOP
# ==============================================================================
def train_fusion(fusion_model, train_loader, val_loader,
                 epochs, lr, device, save_path, txt_path):

    fusion_model = fusion_model.to(device)
    criterion    = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PAT
    )

    acc_metric  = MulticlassAccuracy(num_classes=NUM_CLASSES).to(device)
    f1_macro    = MulticlassF1Score(num_classes=NUM_CLASSES, average="macro").to(device)
    f1_weighted = MulticlassF1Score(num_classes=NUM_CLASSES, average="weighted").to(device)

    train_losses, train_accs         = [], []
    val_losses,   val_accs           = [], []
    val_macro_f1s, val_weighted_f1s  = [], []

    best_val_acc      = -1.0
    best_val_loss     = float("inf")
    best_epoch        = 0
    best_macro_f1     = 0.0
    best_weighted_f1  = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):

        # ---- Train ----
        fusion_model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for frames, spec, labels in train_loader:
            frames = frames.to(device)
            spec   = spec.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out  = fusion_model(frames, spec)
            loss = criterion(out, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0)
            optimizer.step()

            t_loss    += loss.item() * labels.size(0)
            t_correct += (out.argmax(1) == labels).sum().item()
            t_total   += labels.size(0)

        train_loss = t_loss / t_total
        train_acc  = t_correct / t_total

        # ---- Validate ----
        fusion_model.eval()
        acc_metric.reset(); f1_macro.reset(); f1_weighted.reset()
        v_loss, v_total = 0.0, 0

        with torch.no_grad():
            for frames, spec, labels in val_loader:
                frames = frames.to(device)
                spec   = spec.to(device)
                labels = labels.to(device)

                out  = fusion_model(frames, spec)
                loss = criterion(out, labels)
                v_loss  += loss.item() * labels.size(0)
                v_total += labels.size(0)

                acc_metric.update(out, labels)
                f1_macro.update(out, labels)
                f1_weighted.update(out, labels)

        val_loss     = v_loss / v_total
        val_acc      = acc_metric.compute().item()
        vmacro_f1    = f1_macro.compute().item()
        vweighted_f1 = f1_weighted.compute().item()

        scheduler.step(val_loss)

        train_losses.append(train_loss);   train_accs.append(train_acc)
        val_losses.append(val_loss);       val_accs.append(val_acc)
        val_macro_f1s.append(vmacro_f1);   val_weighted_f1s.append(vweighted_f1)

        print(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
            f"macro_f1={vmacro_f1*100:.2f}% weighted_f1={vweighted_f1*100:.2f}%"
        )

        improved = (val_acc > best_val_acc) or \
                   (val_acc == best_val_acc and val_loss < best_val_loss)

        if improved:
            best_val_acc      = val_acc
            best_val_loss     = val_loss
            best_epoch        = epoch + 1
            best_macro_f1     = vmacro_f1
            best_weighted_f1  = vweighted_f1
            epochs_no_improve = 0

            torch.save({
                "epoch":                epoch + 1,
                "model_state_dict":     fusion_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc":         best_val_acc,
                "best_val_loss":        best_val_loss,
                "best_macro_f1":        best_macro_f1,
                "best_weighted_f1":     best_weighted_f1,
                "train_losses":         train_losses,
                "val_losses":           val_losses,
                "train_accs":           train_accs,
                "val_accs":             val_accs,
                "val_macro_f1s":        val_macro_f1s,
                "val_weighted_f1s":     val_weighted_f1s,
                "num_classes":          NUM_CLASSES,
                "video_feat_dim":       VIDEO_FEAT_DIM,
                "audio_feat_dim":       AUDIO_FEAT_DIM,
                "fused_dim":            FUSED_DIM,
            }, save_path)
            print(f"  --> Saved best checkpoint (val_acc={best_val_acc*100:.2f}%)")
        else:
            epochs_no_improve += 1

        if epoch + 1 >= MIN_EPOCHS and epochs_no_improve >= EARLY_STOP:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    # Save text summary
    lines = [
        "=== FUSION MODEL V2 TRAINING SUMMARY ===",
        f"Timestamp : {timestamp}",
        f"Device    : {device}",
        "",
        "---- ARCHITECTURE ----",
        f"Video backbone : MouthVideoCNN V6 (RGB, bidirectional GRU)",
        f"Audio backbone : AudioResNet (4 ResBlocks)",
        f"Video feat dim : {VIDEO_FEAT_DIM}",
        f"Audio feat dim : {AUDIO_FEAT_DIM}",
        f"Fused dim      : {FUSED_DIM}",
        "",
        "---- DATASET ----",
        f"Num classes : {NUM_CLASSES}",
        f"Train size  : {len(train_loader.dataset)}",
        f"Val size    : {len(val_loader.dataset)}",
        "",
        "---- HYPERPARAMETERS ----",
        f"Epochs: {epochs}  Batch: {BATCH_SIZE}  LR: {lr}",
        "",
        "---- BEST CHECKPOINT ----",
        f"Best Epoch       : {best_epoch}",
        f"Best Val Acc     : {best_val_acc*100:.4f}%",
        f"Best Val Loss    : {best_val_loss:.4f}",
        f"Best Macro F1    : {best_macro_f1*100:.4f}%",
        f"Best Weighted F1 : {best_weighted_f1*100:.4f}%",
        "",
        "---- FULL HISTORY ----",
        f"Train Losses     : {train_losses}",
        f"Val Losses       : {val_losses}",
        f"Train Accs       : {train_accs}",
        f"Val Accs         : {val_accs}",
        f"Val Macro F1s    : {val_macro_f1s}",
        f"Val Weighted F1s : {val_weighted_f1s}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved : {txt_path}")
    print(f"Best val acc  : {best_val_acc*100:.2f}%")
    print(f"Best macro F1 : {best_macro_f1*100:.2f}%")


# ==============================================================================
# 7. MAIN
# ==============================================================================
if __name__ == "__main__":

    # Build paired dataset
    paired_rows, label_to_idx = build_paired_rows(
        video_csv     = VIDEO_CSV,
        roi_cache_dir = ROI_CACHE_DIR,
        audio_csv     = AUDIO_CSV,
        seed          = SEED,
    )

    # 80/20 split
    indices = list(range(len(paired_rows)))
    random.Random(SEED).shuffle(indices)
    split      = int(0.8 * len(indices))
    train_rows = [paired_rows[i] for i in indices[:split]]
    val_rows   = [paired_rows[i] for i in indices[split:]]

    train_ds = MultimodalDataset(train_rows, label_to_idx, training=True)
    val_ds   = MultimodalDataset(val_rows,   label_to_idx, training=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Load pretrained backbones
    video_model = load_video_model(VIDEO_CHECKPOINT, NUM_CLASSES, DEVICE)
    audio_model = load_audio_model(AUDIO_CHECKPOINT, NUM_CLASSES, DEVICE)

    # Build fusion model
    fusion_model = MultimodalFusionModel(
        video_model      = video_model,
        audio_model      = audio_model,
        num_classes      = NUM_CLASSES,
        freeze_backbones = True,
    )

    trainable = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in fusion_model.parameters())
    print(f"\nFusion model ready:")
    print(f"  Video feat dim   : {VIDEO_FEAT_DIM}")
    print(f"  Audio feat dim   : {AUDIO_FEAT_DIM}")
    print(f"  Fused dim        : {FUSED_DIM}")
    print(f"  Trainable params : {trainable:,}  (fusion head only)")
    print(f"  Total params     : {total:,}")

    # Train
    train_fusion(
        fusion_model = fusion_model,
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = EPOCHS,
        lr           = LR,
        device       = DEVICE,
        save_path    = FUSION_SAVE,
        txt_path     = FUSION_TXT,
    )