"""
Fusion Model V2 - Video Demo with Mouth ROI
=============================================
Plays actual .mp4 video clips with:
  - Prediction overlay
  - Live mouth ROI detection shown in corner
  - Audio playback

Controls:
    N     = next sample
    SPACE = pause / resume
    Q     = quit

Usage:
    python demo.py
"""

from pathlib import Path
import csv
import random
import subprocess
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


# ==============================================================================
# Config
# ==============================================================================
SEED             = 611
NUM_CLASSES      = 60

VIDEO_CSV        = Path("./vox_celeb_subset/subset.csv")
VIDEO_DATA_ROOT  = Path("./vox_celeb_subset/")
ROI_CACHE_DIR    = Path("./vox_celeb_subset/roi_cache/mediapipe_image_frames-32_size-64_seed-611")
AUDIO_CSV        = Path("spectrogram_labels.csv")

VIDEO_CHECKPOINT  = Path("saved_models/video_cnn_V6_2026-05-05_11-36-47.pt")
AUDIO_CHECKPOINT  = Path("saved_models/audioresnet.pth")
FUSION_CHECKPOINT = Path("saved_models/fusion_model_v2_2026-05-07_22-21-42.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
torch.manual_seed(SEED)

# Face detector — built into OpenCV, no extra install needed
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ==============================================================================
# Model Definitions
# ==============================================================================
class MouthVideoCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1),
            nn.AdaptiveMaxPool2d((1, 1)),
        )
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
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.frame_encoder(x)
        x = x.view(b, t, 128)
        gru_out, _ = self.gru(x)
        x = gru_out.mean(dim=1)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1    = nn.Conv2d(in_channels, out_channels, 3,
                                  stride=stride, padding=1)
        self.bn1      = nn.BatchNorm2d(out_channels)
        self.conv2    = nn.Conv2d(out_channels, out_channels, 3,
                                  stride=1, padding=1)
        self.bn2      = nn.BatchNorm2d(out_channels)
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
        self.layer1  = ResBlock(1,   64,  stride=2)
        self.layer2  = ResBlock(64,  128, stride=2)
        self.layer3  = ResBlock(128, 256, stride=2)
        self.layer4  = ResBlock(256, 512, stride=2)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def extract_features(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.dropout(x)
        return x


class MultimodalFusionModel(nn.Module):
    def __init__(self, video_model, audio_model, num_classes):
        super().__init__()
        self.video_model = video_model
        self.audio_model = audio_model
        self.fusion_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.LeakyReLU(0.1), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, frames, spectrogram):
        v     = self.video_model.extract_features(frames)
        a     = self.audio_model.extract_features(spectrogram)
        fused = torch.cat([v, a], dim=1)
        return self.fusion_head(fused)


# ==============================================================================
# Mouth ROI detection using OpenCV Haar cascade
# ==============================================================================
def detect_mouth_roi(frame, size=120):
    """
    Detect face, crop mouth region, return:
      - frame with face + mouth boxes drawn
      - mouth ROI image (size x size color)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    mouth_roi = None

    if len(faces) > 0:
        # Use largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(frame, "Face", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Estimate mouth region — lower 35% of face
        mx1 = x + int(0.15 * w)
        mx2 = x + int(0.85 * w)
        my1 = y + int(0.65 * h)
        my2 = y + int(0.95 * h)

        # Clamp to frame bounds
        mx1 = max(0, mx1); my1 = max(0, my1)
        mx2 = min(frame.shape[1], mx2)
        my2 = min(frame.shape[0], my2)

        # Draw mouth box
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (0, 255, 100), 2)
        cv2.putText(frame, "Mouth ROI", (mx1, my1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1)

        # Crop and resize mouth
        roi = frame[my1:my2, mx1:mx2]
        if roi.size > 0:
            mouth_roi = cv2.resize(roi, (size, size))

    return frame, mouth_roi


# ==============================================================================
# Audio playback
# ==============================================================================
audio_process = None

def play_audio(mp4_path):
    global audio_process
    stop_audio()
    try:
        audio_process = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loop", "0", str(mp4_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[AUDIO] ffplay not found. Install: winget install ffmpeg")

def stop_audio():
    global audio_process
    if audio_process and audio_process.poll() is None:
        audio_process.terminate()
        audio_process = None


# ==============================================================================
# Helpers
# ==============================================================================
def fix_path(s):
    return Path(s.replace("\\", "/"))

def build_cache_name(row, cache_dir):
    raw       = row["mp4_path"].replace("\\", "/")
    safe_name = raw.replace("/", "_")
    return cache_dir / f"{safe_name}.pt"

def draw_text(frame, txt, pos, color, scale=0.65, thickness=2):
    x, y = pos
    cv2.putText(frame, txt, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(frame, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ==============================================================================
# Load models
# ==============================================================================
print("Loading models...")

video_model = MouthVideoCNN(num_classes=NUM_CLASSES).to(DEVICE)
ckpt = torch.load(VIDEO_CHECKPOINT, map_location=DEVICE)
video_model.load_state_dict(ckpt["model_state_dict"])
video_model.eval()
print(f"  [VIDEO]  loaded  (val_acc={ckpt.get('best_val_acc',0)*100:.2f}%)")

audio_model = AudioResNet(num_classes=NUM_CLASSES).to(DEVICE)
state = torch.load(AUDIO_CHECKPOINT, map_location=DEVICE)
audio_model.load_state_dict(state)
audio_model.eval()
print(f"  [AUDIO]  loaded")

fusion_model = MultimodalFusionModel(video_model, audio_model, NUM_CLASSES).to(DEVICE)
fusion_ckpt = torch.load(FUSION_CHECKPOINT, map_location=DEVICE)
fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
fusion_model.eval()
print(f"  [FUSION] loaded  (val_acc={fusion_ckpt.get('best_val_acc',0)*100:.2f}%)\n")


# ==============================================================================
# Build paired sample list
# ==============================================================================
video_rows = []
with open(VIDEO_CSV, newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        video_rows.append(r)

audio_df     = pd.read_csv(AUDIO_CSV)
spk_to_specs = {}
for _, row in audio_df.iterrows():
    spk_to_specs.setdefault(row["speaker_id"], []).append(row["spectrogram_path"])

all_labels   = sorted({int(r["label"]) for r in video_rows})
label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}
idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}
spk_to_label = {r["speaker_id"]: int(r["label"]) for r in video_rows}
label_to_spk = {int(r["label"]): r["speaker_id"] for r in video_rows}

spk_spec_cursor = {sid: 0 for sid in spk_to_specs}
paired = []
for r in video_rows:
    sid        = r["speaker_id"]
    cache_path = build_cache_name(r, ROI_CACHE_DIR)
    mp4_path   = VIDEO_DATA_ROOT / fix_path(r["mp4_path"])
    if not cache_path.exists() or not mp4_path.exists():
        continue
    specs = spk_to_specs.get(sid, [])
    if not specs:
        continue
    spec_path = specs[spk_spec_cursor[sid] % len(specs)]
    spk_spec_cursor[sid] += 1
    paired.append({
        "cache_path": cache_path,
        "spec_path":  spec_path,
        "mp4_path":   mp4_path,
        "speaker_id": sid,
        "label":      spk_to_label[sid],
    })

print(f"Demo samples available: {len(paired)}")
print("Controls: SPACE = pause | N = next | Q = quit\n")


# ==============================================================================
# Run inference
# ==============================================================================
def run_inference(sample):
    frames = torch.load(sample["cache_path"], weights_only=True)
    frames_input = frames.unsqueeze(0).to(DEVICE)

    spec = np.load(sample["spec_path"])
    spec_input = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out   = fusion_model(frames_input, spec_input)
        probs = torch.softmax(out, dim=1)
        pred  = out.argmax(dim=1).item()
        conf  = probs[0, pred].item()

    true_idx     = label_to_idx[sample["label"]]
    pred_speaker = label_to_spk[idx_to_label[pred]]
    correct      = pred == true_idx

    return {
        "true_speaker": sample["speaker_id"],
        "pred_speaker": pred_speaker,
        "confidence":   conf,
        "correct":      correct,
    }


# ==============================================================================
# Draw overlay
# ==============================================================================
ROI_PANEL_SIZE = 130  # size of mouth ROI preview box

def draw_overlay(frame, result, mouth_roi):
    h, w         = frame.shape[:2]
    correct      = result["correct"]
    pred_spk     = result["pred_speaker"]
    true_spk     = result["true_speaker"]
    conf         = result["confidence"]
    result_color = (0, 200, 0) if correct else (0, 60, 220)

    # ---- Mouth ROI preview — top right corner ----
    panel_size = ROI_PANEL_SIZE
    panel      = np.zeros((panel_size + 24, panel_size, 3), dtype=np.uint8)

    if mouth_roi is not None:
        roi_resized = cv2.resize(mouth_roi, (panel_size, panel_size))
        panel[:panel_size, :] = roi_resized
    else:
        cv2.putText(panel, "No face", (8, panel_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,100,100), 1)

    cv2.putText(panel, "Mouth ROI", (4, panel_size + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 100), 1)

    # Paste into top right
    px = w - panel_size - 10
    py = 10
    frame[py:py + panel_size + 24, px:px + panel_size] = panel

    # Border around ROI panel
    cv2.rectangle(frame, (px - 1, py - 1),
                  (px + panel_size + 1, py + panel_size + 24),
                  (0, 255, 100), 1)

    # ---- Bottom prediction bar ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 90), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    draw_text(frame, "PREDICTED SPEAKER:", (16, h - 68),
              (180, 180, 180), scale=0.52)
    draw_text(frame, pred_spk, (16, h - 38),
              result_color, scale=1.1, thickness=2)
    tick = "CORRECT" if correct else "WRONG"
    draw_text(frame, f"[{tick}]", (16, h - 12),
              result_color, scale=0.52)

    # Confidence bar
    bx, by, bw, bh = w // 2, h - 72, w // 2 - 20, 20
    fill = int(bw * conf)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50,50,50), -1)
    cv2.rectangle(frame, (bx, by), (bx + fill, by + bh), result_color, -1)
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (150,150,150), 1)
    draw_text(frame, f"Confidence: {conf*100:.1f}%",
              (bx, by - 8), (255,255,255), scale=0.55)

    # True speaker
    draw_text(frame, f"True: {true_spk}",
              (16, 28), (0, 220, 220), scale=0.55, thickness=1)

    # Controls
    draw_text(frame, "N=next  SPACE=pause  Q=quit",
              (w - 270, h - 8), (100,100,100), scale=0.42, thickness=1)

    return frame


# ==============================================================================
# Main loop
# ==============================================================================
sample_indices = list(range(len(paired)))
random.shuffle(sample_indices)
cursor      = 0
paused      = False
current_mp4 = None

def load_next(idx):
    global current_mp4
    stop_audio()
    sample      = paired[sample_indices[idx % len(sample_indices)]]
    result      = run_inference(sample)
    cap         = cv2.VideoCapture(str(sample["mp4_path"]))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    current_mp4 = sample["mp4_path"]
    play_audio(current_mp4)

    status = "✅" if result["correct"] else "❌"
    print(f"  Sample {idx+1}: True={result['true_speaker']} | "
          f"Pred={result['pred_speaker']} "
          f"({result['confidence']*100:.1f}%) {status}")

    return cap, fps, result


cv2.namedWindow("Multimodal Speaker Identification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Multimodal Speaker Identification", 900, 540)

cap, fps, result = load_next(cursor)
mouth_roi = None

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            stop_audio()
            play_audio(current_mp4)
            ret, frame = cap.read()
            if not ret:
                break

        # Detect face and mouth ROI on every frame
        frame, mouth_roi = detect_mouth_roi(frame)
        frame = draw_overlay(frame, result, mouth_roi)
        cv2.imshow("Multimodal Speaker Identification", frame)

    delay = max(1, int(1000 / fps))
    key   = cv2.waitKey(delay) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('n'):
        cap.release()
        cursor += 1
        cap, fps, result = load_next(cursor)
        paused = False
        mouth_roi = None
    elif key == ord(' '):
        paused = not paused
        if paused:
            stop_audio()
        else:
            play_audio(current_mp4)

stop_audio()
cap.release()
cv2.destroyAllWindows()
print("Demo closed.")