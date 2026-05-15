# MouthVideoCNN Notebook
This project trains and evaluates a video-based speaker identification model. The notebook loads short video clips, extracts mouth-region frame sequences, trains a custom CNN model, and compares it with a ResNet-18 baseline.

The main notebook is:

```
VideoModel\_v7.ipynb
```

## Dataset
The original dataset is VoxCeleb2 from Hugging Face:

```
https://huggingface.co/datasets/Reverb/voxceleb2
```

The full dataset is large, so this project expects a smaller local subset. Create that subset by running:
```
python make\_subset.py
```

After running the script, the project should contain this folder:
```
vox\_celeb\_subset/
├── subset.csv
└── ...video files copied or arranged by make\_subset.py
```

The notebook expects `subset.csv` to exist at:
```
./vox\_celeb\_subset/subset.csv
```

The notebook also expects video paths in `subset.csv` to be relative to:
```
./vox\_celeb\_subset/
```

The CSV must include at least these columns:

```
mp4\_path
label
speaker\_id
```

## Project layout

Recommended file layout:
```
project-folder/
├── VideoModel\_v7.ipynb
├── make\_subset.py
├── README.md
├── vox\_celeb\_subset/
│   ├── subset.csv
│   └── ...subset video files
├── saved\_models/
├── roi\_cache/
└── facedetection\_model/
```

The `saved\_models/`, `roi\_cache/`, and `facedetection\_model/` folders are created or used by the notebook.

## Environment setup
Create and activate a virtual environment.

On macOS or Linux:
```
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:
```
pip install jupyterlab notebook numpy matplotlib opencv-python mediapipe torch torchvision torchmetrics
```

If you want GPU support, install the PyTorch build that matches your CUDA version. The standard `pip install torch torchvision` command may install a CPU-only build depending on your system.



