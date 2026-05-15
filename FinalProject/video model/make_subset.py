import csv
import random
import shutil
from pathlib import Path

# Metadata CSV from VoxCeleb2
META_CSV = Path("vox2_meta.csv")
# Original full dataset roots
AAC_ROOT = Path("aac")
MP4_ROOT = Path("mp4")
# New smaller subset folder we are creating
OUT_ROOT = Path("vox_celeb_subset")
# How many speakers we want in the subset
NUM_SPEAKERS = 60
# How many clips per speaker
CLIPS_PER_SPEAKER = 20
# Random seed so results are reproducible
SEED = 611

print("> Loading speakers from CSV...")

# Store speaker IDs that belong to the dev split
dev_speakers = set()

with open(META_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        speaker_id = row["VoxCeleb2 ID "].strip()
        set_type = row["Set "].strip()

        if set_type == "dev":
            dev_speakers.add(speaker_id)

print(f"> Loaded {len(dev_speakers)} dev speakers")

print("> Scanning speaker folders...")

random.seed(SEED)

# Find speaker folders that exist in the mp4 dataset and are part of dev
speaker_dirs = [p for p in MP4_ROOT.iterdir() if p.is_dir() and p.name in dev_speakers]
random.shuffle(speaker_dirs)

# pairs[speaker] = list of valid matched clips for that speaker
pairs = {}

# List of speakers who have enough usable clips
eligible = []

for idx, speaker_dir in enumerate(speaker_dirs, start=1):
    speaker = speaker_dir.name
    clips = []

    # Search through this speaker's video files
    for mp4 in speaker_dir.rglob("*.mp4"):
        # Path relative to MP4_ROOT, for example:
        # id01840/0nJOuiOV2RM/00004.mp4
        rel = mp4.relative_to(MP4_ROOT)

        # Matching audio file should exist under AAC_ROOT with .m4a suffix
        aac = AAC_ROOT / rel.with_suffix(".m4a")

        # Only keep clip if both video and audio exist
        if aac.exists():
            clips.append((aac, mp4, rel))

        # Stop early once we have enough clips for this speaker
        if len(clips) >= CLIPS_PER_SPEAKER:
            break

    # Only keep speakers with enough matched clips
    if len(clips) >= CLIPS_PER_SPEAKER:
        pairs[speaker] = clips
        eligible.append(speaker)
        print(f"> Eligible speaker {len(eligible)}/{NUM_SPEAKERS}: {speaker}")

    if idx % 100 == 0:
        print(f"> Checked {idx} speaker folders...")

    # Stop once we have enough speakers
    if len(eligible) >= NUM_SPEAKERS:
        break

if len(eligible) < NUM_SPEAKERS:
    raise RuntimeError(
        f"Only found {len(eligible)} speakers with at least {CLIPS_PER_SPEAKER} matched clips."
    )

selected = eligible[:NUM_SPEAKERS]

print("> Copying files to new subset directory...")

rows = []
copied = 0
total = NUM_SPEAKERS * CLIPS_PER_SPEAKER

for label, speaker in enumerate(selected):
    # Randomly choose the final clips for this speaker
    clips = random.sample(pairs[speaker], CLIPS_PER_SPEAKER)

    print(f"> Copying speaker {label + 1}/{NUM_SPEAKERS}: {speaker}")

    for aac, mp4, rel in clips:
        # Actual destination paths on disk
        out_aac = OUT_ROOT / "aac" / rel.with_suffix(".m4a")
        out_mp4 = OUT_ROOT / "mp4" / rel

        # Create parent folders if needed
        out_aac.parent.mkdir(parents=True, exist_ok=True)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        # Copy files into subset folder
        shutil.copy(aac, out_aac)
        shutil.copy(mp4, out_mp4)

        copied += 1

        # Save paths RELATIVE to OUT_ROOT in the CSV.
        rows.append({
            "speaker_id": speaker,
            "label": label,
            "aac_path": str(Path("aac") / rel.with_suffix(".m4a")),
            "mp4_path": str(Path("mp4") / rel),
        })

        if copied % 100 == 0 or copied == total:
            print(f"> Copied {copied}/{total} files")

print("> Saving new metadata CSV...")

OUT_ROOT.mkdir(exist_ok=True)

with open(OUT_ROOT / "subset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["speaker_id", "label", "aac_path", "mp4_path"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Done: {len(rows)} samples copied to '{OUT_ROOT}'")

# Final check, print out a line from the new csv to make sure its good
CSV_PATH = Path("./vox_celeb_subset/subset.csv")
DATA_ROOT = Path("./vox_celeb_subset")

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        p = DATA_ROOT / Path(row["mp4_path"])
        print(row["mp4_path"], "->", p.exists())
        if i == 4:
            break