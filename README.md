# Speaker Diarization & Identification — Raspberry Pi 5

On-device speaker diarization ("who spoke when") and open-set speaker
identification ("who is speaking"), running fully on a Raspberry Pi 5.

## Pipeline

```
Mic -> VAD (trim silence) -> Sliding window segmentation (1.5s window, 0.75s hop)
    -> ECAPA-TDNN embedding extraction -> Clustering (agglomerative/spectral)
    -> Diarization output -> Speaker-DB match (cosine similarity) -> Identification
```

## Project structure

- `vad/` — Voice Activity Detection (Stage 1)
- `embeddings/` — ECAPA-TDNN speaker embedding extraction (Stage 2)
- `clustering/` — Speaker clustering (Stage 3)
- `identification/` — Open-set speaker identification against enrolled DB (Stage 4)
- `data/` — Test audio files (raw recordings/datasets are gitignored — see `.gitignore`)
- `scripts/` — Utility/benchmark scripts

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Status

- [x] Stage 1: VAD (`vad/vad_test.py`)
- [ ] Stage 2: ECAPA-TDNN embeddings
- [ ] Stage 3: Clustering
- [ ] Stage 4: Open-set identification
- [ ] Live mic capture + real-time pipeline (Pi 5, lab)
- [ ] Benchmarking: DER, JER, RTF, identification accuracy, power
