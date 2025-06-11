# 🎬 AKAI Space Task – Video-to-Text Captioning Pipeline

This project provides a streamlined video-to-text captioning interface using the [BLIP model](https://huggingface.co/Salesforce/blip-image-captioning-base). It allows both automatic caption generation from videos and comparison against human-labeled data.

---

## ✨ Features

- Extracts the **middle frame** from each video
- Uses **BLIP** (Bootstrapped Language Image Pretraining) for captioning
- Compares AI-generated captions with **human-labeled captions**
- Supports:
  - 📁 Local video folder (`videos/`)
  - 📤 Uploading new video files
- Downloadable results as `.csv`

---

## Deploy

- pip install -r requirements.txt
- streamlit run sample.py


