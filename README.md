# 🎬 AKAI Space Task – Video-to-Text Captioning Pipeline

This project provides a simple and efficient pipeline for **automated video caption generation** using **BLIP (Bootstrapped Language Image Pretraining)** and a comparison with **human-labeled captions**.

Streamlit is used for a clean UI to support:
- Loading and captioning videos from a local folder (`videos/`)
- Uploading videos and generating AI-based captions
- Comparing AI vs. human annotations (from `human.csv`)

---

## 🚀 Features

- Extracts the **middle frame** from each video
- Generates image captions using [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base)
- Displays human-labeled and AI-generated captions side-by-side
- Allows CSV download of results

---

## 📁 Directory Structure

