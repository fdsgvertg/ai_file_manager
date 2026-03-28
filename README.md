# AI File Manager

An intelligent, local-first file organization system that scans, understands, and restructures files (images, videos, audio, PDFs, documents) using lightweight multimodal AI models.

---

## 🚀 Overview

AI File Manager automates file organization by:

- Scanning folders recursively
- Extracting semantic meaning from files (text, images, audio, video)
- Grouping files by **topic, date, or semantic similarity**
- Creating structured folders
- Moving files safely with undo support

Designed to run **locally** on modest hardware (RTX 2050, 4GB VRAM).

---

## 🧠 Core Capabilities

| Feature | Description |
|--------|------------|
| Multimodal Analysis | Handles images, PDFs, audio, video, and documents |
| AI Classification | Uses LLM + embeddings to group files |
| Smart Folder Creation | Automatically generates meaningful folder structures |
| Undo System | Restore files to original locations |
| Windows Integration | Right-click → "Sort with AI File Manager" |

---

## 📁 Project Structure

```
ai_file_manager/
├── config/
├── core/
├── pipelines/
├── models/
├── api/
├── utils/
├── windows/
├── logs/
├── models/weights/
├── cli.py
└── requirements.txt
```

---

## ⚙️ System Requirements

### Minimum Hardware
- GPU: NVIDIA MX 450 VRAM 2GB
- RAM: 8GB+ recommended

### Software
- Python 3.10+
- CUDA 12.1
- FFmpeg
- Tesseract OCR

---

## 🛠️ Installation

### 1. Clone & Setup Environment

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Enable GPU Acceleration

```bash
pip install llama-cpp-python ^
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 ^
  --force-reinstall

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

### 3. Install External Dependencies

#### FFmpeg
- Download and add `bin/` to PATH

#### Tesseract OCR
- Install and ensure it's accessible via PATH  

---

## 🧠 Model Setup

### Required Model (Manual)

Download:
Phi-3-mini-4k-instruct-q4.gguf (~2.2GB)

Place in:
models/weights/

---

### Auto-Downloaded Models (on first run)

- Qwen2-VL (vision)
- Whisper (audio transcription)
- BGE-small (embeddings)

---

## ⚙️ Configuration

Edit:
config/settings.yaml

---

## ▶️ Usage

### CLI Mode

```bash
python cli.py --path "C:\Your\Folder" --mode topic
```

### Modes

| Mode | Behavior |
|------|--------|
| topic | Groups by semantic meaning |
| date | Groups by file timestamps |
| relation | Clusters similar files |

---

### Example

```bash
python cli.py --path "D:\Downloads" --mode topic
```

---

## 🖱️ Windows Explorer Integration

### Install Right-Click Menu

```powershell
cd windows
.\install.ps1
```

---

### Uninstall

windows/uninstall_context_menu.reg

---

## 🔄 Undo System

```bash
python cli.py --undo
```

---

## 🔌 API Mode (Optional)

```bash
uvicorn api.server:app --reload
```

---

## 🧩 How It Works (Pipeline)

1. Scanner
2. Router
3. Pipelines (Image, PDF, Audio, Video, Docs)
4. LLM Processing
5. Embeddings + Clustering
6. Organizer
7. Undo Manager

---

## ⚠️ Common Issues

### CUDA Not Working
```python
import torch
print(torch.cuda.is_available())
```

---

## 📌 Summary

A **local multimodal AI pipeline + clustering engine + file system orchestrator**.
