# AI File Manager — Setup & Usage Guide

> **Target hardware:** Windows 10/11 · NVIDIA RTX 2050 (4 GB VRAM) · Python 3.10+

---

## 📁 Project Structure

```
ai_file_manager/
├── config/
│   └── settings.yaml          ← All configuration (model paths, thresholds, etc.)
│
├── core/
│   ├── scanner.py             ← Recursive folder scanning → FileManifest
│   ├── router.py              ← Dispatches files to correct pipeline
│   ├── organizer.py           ← Builds folder plan & moves files
│   └── undo_manager.py        ← Session-based undo / restore
│
├── pipelines/
│   ├── image_pipeline.py      ← Qwen2-VL vision → text description
│   ├── pdf_pipeline.py        ← PyMuPDF + Tesseract OCR → text
│   ├── audio_pipeline.py      ← faster-whisper transcription
│   ├── video_pipeline.py      ← FFmpeg frames → vision model
│   └── document_pipeline.py   ← .txt/.docx/.xlsx/.pptx → text
│
├── models/
│   ├── llm_client.py          ← Phi-3 Mini (GGUF) metadata extraction
│   ├── vision_client.py       ← Qwen2-VL-2B image description
│   └── embedding_client.py    ← BGE-small embeddings + FAISS clustering
│
├── api/
│   └── server.py              ← FastAPI REST API (background job queue)
│
├── utils/
│   ├── config.py              ← YAML config loader (dot-access)
│   ├── logger.py              ← Unified logging (console + file)
│   ├── file_utils.py          ← Path helpers, safe_move, slugify
│   └── validator.py           ← Strict LLM JSON extraction & validation
│
├── windows/
│   ├── sort_files.bat         ← Explorer launcher (called by registry)
│   ├── install_context_menu.reg   ← Adds right-click menu
│   ├── uninstall_context_menu.reg ← Removes right-click menu
│   └── install.ps1            ← Automated PowerShell installer
│
├── logs/                      ← Auto-created; daily log files
├── models/weights/            ← Place downloaded GGUF model here
├── cli.py                     ← Main CLI entry point
└── requirements.txt
```

---

## ⚙️ Step 1 — System Prerequisites

### Python 3.10+
```
https://www.python.org/downloads/
```
Ensure `python` is on your PATH.

### CUDA Toolkit 12.1 (for GPU acceleration)
```
https://developer.nvidia.com/cuda-12-1-0-download-archive
```

### FFmpeg (required for video processing)
```
https://ffmpeg.org/download.html
```
Download the Windows build, extract, and add the `bin/` folder to your PATH.

### Tesseract OCR (required for scanned PDFs)
```
https://github.com/UB-Mannheim/tesseract/wiki
```
Install the Windows installer. Default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

Add to PATH, or configure in `settings.yaml`.

---

## 📦 Step 2 — Python Environment

```batch
:: Create a dedicated virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

:: Install CPU baseline packages first
pip install -r requirements.txt

:: Re-install llama-cpp-python with CUDA support (RTX 2050 = sm_86)
pip install llama-cpp-python ^
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 ^
    --force-reinstall

:: Install PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

---

## 🧠 Step 3 — Download Model Weights

### Phi-3 Mini 4K Instruct (GGUF, 4-bit quantized)
```
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
```
Download: `Phi-3-mini-4k-instruct-q4.gguf` (~2.2 GB)

Place at: `models/weights/phi-3-mini-4k-instruct-q4.gguf`

### Qwen2-VL 2B Instruct
Downloaded automatically on first run via HuggingFace (`Qwen/Qwen2-VL-2B-Instruct`).
To pre-download:
```python
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
```

### Whisper Small
Downloaded automatically on first run via faster-whisper.

### BGE-Small Embeddings
Downloaded automatically on first run via sentence-transformers.

---

## 🖱️ Step 4 — Windows Explorer Integration

### Option A: Automated (PowerShell, run as Administrator)
```powershell
# Open PowerShell as Administrator
cd C:\ai_file_manager
powershell -ExecutionPolicy Bypass -File windows\install.ps1 `
    -InstallDir "C:\ai_file_manager" `
    -PythonExe "C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe"
```

### Option B: Manual
1. Open `windows\install_context_menu.reg` in Notepad
2. Replace ALL occurrences of `C:\\ai_file_manager` with your actual install path (use double backslashes)
3. Replace `sort_files.bat` path similarly
4. Save and double-click the `.reg` file → click Yes

### Verify
Right-click any folder in Windows Explorer. You should see **"Sort using AI File Manager"**.

---

## 🚀 Running the System

### Via Windows Explorer (recommended)
1. Right-click any folder
2. Select **"Sort using AI File Manager"**
3. Choose mode: 1 = Topic, 2 = Date, 3 = Relation
4. Review the preview
5. Confirm to execute

### Via CLI
```batch
:: Activate your environment first
venv\Scripts\activate

:: Scan a folder (no AI — just statistics)
python cli.py scan "C:\Users\Dash\Downloads"

:: Preview sort by topic (DRY RUN — no files moved)
python cli.py sort "C:\Users\Dash\Downloads" --mode topic

:: Execute sort by topic
python cli.py sort "C:\Users\Dash\Downloads" --mode topic --execute

:: Sort by date
python cli.py sort "C:\Users\Dash\Downloads" --mode date --execute

:: Sort by semantic relation (clusters similar files)
python cli.py sort "C:\Users\Dash\Downloads" --mode relation --execute

:: Undo the last sort
python cli.py undo "C:\Users\Dash\Downloads"

:: List all undo sessions
python cli.py undo "C:\Users\Dash\Downloads" --list

:: Undo a specific session
python cli.py undo "C:\Users\Dash\Downloads" --session-id 20240315_143022
```

### Via REST API (for integrations / GUI frontends)
```batch
python cli.py server
:: Server starts at http://127.0.0.1:8765
:: API docs at   http://127.0.0.1:8765/docs
```

Example API calls:
```bash
# Scan a folder
curl -X POST http://127.0.0.1:8765/scan \
  -H "Content-Type: application/json" \
  -d '{"folder": "C:\\Users\\Dash\\Downloads"}'

# Start a sort job (preview only)
curl -X POST http://127.0.0.1:8765/sort \
  -H "Content-Type: application/json" \
  -d '{"folder": "C:\\Users\\Dash\\Downloads", "mode": "topic", "execute": false}'

# Check job status
curl http://127.0.0.1:8765/jobs/{job_id}

# Undo last sort
curl -X POST http://127.0.0.1:8765/undo \
  -H "Content-Type: application/json" \
  -d '{"folder": "C:\\Users\\Dash\\Downloads"}'
```

---

## 📊 Example Output

### Preview (before execution)
```
============================================================
📁  FILE SORT PREVIEW
============================================================

  C:\Downloads\Travel_Photos\Images/
    └─ beach_vacation_2023.jpg
    └─ eiffel_tower_night.png

  C:\Downloads\Work_Projects\PDFs/
    └─ quarterly_report_q3.pdf
    └─ project_proposal_final.pdf

  C:\Downloads\Python_Code\Documents/
    └─ data_analysis.py
    └─ ml_pipeline.ipynb

  C:\Downloads\Miscellaneous\Audio/
    └─ voice_memo_unknown.m4a

============================================================
Total: 7 files across 4 folders
```

### LLM Metadata for a file
```json
{
  "topic": "Travel Photos",
  "summary": "Image shows the Eiffel Tower illuminated at night with crowds below",
  "keywords": ["eiffel tower", "paris", "travel", "night photography", "landmark"],
  "confidence": 0.95,
  "file_date": null
}
```

---

## 🔧 Configuration Tuning for RTX 2050 (4 GB VRAM)

Edit `config/settings.yaml`:

```yaml
models:
  llm:
    n_gpu_layers: 20    # Start here; increase to 32 if VRAM allows
    n_ctx: 2048         # Reduce to 2048 if OOM errors occur

  vision:
    load_in_4bit: true  # REQUIRED for 4 GB VRAM
    max_new_tokens: 128 # Reduce if VRAM errors

  audio:
    model_size: "base"  # Use "base" instead of "small" to save ~200 MB VRAM

concurrency:
  gpu_semaphore: 1      # Keep at 1 — only one GPU task at a time
  max_workers: 2        # Reduce to 2 if CPU bottlenecked
```

---

## 🐞 Troubleshooting

| Problem | Solution |
|---|---|
| `CUDA out of memory` | Reduce `n_gpu_layers`, enable 4-bit quant, reduce `max_new_tokens` |
| `llama-cpp-python not found` | Reinstall with CUDA: see Step 2 |
| `tesseract is not installed` | Install Tesseract and add to PATH |
| `ffmpeg not found` | Install FFmpeg and add `bin/` to PATH |
| Right-click menu missing | Re-run `install.ps1` as Administrator |
| LLM returns garbled JSON | Validator handles this; check logs for pattern |
| Slow processing | Use `--flat` flag; reduce `n_gpu_layers`; use smaller Whisper model |

---

## 📋 Logs

Daily log files are written to the `logs/` directory:
```
logs/ai_file_manager_20240315.log
```

Set `log_level: DEBUG` in `settings.yaml` for verbose output.
