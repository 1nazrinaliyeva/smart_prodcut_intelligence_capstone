# Smart Product Intelligence Capstone

End-to-end deep learning capstone for the Amazon Reviews 2023 `All_Beauty`
category. The project follows the milestone structure from the brief: tabular
MLP, computer vision, text embeddings, transformers, LLM/RAG, diffusion, and a
final Streamlit demo.

## Project Structure

```text
smart-product-intelligence/
  README.md
  requirements.txt
  data/                         # cached subsets + images (gitignored)
  notebooks/
    00_eda.ipynb
    01_tabular_mlp.ipynb
    02_vision_cnn_transfer.ipynb
    03_text_embeddings.ipynb
    04_transformers.ipynb
    05_llm_rag_finetune.ipynb
    06_diffusion.ipynb
  src/                          # reusable code shared across milestones
    data.py
    models.py
    utils.py
  app/                          # Milestone 7 Streamlit demo
    app.py
  reports/                      # metrics, comparisons, generated summaries
  models/                       # trained local artifacts
```

## Setup

Clone the repository first:

```bash
git clone https://github.com/1nazrinaliyeva/smart_prodcut_intelligence_capstone.git
cd smart_prodcut_intelligence_capstone
```

Create and activate a virtual environment.

macOS / Linux:

```bash
python3 -m venv notebooks/.venv
source notebooks/.venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Windows PowerShell:

```powershell
py -m venv notebooks\.venv
notebooks\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run Notebooks

Run the notebooks in milestone order:

```text
notebooks/00_eda.ipynb
notebooks/01_tabular_mlp.ipynb
notebooks/02_vision_cnn_transfer.ipynb
notebooks/03_text_embeddings.ipynb
notebooks/04_transformers.ipynb
notebooks/05_llm_rag_finetune.ipynb
notebooks/06_diffusion.ipynb
```

## Run Demo App

After setup, the simplest command is the same on macOS, Linux, and Windows:

```bash
smart-product-intelligence
```

Then open:

```text
http://127.0.0.1:8501
```

You can also run Streamlit directly.

macOS / Linux:

```bash
streamlit run app/app.py
```

Windows PowerShell:

```powershell
streamlit run app\app.py
```

Or use the Python module entrypoint.

macOS / Linux:

```bash
python3 -m app.cli --port 8501
```

Windows PowerShell:

```powershell
py -m app.cli --port 8501
```

The app can also be started without manually cloning the repository if `uvx` or
`pipx` is installed.

```bash
uvx --from git+https://github.com/1nazrinaliyeva/smart_prodcut_intelligence_capstone.git smart-product-intelligence --port 8501
```

or:

```bash
pipx run --spec git+https://github.com/1nazrinaliyeva/smart_prodcut_intelligence_capstone.git smart-product-intelligence --port 8501
```

The app reads the saved data, reports, and model outputs produced by the
milestone notebooks. If an artifact is missing, the app shows a clear message
instead of failing.

Useful CLI options:

macOS / Linux:

```bash
python3 -m app.cli --host 0.0.0.0 --port 8501
python3 -m app.cli --project-root /path/to/smart-product-intelligence
python3 -m app.cli --processed-dir data/processed --reports-dir reports --models-dir models
```

Windows PowerShell:

```powershell
py -m app.cli --host 0.0.0.0 --port 8501
py -m app.cli --project-root C:\path\to\smart-product-intelligence
py -m app.cli --processed-dir data\processed --reports-dir reports --models-dir models
```

## Docker

Build and run:

```bash
docker build -t smart-product-intelligence .
docker run --rm -p 8501:8501 smart-product-intelligence
```

If you want the Docker app to read local generated artifacts, mount the local
`data`, `reports`, and `models` directories.

macOS / Linux:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/models:/app/models" \
  smart-product-intelligence
```

Windows PowerShell:

```powershell
docker run --rm -p 8501:8501 `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\reports:/app/reports" `
  -v "${PWD}\models:/app/models" `
  smart-product-intelligence
```

Windows Command Prompt:

```bat
docker run --rm -p 8501:8501 ^
  -v "%cd%\data:/app/data" ^
  -v "%cd%\reports:/app/reports" ^
  -v "%cd%\models:/app/models" ^
  smart-product-intelligence
```
