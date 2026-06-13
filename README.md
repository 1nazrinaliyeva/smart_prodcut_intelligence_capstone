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

```bash
python3 -m venv notebooks/.venv
source notebooks/.venv/bin/activate
pip install -r requirements.txt
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

```bash
streamlit run app/app.py
```

Or use the CLI entrypoint:

```bash
python3 -m app.cli --port 8501
```

After the project is pushed to GitHub, it can also be started without manually
cloning the repository:

```bash
uvx --from git+https://github.com/<your-user>/<your-repo>.git smart-product-intelligence --port 8501
```

or:

```bash
pipx run --spec git+https://github.com/<your-user>/<your-repo>.git smart-product-intelligence --port 8501
```

Open:

```text
http://127.0.0.1:8501
```

The app reads the saved data, reports, and model outputs produced by the
milestone notebooks. If an artifact is missing, the app shows a clear message
instead of failing.

Useful CLI options:

```bash
python3 -m app.cli --host 0.0.0.0 --port 8501
python3 -m app.cli --project-root /path/to/smart-product-intelligence
python3 -m app.cli --processed-dir data/processed --reports-dir reports --models-dir models
```

## Docker

```bash
docker build -t smart-product-intelligence .
docker run --rm -p 8501:8501 smart-product-intelligence
```

If you want the Docker app to read local generated artifacts:

```bash
docker run --rm -p 8501:8501 \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/models:/app/models" \
  smart-product-intelligence
```
