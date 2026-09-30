# Quantum Network HyperConsole v8 — Desktop

The desktop package is the full CustomTkinter/PennyLane/CQSE/TCQH application
from the conversation, packaged with install metadata and database schema.

## Recommended environment

Python 3.11 or 3.12 in a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
python quantum_network_hyperconsole_v8.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock.txt
python .\quantum_network_hyperconsole_v8.py
```

## Optional dataset support

```bash
pip install -r requirements-datasets.txt
```

Real Kaggle ingestion is explicit and tabular-only. The application rejects
pickle/joblib/notebooks/scripts/binaries/PCAP/model-object formats from that
ingestion path.

## Gemma 4 E2B

The `.litertlm` model is intentionally handled through a LiteRT-LM runtime, not
through llama.cpp. The application checks the configured SHA-256 before using
the model. LiteRT-LM installation is platform-specific, so it is not forced by
the base requirements file.

## Legacy local GGML fallback

The exact historical GGMLv3 Llama-2 model requested earlier needs an old
`llama-cpp-python` binding. Install `requirements-legacy-ggml.txt` only in an
isolated environment if that fallback is needed.

## CLI

```bash
python quantum_network_hyperconsole_v8.py --print-schemas
python quantum_network_hyperconsole_v8.py --init-db
python quantum_network_hyperconsole_v8.py --setup-weaviate
python quantum_network_hyperconsole_v8.py --seed-safe-datasets
python quantum_network_hyperconsole_v8.py --verify-gemma4
python quantum_network_hyperconsole_v8.py --cqse-demo "periodic DNS anomaly"
```

The primary GUI entry point remains `Continue().run()`.


## Dependency security

See `DEPENDENCY_LOCK.md` for the exact reviewed pins and Contriever safetensors revision.
