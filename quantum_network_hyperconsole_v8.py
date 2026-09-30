#!/usr/bin/env python3
"""
Quantum Network HyperConsole 8.0
================================

Single-file defensive network observability + AI + vector-memory console.

Core stack
----------
- CustomTkinter desktop GUI
- asyncio background runtime (Tk never owns the async event loop)
- aiosqlite WAL-mode telemetry/action/analysis/cache database
- PennyLane adaptive 4-qubit feature circuit
- OpenAI Daybreak Blue through:
    1) official Codex CLI / Sign in with ChatGPT boundary
    2) Responses API (approved Daybreak project)
    3) standard GPT-6 Sol API fallback
- Structured protocol:
    [status], [observation], [evidence], [risk], [action],
    [replytemplate], [confidence], [final]
- Weaviate async memory surface with FIVE distinct schemas
- Named vector spaces:
    openai_3072      -> text-embedding-3-large
    fb_contriever_768 -> facebook/contriever-msmarco
    quantum_16       -> PennyLane probability vector, where meaningful
- RRF memory retrieval across BM25 + OpenAI vector + Facebook vector
  + quantum similarity when available
- Session-only secrets: API keys are never written to app_config.json
- Optional legacy local Llama-2 GGMLv3 fallback with pinned SHA-256

Recommended Python
------------------
Python 3.11 is recommended for the modern application stack.

Install:
    pip install \
      customtkinter aiosqlite httpx numpy pennylane matplotlib psutil \
      "weaviate-client>=4.16.4" torch transformers pyarrow litert-lm kaggle

Optional legacy local GGML fallback:
    pip install "llama-cpp-python==0.1.78"

Environment variables (all optional)
------------------------------------
OPENAI_API_KEY              Normal OpenAI API key; use an approved Daybreak project
WEAVIATE_URL                Weaviate Cloud cluster URL
WEAVIATE_API_KEY            Weaviate Cloud key
QNET_DB                     SQLite path
QNET_FB_EMBEDDINGS=0|1      Enable/disable local Facebook Contriever (default 1)
QNET_OPENAI_EMBEDDINGS=0|1  Enable/disable OpenAI embeddings (default 1)

Local Weaviate:
    If WEAVIATE_URL is absent, the app connects to the standard local
    Weaviate ports (HTTP 8080 / gRPC 50051).

Codex authentication
--------------------
The app NEVER reads ~/.codex/auth.json, browser cookies, OAuth refresh tokens,
or ChatGPT session tokens. It invokes the official `codex` binary and lets Codex
own its authentication boundary.

Entry points
------------
    Continue().run()
or:
    run()

CLI:
    python quantum_network_hyperconsole.py --print-schemas
    python quantum_network_hyperconsole.py --init-db
    python quantum_network_hyperconsole.py --setup-weaviate
    python quantum_network_hyperconsole_v8.py --print-requirements
    python quantum_network_hyperconsole_v8.py --seed-safe-datasets
    python quantum_network_hyperconsole_v8.py --verify-gemma4
    python quantum_network_hyperconsole_v8.py --download-gemma4
    python quantum_network_hyperconsole_v8.py --cqse-demo "suspicious beaconing with DNS anomalies"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import colorsys
import io
import random
import shlex
import zipfile
import csv
import hashlib
import json
import logging
import math
import os
import platform
import queue
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import aiosqlite
import customtkinter as ctk
import httpx
import numpy as np
import pennylane as qml
import psutil
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import filedialog, messagebox, ttk

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_IMPORT_OK = True
except Exception:
    weaviate = None
    Configure = DataType = Property = Auth = MetadataQuery = None
    WEAVIATE_IMPORT_OK = False


# =============================================================================
# Constants / paths
# =============================================================================

APP_NAME = "Quantum Network HyperConsole"
APP_VERSION = "8.0"
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "hyperconsole_config.json"
LOG_PATH = BASE_DIR / "hyperconsole.log"
DEFAULT_DB_PATH = Path(os.getenv("QNET_DB", str(BASE_DIR / "network_data.db")))
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

DAYBREAK_ALIAS = "gpt-daybreak-blue-latest"
STANDARD_MODEL = "gpt-6-sol"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
FB_EMBED_MODEL = "facebook/contriever-msmarco"
FB_EMBED_REVISION = "287db3b2d7662e688c051c351a0739de4ef2d6d7"
FB_EMBED_SAFETENSORS_SHA256 = "185c2c743bbb9ec15727575a931dd95fdf035e1fc377dd846a9a4f9fa89dd54b"
FB_EMBED_DIM = 768

RESPONSES_URL = "https://api.openai.com/v1/responses"
EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

LEGACY_MODEL_NAME = "llama-2-7b-chat.ggmlv3.q8_0.bin"
LEGACY_MODEL_REVISION = "be241fc7eee78d12264466e56e016dea4d01eeb6"
LEGACY_MODEL_URL = (
    "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/"
    f"{LEGACY_MODEL_REVISION}/{LEGACY_MODEL_NAME}?download=true"
)
LEGACY_MODEL_SHA256 = "3bfdde943555c78294626a6ccd40184162d066d39774bd2c98dae24943d32cc3"
LEGACY_MODEL_SIZE = 7_160_799_872
LEGACY_MODEL_PATH = MODEL_DIR / LEGACY_MODEL_NAME
LEGACY_PART_PATH = MODEL_DIR / f"{LEGACY_MODEL_NAME}.part"

HTTP_TIMEOUT = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=30.0)

VECTOR_OPENAI = "openai_3072"
VECTOR_FB = "fb_contriever_768"
VECTOR_QUANTUM = "quantum_16"
VECTOR_CHROMA = "chromatic_surface_96"
VECTOR_CHROMA_QUANTUM = "chromatic_quantum_32"

GEMMA4_MODEL_NAME = "gemma-4-E2B-it.litertlm"
# Exact mirror selected because it publishes the SHA supplied by the user.
GEMMA4_REPO = "multimodel/gemma-4-E2B-it-litert-lm"
GEMMA4_MODEL_URL = (
    "https://huggingface.co/multimodel/gemma-4-E2B-it-litert-lm/resolve/"
    "main/gemma-4-E2B-it.litertlm?download=true"
)
GEMMA4_MODEL_SHA256 = "ab7838cdfc8f77e54d8ca45eadceb20452d9f01e4bfade03e5dce27911b27e42"
GEMMA4_MODEL_PATH = MODEL_DIR / GEMMA4_MODEL_NAME
GEMMA4_PART_PATH = MODEL_DIR / f"{GEMMA4_MODEL_NAME}.part"
# The upstream/mirror pages describe this as approximately 2.58 GB.
# Exact size is learned from Content-Length/Content-Range and then pinned in the local receipt.
GEMMA4_RECEIPT_PATH = MODEL_DIR / f"{GEMMA4_MODEL_NAME}.receipt.json"

DATASET_SANDBOX_DIR = BASE_DIR / "dataset_sandbox"
DATASET_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
DATASET_MAX_MEMBER_BYTES = 768 * 1024 * 1024
DATASET_MAX_ROWS_REAL = 250_000
DATASET_SIM_ROWS_PER_SOURCE = 320


# =============================================================================
# Five Weaviate memory schemas
# =============================================================================

WEAVIATE_SCHEMAS: dict[str, dict[str, Any]] = {
    "QNTelemetryMemoryV1": {
        "purpose": "Dense historical network state, trends, anomalies, and quantum signatures.",
        "quantum": True,
        "chromatic": True,
        "properties": [
            ("event_id", "TEXT"),
            ("timestamp", "TEXT"),
            ("summary", "TEXT"),
            ("ping_ms", "NUMBER"),
            ("jitter_ms", "NUMBER"),
            ("download_mbps", "NUMBER"),
            ("upload_mbps", "NUMBER"),
            ("anomaly_count", "INT"),
            ("health_state", "TEXT"),
            ("trends_json", "TEXT"),
            ("quantum_json", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNIncidentMemoryV1": {
        "purpose": "Incident hypotheses, evidence, severity, ownership, state transitions, and outcomes.",
        "quantum": False,
        "chromatic": False,
        "properties": [
            ("incident_id", "TEXT"),
            ("opened_at", "TEXT"),
            ("updated_at", "TEXT"),
            ("severity", "TEXT"),
            ("title", "TEXT"),
            ("evidence", "TEXT"),
            ("hypothesis", "TEXT"),
            ("status", "TEXT"),
            ("source", "TEXT"),
            ("resolution", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNTrafficFlowMemoryV1": {
        "purpose": "Flow-level top talkers and statistical traffic anomalies for defensive triage.",
        "quantum": False,
        "chromatic": False,
        "properties": [
            ("flow_id", "TEXT"),
            ("timestamp", "TEXT"),
            ("actor", "TEXT"),
            ("src_ip", "TEXT"),
            ("dst_ip", "TEXT"),
            ("protocol", "TEXT"),
            ("bytes", "NUMBER"),
            ("packets", "NUMBER"),
            ("anomaly_type", "TEXT"),
            ("evidence", "TEXT"),
            ("summary", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNActionPlaybookMemoryV1": {
        "purpose": "Human-reviewed actions, operational playbooks, approvals, outcomes, and reusable fixes.",
        "quantum": False,
        "chromatic": False,
        "properties": [
            ("action_id", "TEXT"),
            ("created_at", "TEXT"),
            ("title", "TEXT"),
            ("priority", "TEXT"),
            ("risk", "TEXT"),
            ("description", "TEXT"),
            ("command", "TEXT"),
            ("approval_state", "TEXT"),
            ("outcome", "TEXT"),
            ("source_analysis_id", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNAnalysisEpisodeMemoryV1": {
        "purpose": "LLM analysis episodes with provider/model provenance, protocol blocks, and telemetry context.",
        "quantum": True,
        "chromatic": True,
        "properties": [
            ("analysis_id", "TEXT"),
            ("created_at", "TEXT"),
            ("mode", "TEXT"),
            ("provider", "TEXT"),
            ("model", "TEXT"),
            ("confidence", "NUMBER"),
            ("final_text", "TEXT"),
            ("observations", "TEXT"),
            ("risks", "TEXT"),
            ("actions_json", "TEXT"),
            ("reply_template", "TEXT"),
            ("raw_protocol", "TEXT"),
            ("telemetry_json", "TEXT"),
            ("quantum_json", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNDatasetCorpusMemoryV1": {
        "purpose": "Safely normalized cybersecurity dataset rows and simulated reference records.",
        "quantum": False,
        "chromatic": True,
        "properties": [
            ("record_id", "TEXT"),
            ("dataset_id", "TEXT"),
            ("dataset_name", "TEXT"),
            ("source_url", "TEXT"),
            ("source_kind", "TEXT"),
            ("simulated", "BOOL"),
            ("label", "TEXT"),
            ("category", "TEXT"),
            ("summary", "TEXT"),
            ("features_json", "TEXT"),
            ("safety_json", "TEXT"),
            ("ingested_at", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    },
    "QNChromaticSurfaceMemoryV1": {
        "purpose": "Experimental Chromatic Quantum Surface Embedding records and palette-conditioned semantic states.",
        "quantum": True,
        "chromatic": True,
        "properties": [
            ("surface_id", "TEXT"),
            ("created_at", "TEXT"),
            ("source_kind", "TEXT"),
            ("source_id", "TEXT"),
            ("summary", "TEXT"),
            ("palette_json", "TEXT"),
            ("colorspace_json", "TEXT"),
            ("entropy_json", "TEXT"),
            ("quantum_json", "TEXT"),
            ("experimental_notice", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    }
,
    "QNTemporalHyperfieldMemoryV1": {
        "purpose": "Temporal CQSE worldlines, attractor regimes, regime transitions, boundary energy, and quantum divergence summaries.",
        "quantum": True,
        "chromatic": True,
        "properties": [
            ("worldline_id", "TEXT"),
            ("created_at", "TEXT"),
            ("summary", "TEXT"),
            ("point_count", "INT"),
            ("regime_count", "INT"),
            ("boundary_count", "INT"),
            ("mean_velocity", "NUMBER"),
            ("mean_curvature", "NUMBER"),
            ("mean_quantum_entropy", "NUMBER"),
            ("mean_quantum_js", "NUMBER"),
            ("max_anomaly_score", "NUMBER"),
            ("regimes_json", "TEXT"),
            ("boundaries_json", "TEXT"),
            ("scales_json", "TEXT"),
            ("experimental_notice", "TEXT"),
            ("tags", "TEXT_ARRAY"),
        ],
    }

}


# =============================================================================
# Utilities
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def human_bytes(value: float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TiB"


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


def deterministic_uuid(namespace: str, key: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"qnet://{namespace}/{key}")


def l2_normalize(values: Iterable[float]) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float32).tolist()


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
        fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
        ))
        root.addHandler(fh)


# =============================================================================
# Config — secrets intentionally excluded
# =============================================================================

@dataclass
class AppConfig:
    database_path: str = str(DEFAULT_DB_PATH)
    history_limit: int = 300
    flow_limit: int = 3000
    refresh_seconds: int = 15
    reasoning_effort: str = "high"
    provider_preference: str = "Auto"
    use_openai_embeddings: bool = os.getenv("QNET_OPENAI_EMBEDDINGS", "1") != "0"
    use_fb_embeddings: bool = os.getenv("QNET_FB_EMBEDDINGS", "1") != "0"
    auto_memory_recall: bool = True
    recall_limit: int = 10
    rrf_k: int = 60

    @classmethod
    def load(cls) -> "AppConfig":
        if not CONFIG_PATH.exists():
            cfg = cls()
            cfg.save()
            return cfg
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            allowed = {k: v for k, v in data.items() if k in cls.__annotations__}
            return cls(**allowed)
        except Exception:
            logging.exception("Config load failed; defaults used")
            return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2),
            encoding="utf-8",
        )


# =============================================================================
# Async runtime bridge
# =============================================================================

class AsyncRuntime:
    """Own one asyncio loop in a daemon thread and expose submit(coroutine)."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run,
            name="qnet-async-runtime",
            daemon=True,
        )
        self.thread.start()

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)


# =============================================================================
# aiosqlite state store / migrations / embedding cache
# =============================================================================

class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path

    @asynccontextmanager
    async def connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = await aiosqlite.connect(self.path)
        try:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA foreign_keys=ON")
            await db.execute("PRAGMA busy_timeout=5000")
            yield db
        finally:
            await db.close()

    async def initialize(self):
        async with self.connection() as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS qnet_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS network_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ping REAL NOT NULL,
                    jitter REAL NOT NULL,
                    download_speed REAL NOT NULL,
                    upload_speed REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_network_logs_timestamp
                    ON network_logs(timestamp);

                CREATE TABLE IF NOT EXISTS network_flows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src_ip TEXT NOT NULL,
                    dst_ip TEXT NOT NULL,
                    src_port INTEGER,
                    dst_port INTEGER,
                    protocol TEXT,
                    bytes INTEGER NOT NULL,
                    packets INTEGER,
                    timestamp TEXT NOT NULL,
                    device TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_network_flows_timestamp
                    ON network_flows(timestamp);

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    analysis_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    prompt_hash TEXT NOT NULL,
                    raw_output TEXT,
                    parsed_json TEXT,
                    telemetry_json TEXT,
                    provider_metadata_json TEXT
                );

                CREATE TABLE IF NOT EXISTS action_queue (
                    action_id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    created_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    risk TEXT NOT NULL,
                    description TEXT NOT NULL,
                    command_text TEXT,
                    approval_state TEXT NOT NULL DEFAULT 'pending',
                    outcome TEXT,
                    FOREIGN KEY(analysis_id) REFERENCES analysis_runs(analysis_id)
                );
                CREATE INDEX IF NOT EXISTS idx_action_queue_state
                    ON action_queue(approval_state);

                CREATE TABLE IF NOT EXISTS embedding_cache (
                    namespace TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_blob BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(namespace, text_hash, model)
                );

                CREATE TABLE IF NOT EXISTS memory_sync_state (
                    source_kind TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    weaviate_uuid TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    PRIMARY KEY(source_kind, source_id, collection_name)
                );
                """
            )
            await db.execute(
                "INSERT OR IGNORE INTO qnet_meta(key,value) VALUES('schema_version','5')"
            )
            await db.execute(
                "INSERT OR IGNORE INTO qnet_meta(key,value) VALUES('install_id',?)",
                (str(uuid.uuid4()),),
            )
            await db.commit()

    async def fetch_metrics(self, limit: int) -> list[dict[str, Any]]:
        async with self.connection() as db:
            cur = await db.execute(
                """
                SELECT ping, jitter, download_speed, upload_speed, timestamp
                FROM network_logs
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = await cur.fetchall()
        return [
            {
                "ping": float(r["ping"]),
                "jitter": float(r["jitter"]),
                "download_speed": float(r["download_speed"]),
                "upload_speed": float(r["upload_speed"]),
                "timestamp": str(r["timestamp"]),
            }
            for r in rows
        ]

    async def fetch_flows(self, limit: int) -> list[dict[str, Any]]:
        async with self.connection() as db:
            cur = await db.execute(
                """
                SELECT src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets,timestamp,device
                FROM network_flows
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def save_analysis(
        self,
        analysis_id: str,
        mode: str,
        provider: str,
        model: str,
        prompt: str,
        raw_output: str,
        parsed: dict[str, Any],
        telemetry: dict[str, Any],
        provider_metadata: dict[str, Any],
    ):
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO analysis_runs(
                    analysis_id,created_at,mode,provider,model,prompt_hash,
                    raw_output,parsed_json,telemetry_json,provider_metadata_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    analysis_id,
                    utc_now(),
                    mode,
                    provider,
                    model,
                    stable_hash(prompt),
                    raw_output,
                    safe_json(parsed),
                    safe_json(telemetry),
                    safe_json(provider_metadata),
                ),
            )
            await db.commit()

    async def save_actions(self, analysis_id: str, actions: list[dict[str, Any]]):
        async with self.connection() as db:
            for a in actions:
                action_id = str(a.get("id") or uuid.uuid4())
                await db.execute(
                    """
                    INSERT OR REPLACE INTO action_queue(
                        action_id,analysis_id,created_at,title,priority,risk,
                        description,command_text,approval_state,outcome
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        action_id,
                        analysis_id,
                        utc_now(),
                        str(a.get("title") or "Untitled"),
                        str(a.get("priority") or "medium"),
                        str(a.get("risk") or "medium"),
                        str(a.get("description") or ""),
                        a.get("command"),
                        "pending",
                        None,
                    ),
                )
            await db.commit()

    async def update_action_state(self, action_id: str, state: str):
        if state not in {"pending", "approved", "rejected", "completed", "failed"}:
            raise ValueError("Invalid action state")
        async with self.connection() as db:
            await db.execute(
                "UPDATE action_queue SET approval_state=? WHERE action_id=?",
                (state, action_id),
            )
            await db.commit()

    async def list_actions(self, limit: int = 200) -> list[dict[str, Any]]:
        async with self.connection() as db:
            cur = await db.execute(
                """
                SELECT action_id,analysis_id,created_at,title,priority,risk,
                       description,command_text,approval_state,outcome
                FROM action_queue
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def embedding_get(
        self,
        namespace: str,
        text_hash: str,
        model: str,
    ) -> Optional[list[float]]:
        async with self.connection() as db:
            cur = await db.execute(
                """
                SELECT dimensions, vector_blob
                FROM embedding_cache
                WHERE namespace=? AND text_hash=? AND model=?
                """,
                (namespace, text_hash, model),
            )
            row = await cur.fetchone()
        if not row:
            return None
        raw = zlib.decompress(row["vector_blob"])
        arr = np.frombuffer(raw, dtype=np.float32)
        if len(arr) != int(row["dimensions"]):
            return None
        return arr.tolist()

    async def embedding_put(
        self,
        namespace: str,
        text_hash: str,
        model: str,
        vector: list[float],
    ):
        arr = np.asarray(vector, dtype=np.float32)
        blob = zlib.compress(arr.tobytes(), level=6)
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO embedding_cache(
                    namespace,text_hash,model,dimensions,vector_blob,created_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    namespace,
                    text_hash,
                    model,
                    int(arr.size),
                    blob,
                    utc_now(),
                ),
            )
            await db.commit()

    async def mark_synced(
        self,
        source_kind: str,
        source_id: str,
        collection_name: str,
        object_uuid: str,
    ):
        async with self.connection() as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO memory_sync_state(
                    source_kind,source_id,collection_name,weaviate_uuid,synced_at
                ) VALUES(?,?,?,?,?)
                """,
                (source_kind, source_id, collection_name, object_uuid, utc_now()),
            )
            await db.commit()


# =============================================================================
# PennyLane adaptive quantum feature engine
# =============================================================================

class QuantumFeatureEngine:
    METRICS = ("ping", "jitter", "download_speed", "upload_speed")

    @staticmethod
    def _values(rows: list[dict[str, Any]], metric: str) -> np.ndarray:
        return np.asarray([float(r[metric]) for r in rows], dtype=np.float64)

    @staticmethod
    def robust_z(value: float, series: np.ndarray) -> float:
        if len(series) < 4:
            return 0.0
        med = float(np.median(series))
        mad = float(np.median(np.abs(series - med)))
        if mad <= 1e-12:
            std = float(np.std(series))
            return 0.0 if std <= 1e-12 else (value - med) / std
        return 0.67448975 * (value - med) / mad

    @staticmethod
    def trend(series: np.ndarray) -> dict[str, Any]:
        chronological = series[::-1]
        if len(chronological) < 3:
            return {"direction": "insufficient-data", "slope": 0.0, "normalized_slope": 0.0}
        x = np.arange(len(chronological), dtype=np.float64)
        slope, intercept = np.polyfit(x, chronological, 1)
        span = float(np.ptp(chronological))
        norm = float(slope / span) if span > 1e-12 else 0.0
        direction = "flat"
        if norm > 0.01:
            direction = "rising"
        elif norm < -0.01:
            direction = "falling"
        return {
            "direction": direction,
            "slope": float(slope),
            "normalized_slope": norm,
            "intercept": float(intercept),
        }

    @staticmethod
    def forecast(series: np.ndarray, steps: int = 8) -> list[float]:
        chronological = series[::-1]
        if len(chronological) < 3:
            return [float(chronological[-1])] * steps
        window = chronological[-min(48, len(chronological)):]
        x = np.arange(len(window), dtype=np.float64)
        slope, intercept = np.polyfit(x, window, 1)
        future = np.arange(len(window), len(window) + steps, dtype=np.float64)
        return [float(max(0.0, slope * t + intercept)) for t in future]

    def circuit_features(
        self,
        current: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Robust p05/p95 scaling gives the circuit useful dynamic range.
        scales: dict[str, dict[str, float]] = {}
        normalized: dict[str, float] = {}
        for metric in self.METRICS:
            values = np.maximum(self._values(rows, metric), 0.0)
            lo = float(np.percentile(values, 5))
            hi = float(np.percentile(values, 95))
            if hi - lo < 1e-9:
                hi = lo + max(abs(lo) * 0.25, 1.0)
            scales[metric] = {"p05": lo, "p95": hi}
            normalized[metric] = float(
                np.clip((float(current[metric]) - lo) / (hi - lo), 0.0, 1.0)
            )

        # Data re-uploading: RY encodes normalized magnitude, RZ encodes
        # deviations relative to the historical median.
        rz_phase: dict[str, float] = {}
        for metric in self.METRICS:
            vals = self._values(rows, metric)
            z = float(np.clip(self.robust_z(float(current[metric]), vals), -6, 6))
            rz_phase[metric] = math.pi * (z / 6.0)

        angles = {
            "ping": math.pi * normalized["ping"],
            "jitter": math.pi * normalized["jitter"],
            "download_speed": math.pi * normalized["download_speed"],
            "upload_speed": math.pi * normalized["upload_speed"],
        }

        dev = qml.device("default.qubit", wires=4)
        metric_order = list(self.METRICS)

        @qml.qnode(dev)
        def circuit():
            for layer in range(3):
                multiplier = 1.0 + 0.17 * layer
                for wire, metric in enumerate(metric_order):
                    qml.RY(angles[metric] * multiplier, wires=wire)
                    qml.RZ(rz_phase[metric] / (layer + 1), wires=wire)

                # Alternating ring entanglement avoids a purely linear chain.
                if layer % 2 == 0:
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 3])
                    qml.CNOT(wires=[3, 0])
                else:
                    qml.CZ(wires=[0, 2])
                    qml.CZ(wires=[1, 3])

            return (
                qml.probs(wires=[0, 1, 2, 3]),
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2)),
                qml.expval(qml.PauliZ(3)),
            )

        output = circuit()
        probs = np.asarray(output[0], dtype=np.float64)
        eps = 1e-12
        shannon_entropy_bits = float(-np.sum(probs * np.log2(probs + eps)))

        return {
            "scales": scales,
            "normalized": normalized,
            "ry_angles": angles,
            "rz_phases": rz_phase,
            "probabilities": [float(x) for x in probs],
            "pauli_z_expectations": [float(x) for x in output[1:]],
            "shannon_entropy_bits": shannon_entropy_bits,
            "saturation": {m: normalized[m] <= 0.02 or normalized[m] >= 0.98 for m in self.METRICS},
        }

    def analyze(
        self,
        rows: list[dict[str, Any]],
        flows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not rows:
            raise RuntimeError("network_logs contains no samples.")

        current = rows[0]
        metrics: dict[str, Any] = {}
        anomalies: list[dict[str, Any]] = []

        for metric in self.METRICS:
            values = self._values(rows, metric)
            rz = self.robust_z(float(current[metric]), values)
            metrics[metric] = {
                "current": float(current[metric]),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p05": float(np.percentile(values, 5)),
                "p95": float(np.percentile(values, 95)),
                "robust_z": float(rz),
                "trend": self.trend(values),
                "forecast_next_8_samples": self.forecast(values, 8),
            }
            if abs(rz) >= 3.5:
                anomalies.append({
                    "metric": metric,
                    "value": float(current[metric]),
                    "robust_z": float(rz),
                    "severity": "high" if abs(rz) >= 5 else "moderate",
                })

        # Hour-of-day temporal surface.
        hourly: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            try:
                dt = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            except Exception:
                continue
            hourly.setdefault(dt.hour, []).append(row)

        hourly_patterns = []
        for hour, bucket in sorted(hourly.items()):
            if len(bucket) < 2:
                continue
            hourly_patterns.append({
                "hour": hour,
                "samples": len(bucket),
                "median_ping": float(np.median([x["ping"] for x in bucket])),
                "median_jitter": float(np.median([x["jitter"] for x in bucket])),
                "median_download": float(np.median([x["download_speed"] for x in bucket])),
                "median_upload": float(np.median([x["upload_speed"] for x in bucket])),
            })

        flow_analysis = self._analyze_flows(flows)

        return {
            "generated_at": utc_now(),
            "sample_count": len(rows),
            "current": current,
            "metrics": metrics,
            "current_anomalies": anomalies,
            "hourly_patterns": hourly_patterns,
            "quantum": self.circuit_features(current, rows),
            "flow_analysis": flow_analysis,
        }

    @staticmethod
    def _analyze_flows(flows: list[dict[str, Any]]) -> dict[str, Any]:
        if not flows:
            return {
                "available": False,
                "message": (
                    "No flow rows are available. Ping/jitter/throughput cannot identify "
                    "devices or establish malware."
                ),
            }

        byte_values = np.asarray(
            [max(0.0, float(f.get("bytes") or 0.0)) for f in flows],
            dtype=np.float64,
        )
        med = float(np.median(byte_values))
        mad = float(np.median(np.abs(byte_values - med)))

        totals: dict[str, float] = {}
        destinations: dict[str, set[str]] = {}
        anomalies: list[dict[str, Any]] = []

        for flow, bytes_value in zip(flows, byte_values):
            actor = str(flow.get("device") or flow.get("src_ip") or "unknown")
            src = str(flow.get("src_ip") or "unknown")
            dst = str(flow.get("dst_ip") or "unknown")
            totals[actor] = totals.get(actor, 0.0) + float(bytes_value)
            destinations.setdefault(src, set()).add(dst)

            if mad > 1e-12:
                rz = 0.67448975 * (float(bytes_value) - med) / mad
                if rz >= 5.0:
                    anomalies.append({
                        "type": "large-flow-outlier",
                        "src_ip": src,
                        "dst_ip": dst,
                        "bytes": float(bytes_value),
                        "robust_z": float(rz),
                    })

        for src, dsts in destinations.items():
            if len(dsts) >= 25:
                anomalies.append({
                    "type": "high-destination-fanout",
                    "src_ip": src,
                    "unique_destinations": len(dsts),
                })

        top_talkers = sorted(
            ({"actor": actor, "bytes": total} for actor, total in totals.items()),
            key=lambda x: x["bytes"],
            reverse=True,
        )[:20]

        return {
            "available": True,
            "flow_count": len(flows),
            "total_bytes": float(np.sum(byte_values)),
            "median_flow_bytes": med,
            "top_talkers": top_talkers,
            "statistical_anomalies": anomalies[:50],
            "security_note": (
                "These are statistical anomalies, not malware signatures. Confirm with "
                "packet, DNS, IDS/EDR, endpoint, or threat-intelligence evidence."
            ),
        }


# =============================================================================
# Temporal Chromatic Quantum Hyperfield (TCQH)
# =============================================================================

@dataclass
class HyperfieldPoint:
    index: int
    timestamp: str
    t_norm: float
    lab_l: float
    lab_a: float
    lab_b: float
    lch_c: float
    lch_h: float
    ping: float
    jitter: float
    download: float
    upload: float
    semantic_entropy: float
    quantum_entropy: float
    telemetry_quantum_entropy: float
    quantum_js_divergence: float
    velocity: float
    acceleration: float
    curvature: float
    change_score: float
    regime_id: int = -1
    attractor_distance: float = 0.0
    anomaly_score: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.lab_l,
                self.lab_a,
                self.lab_b,
                self.lch_c,
                math.sin(2 * math.pi * self.lch_h),
                math.cos(2 * math.pi * self.lch_h),
                self.semantic_entropy,
                self.quantum_entropy,
                self.telemetry_quantum_entropy,
                self.quantum_js_divergence,
                self.velocity,
                self.acceleration,
                self.curvature,
                self.change_score,
            ],
            dtype=np.float64,
        )


class TemporalChromaticHyperfieldEngine:
    """
    Experimental temporal field built on top of CQSE.

    A telemetry sequence becomes a worldline in a derived field:
      x,y,z  -> OKLab L/a/b centroid
      t      -> normalized event time/sample order
      energy -> CQSE / telemetry quantum entropy
      motion -> velocity, acceleration, curvature
      regime -> unsupervised deterministic k-means state
      boundary -> robust multivariate change score

    This is a visualization/retrieval feature transform, not a physical claim that
    network telemetry literally evolves in quantum spacetime.
    """

    def __init__(self, cqse: ChromaticQuantumSurfaceEncoder):
        self.cqse = cqse

    @staticmethod
    def _entropy(probs: Iterable[float]) -> float:
        arr = np.asarray(list(probs), dtype=np.float64)
        if arr.size == 0:
            return 0.0
        arr = np.clip(arr, 0.0, None)
        total = float(arr.sum())
        if total <= 0:
            return 0.0
        arr /= total
        return float(-np.sum(arr * np.log2(arr + 1e-12)))

    @staticmethod
    def _js_divergence(p: Iterable[float], q: Iterable[float]) -> float:
        p = np.asarray(list(p), dtype=np.float64)
        q = np.asarray(list(q), dtype=np.float64)
        if p.size == 0 or q.size == 0 or p.size != q.size:
            return 0.0
        p = np.clip(p, 0.0, None)
        q = np.clip(q, 0.0, None)
        p /= max(float(p.sum()), 1e-12)
        q /= max(float(q.sum()), 1e-12)
        m = 0.5 * (p + q)

        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log2((a[mask] + 1e-12) / (b[mask] + 1e-12))))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    @staticmethod
    def _rolling_scale(values: np.ndarray) -> tuple[float, float]:
        lo = float(np.percentile(values, 5))
        hi = float(np.percentile(values, 95))
        if hi - lo < 1e-12:
            hi = lo + max(abs(lo) * 0.25, 1.0)
        return lo, hi

    @staticmethod
    def _telemetry_q16(
        row: dict[str, Any],
        scales: dict[str, tuple[float, float]],
    ) -> list[float]:
        metrics = ("ping", "jitter", "download_speed", "upload_speed")
        norm = {}
        for metric in metrics:
            lo, hi = scales[metric]
            norm[metric] = float(
                np.clip((float(row[metric]) - lo) / (hi - lo), 0.0, 1.0)
            )

        dev = qml.device("default.qubit", wires=4)

        @qml.qnode(dev)
        def circuit():
            vals = [
                norm["ping"],
                norm["jitter"],
                norm["download_speed"],
                norm["upload_speed"],
            ]
            for wire, value in enumerate(vals):
                qml.RY(math.pi * value, wires=wire)
                qml.RZ(math.pi * (value - 0.5), wires=wire)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 3])
            qml.CNOT(wires=[3, 0])
            return qml.probs(wires=[0, 1, 2, 3])

        return np.asarray(circuit(), dtype=np.float32).tolist()

    @staticmethod
    def _row_text(row: dict[str, Any], index: int) -> str:
        return (
            f"telemetry index={index} timestamp={row.get('timestamp')} "
            f"ping={float(row.get('ping', 0.0)):.6f}ms "
            f"jitter={float(row.get('jitter', 0.0)):.6f}ms "
            f"download={float(row.get('download_speed', 0.0)):.6f}Mbps "
            f"upload={float(row.get('upload_speed', 0.0)):.6f}Mbps"
        )

    @staticmethod
    def _weighted_lab_centroid(
        labs: list[tuple[float, float, float]],
        rgbs: list[tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        if not labs:
            return 0.0, 0.0, 0.0
        weights = []
        for rgb in rgbs:
            weights.append(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2] + 1e-6)
        if len(weights) != len(labs):
            weights = [1.0] * len(labs)
        w = np.asarray(weights, dtype=np.float64)
        w /= w.sum()
        arr = np.asarray(labs, dtype=np.float64)
        centroid = np.sum(arr * w[:, None], axis=0)
        return tuple(float(x) for x in centroid)

    @staticmethod
    def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        med = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - med), axis=0)
        mad = np.where(mad < 1e-9, np.std(matrix, axis=0) + 1e-9, mad)
        return (matrix - med) / (1.4826 * mad + 1e-9)

    @staticmethod
    def _deterministic_kmeans(
        x: np.ndarray,
        k: int,
        iterations: int = 80,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = len(x)
        if n == 0:
            return np.empty((0,), dtype=int), np.empty((0, x.shape[1])), 0.0
        k = max(1, min(k, n))

        # Farthest-first deterministic initialization.
        norms = np.linalg.norm(x, axis=1)
        first = int(np.argmax(norms))
        centers = [x[first].copy()]
        while len(centers) < k:
            distances = np.min(
                np.stack([np.linalg.norm(x - c, axis=1) for c in centers], axis=1),
                axis=1,
            )
            centers.append(x[int(np.argmax(distances))].copy())
        centers = np.asarray(centers, dtype=np.float64)

        labels = np.zeros(n, dtype=int)
        for _ in range(iterations):
            distances = np.stack(
                [np.linalg.norm(x - c, axis=1) for c in centers],
                axis=1,
            )
            new_labels = np.argmin(distances, axis=1)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels
            new_centers = []
            for cluster in range(k):
                members = x[labels == cluster]
                if len(members):
                    new_centers.append(members.mean(axis=0))
                else:
                    new_centers.append(centers[cluster])
            centers = np.asarray(new_centers, dtype=np.float64)

        within = 0.0
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                within += float(np.sum((members - centers[cluster]) ** 2))

        # BIC-ish complexity penalty; used only to choose a stable visual regime count.
        penalty = k * x.shape[1] * math.log(max(n, 2))
        score = within + 0.035 * penalty
        return labels, centers, score

    def _choose_regimes(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(matrix) < 6:
            labels, centers, _ = self._deterministic_kmeans(matrix, min(2, len(matrix)))
            return labels, centers

        best = None
        max_k = min(6, max(2, len(matrix) // 8))
        for k in range(2, max_k + 1):
            labels, centers, score = self._deterministic_kmeans(matrix, k)
            if best is None or score < best[0]:
                best = (score, labels, centers)
        assert best is not None
        return best[1], best[2]

    @staticmethod
    def _local_change_score(matrix: np.ndarray) -> np.ndarray:
        if len(matrix) < 2:
            return np.zeros(len(matrix))
        normalized = TemporalChromaticHyperfieldEngine._normalize_matrix(matrix)
        delta = np.linalg.norm(np.diff(normalized, axis=0), axis=1)
        scores = np.concatenate([[0.0], delta])
        med = float(np.median(scores))
        mad = float(np.median(np.abs(scores - med)))
        scale = 1.4826 * mad if mad > 1e-9 else float(np.std(scores) + 1e-9)
        return np.maximum(0.0, (scores - med) / scale)

    def build(
        self,
        rows_newest_first: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if len(rows_newest_first) < 2:
            raise RuntimeError("Temporal hyperfield needs at least two telemetry samples.")

        rows = list(reversed(rows_newest_first))
        scales = {
            metric: self._rolling_scale(
                np.asarray([float(r[metric]) for r in rows], dtype=np.float64)
            )
            for metric in ("ping", "jitter", "download_speed", "upload_speed")
        }

        timestamps = []
        for row in rows:
            try:
                timestamps.append(
                    datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00")).timestamp()
                )
            except Exception:
                timestamps.append(float(len(timestamps)))

        t_arr = np.asarray(timestamps, dtype=np.float64)
        if np.ptp(t_arr) <= 1e-9:
            t_norm = np.linspace(0.0, 1.0, len(rows))
        else:
            t_norm = (t_arr - t_arr.min()) / np.ptp(t_arr)

        raw: list[dict[str, Any]] = []
        prev_q32: Optional[list[float]] = None

        for i, row in enumerate(rows):
            q16 = self._telemetry_q16(row, scales)
            text = self._row_text(row, i)
            cqse = self.cqse.encode(text, {}, q16, None)

            labs = [tuple(float(v) for v in x) for x in cqse["oklab_anchors"]]
            lchs = [tuple(float(v) for v in x) for x in cqse["oklch_anchors"]]
            rgbs = [tuple(float(v) for v in x) for x in cqse["rgb_anchors"]]
            lab = self._weighted_lab_centroid(labs, rgbs)

            if lchs:
                c_mean = float(np.mean([x[1] for x in lchs]))
                # Circular mean hue.
                angles = np.asarray([2 * math.pi * x[2] for x in lchs], dtype=np.float64)
                h_mean = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
                h_mean = (h_mean / (2 * math.pi)) % 1.0
            else:
                c_mean = 0.0
                h_mean = 0.0

            q32 = cqse["chromatic_quantum_32"]
            q_js = self._js_divergence(prev_q32, q32) if prev_q32 is not None else 0.0
            prev_q32 = q32

            raw.append({
                "index": i,
                "timestamp": str(row["timestamp"]),
                "t_norm": float(t_norm[i]),
                "lab": lab,
                "lch_c": c_mean,
                "lch_h": h_mean,
                "semantic_entropy": float(cqse["distribution"]["shannon_abs_bits"]),
                "q32_entropy": self._entropy(q32),
                "q16_entropy": self._entropy(q16),
                "q_js": q_js,
                "q16": q16,
                "q32": q32,
                "cqse96": cqse["chromatic_surface_96"],
                "row": row,
            })

        # Geometry in LAB + time-scaled space.
        coords = np.asarray(
            [
                [
                    item["lab"][0],
                    item["lab"][1],
                    item["lab"][2],
                    0.35 * item["t_norm"],
                ]
                for item in raw
            ],
            dtype=np.float64,
        )
        coords_norm = self._normalize_matrix(coords)

        velocity = np.zeros(len(raw), dtype=np.float64)
        acceleration = np.zeros(len(raw), dtype=np.float64)
        curvature = np.zeros(len(raw), dtype=np.float64)

        if len(raw) > 1:
            deltas = np.diff(coords_norm, axis=0)
            velocity[1:] = np.linalg.norm(deltas, axis=1)

        if len(raw) > 2:
            acceleration[2:] = np.abs(np.diff(velocity[1:]))
            for i in range(1, len(raw) - 1):
                a = coords_norm[i] - coords_norm[i - 1]
                b = coords_norm[i + 1] - coords_norm[i]
                na = np.linalg.norm(a)
                nb = np.linalg.norm(b)
                if na > 1e-9 and nb > 1e-9:
                    cosine = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
                    curvature[i] = math.acos(cosine)

        pre_matrix = np.asarray(
            [
                [
                    item["lab"][0],
                    item["lab"][1],
                    item["lab"][2],
                    item["semantic_entropy"],
                    item["q32_entropy"],
                    item["q16_entropy"],
                    item["q_js"],
                    velocity[i],
                    acceleration[i],
                    curvature[i],
                    float(item["row"]["ping"]),
                    float(item["row"]["jitter"]),
                    float(item["row"]["download_speed"]),
                    float(item["row"]["upload_speed"]),
                ]
                for i, item in enumerate(raw)
            ],
            dtype=np.float64,
        )
        change_score = self._local_change_score(pre_matrix)

        regime_matrix = self._normalize_matrix(
            np.column_stack(
                [
                    coords[:, :3],
                    np.asarray([r["q32_entropy"] for r in raw]),
                    np.asarray([r["q_js"] for r in raw]),
                    velocity,
                    change_score,
                ]
            )
        )
        labels, centers = self._choose_regimes(regime_matrix)

        # Robust anomaly score on the temporal feature matrix.
        norm_temporal = self._normalize_matrix(pre_matrix)
        robust_distance = np.linalg.norm(norm_temporal, axis=1)
        rd_med = float(np.median(robust_distance))
        rd_mad = float(np.median(np.abs(robust_distance - rd_med)))
        rd_scale = 1.4826 * rd_mad if rd_mad > 1e-9 else float(np.std(robust_distance) + 1e-9)
        anomaly_score = np.maximum(0.0, (robust_distance - rd_med) / rd_scale)

        points: list[HyperfieldPoint] = []
        for i, item in enumerate(raw):
            center = centers[labels[i]] if len(centers) else np.zeros(regime_matrix.shape[1])
            attractor_distance = float(np.linalg.norm(regime_matrix[i] - center))
            points.append(
                HyperfieldPoint(
                    index=i,
                    timestamp=item["timestamp"],
                    t_norm=item["t_norm"],
                    lab_l=item["lab"][0],
                    lab_a=item["lab"][1],
                    lab_b=item["lab"][2],
                    lch_c=item["lch_c"],
                    lch_h=item["lch_h"],
                    ping=float(item["row"]["ping"]),
                    jitter=float(item["row"]["jitter"]),
                    download=float(item["row"]["download_speed"]),
                    upload=float(item["row"]["upload_speed"]),
                    semantic_entropy=item["semantic_entropy"],
                    quantum_entropy=item["q32_entropy"],
                    telemetry_quantum_entropy=item["q16_entropy"],
                    quantum_js_divergence=item["q_js"],
                    velocity=float(velocity[i]),
                    acceleration=float(acceleration[i]),
                    curvature=float(curvature[i]),
                    change_score=float(change_score[i]),
                    regime_id=int(labels[i]),
                    attractor_distance=attractor_distance,
                    anomaly_score=float(anomaly_score[i]),
                )
            )

        boundaries = []
        for i, point in enumerate(points):
            regime_change = i > 0 and points[i - 1].regime_id != point.regime_id
            if point.change_score >= 3.0 or regime_change or point.anomaly_score >= 3.5:
                boundaries.append({
                    "index": i,
                    "timestamp": point.timestamp,
                    "regime_from": points[i - 1].regime_id if i > 0 else None,
                    "regime_to": point.regime_id,
                    "change_score": point.change_score,
                    "anomaly_score": point.anomaly_score,
                    "quantum_js_divergence": point.quantum_js_divergence,
                    "reason": [
                        reason
                        for condition, reason in [
                            (point.change_score >= 3.0, "multivariate-change"),
                            (regime_change, "regime-transition"),
                            (point.anomaly_score >= 3.5, "robust-temporal-outlier"),
                        ]
                        if condition
                    ],
                })

        # Regime / attractor summaries.
        regimes = []
        for regime_id in sorted(set(int(x) for x in labels.tolist())):
            member_idx = np.where(labels == regime_id)[0]
            subset = [points[int(i)] for i in member_idx]
            regimes.append({
                "regime_id": regime_id,
                "samples": len(subset),
                "first_index": int(member_idx.min()),
                "last_index": int(member_idx.max()),
                "mean_lab": [
                    float(np.mean([p.lab_l for p in subset])),
                    float(np.mean([p.lab_a for p in subset])),
                    float(np.mean([p.lab_b for p in subset])),
                ],
                "mean_ping": float(np.mean([p.ping for p in subset])),
                "mean_jitter": float(np.mean([p.jitter for p in subset])),
                "mean_download": float(np.mean([p.download for p in subset])),
                "mean_upload": float(np.mean([p.upload for p in subset])),
                "mean_quantum_entropy": float(np.mean([p.quantum_entropy for p in subset])),
                "mean_change_score": float(np.mean([p.change_score for p in subset])),
                "max_anomaly_score": float(np.max([p.anomaly_score for p in subset])),
            })

        return {
            "generated_at": utc_now(),
            "point_count": len(points),
            "points": [asdict(p) for p in points],
            "boundaries": boundaries,
            "regimes": regimes,
            "scales": {
                k: {"p05": v[0], "p95": v[1]}
                for k, v in scales.items()
            },
            "experimental_notice": (
                "TCQH is an experimental temporal retrieval/visualization transform. "
                "Regimes and attractors are unsupervised mathematical structures, not "
                "proof of root cause, malware, or physical quantum dynamics."
            ),
        }


# =============================================================================
# Structured [tag] protocol
# =============================================================================

TAG_RE = re.compile(
    r"\[(?P<tag>[a-zA-Z][a-zA-Z0-9_-]*)\](?P<body>.*?)\[/\1\]",
    re.DOTALL,
)
KNOWN_TAGS = {
    "status",
    "observation",
    "evidence",
    "risk",
    "action",
    "replytemplate",
    "confidence",
    "recommendation",
    "final",
}

PROTOCOL_INSTRUCTIONS = r"""
Return only operator-facing conclusions and evidence summaries; never expose private
chain-of-thought.

Use these blocks:

[status]
One concise state summary.
[/status]

[observation]
Measured or deterministically derived facts only.
[/observation]

[evidence]
Specific metrics, trends, memory matches, or flow evidence supporting the assessment.
[/evidence]

[risk]
Uncertainty, limitations, and claims that are NOT established by the available data.
[/risk]

[action]
{"id":"A1","title":"Action title","priority":"high|medium|low","risk":"low|medium|high","requires_approval":true,"description":"Safe defensive action","command":null}
[/action]

[action] may repeat. The command field MUST be null unless the user explicitly requested
a command. Do not produce credential-access, destructive, persistence, evasion, or
unauthorized access commands.

[replytemplate]
A reusable operator/stakeholder status message.
[/replytemplate]

[confidence]
0.0
[/confidence]

[final]
Concise synthesis.
[/final]
""".strip()


@dataclass
class ParsedProtocol:
    raw: str
    blocks: dict[str, list[str]] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)
    reply_templates: list[str] = field(default_factory=list)
    confidence: Optional[float] = None
    final_text: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self):
        return {
            "blocks": self.blocks,
            "actions": self.actions,
            "reply_templates": self.reply_templates,
            "confidence": self.confidence,
            "final_text": self.final_text,
            "warnings": self.warnings,
        }


def parse_protocol(text: str) -> ParsedProtocol:
    out = ParsedProtocol(raw=text)
    matches = list(TAG_RE.finditer(text))

    if not matches:
        out.final_text = text.strip()
        out.warnings.append("Model output did not contain structured protocol tags.")
        return out

    for match in matches:
        tag = match.group("tag").lower()
        body = match.group("body").strip()
        if tag not in KNOWN_TAGS:
            out.warnings.append(f"Unknown tag ignored: [{tag}]")
            continue

        out.blocks.setdefault(tag, []).append(body)

        if tag == "action":
            try:
                obj = json.loads(body)
                if not isinstance(obj, dict):
                    raise ValueError("action payload is not an object")
                action = {
                    "id": str(obj.get("id") or f"A{len(out.actions)+1}"),
                    "title": str(obj.get("title") or "Untitled action"),
                    "priority": str(obj.get("priority") or "medium").lower(),
                    "risk": str(obj.get("risk") or "medium").lower(),
                    # Fail closed regardless of model output.
                    "requires_approval": True,
                    "description": str(obj.get("description") or ""),
                    "command": obj.get("command") if isinstance(obj.get("command"), str) else None,
                }
                if action["priority"] not in {"high", "medium", "low"}:
                    action["priority"] = "medium"
                if action["risk"] not in {"high", "medium", "low"}:
                    action["risk"] = "medium"
                out.actions.append(action)
            except Exception as exc:
                out.warnings.append(f"Invalid [action] JSON: {exc}")

        elif tag == "replytemplate":
            out.reply_templates.append(body)

        elif tag == "confidence":
            try:
                out.confidence = float(np.clip(float(body), 0.0, 1.0))
            except Exception:
                out.warnings.append("Invalid [confidence] value.")

        elif tag == "final":
            out.final_text = body

    if not out.final_text:
        out.final_text = text.strip()

    return out


def protocol_markdown(parsed: ParsedProtocol) -> str:
    sections = []
    titles = {
        "status": "Status",
        "observation": "Observations",
        "evidence": "Evidence",
        "risk": "Risk / uncertainty",
        "recommendation": "Recommendations",
        "final": "Final",
    }
    for tag, title in titles.items():
        values = parsed.blocks.get(tag, [])
        if values:
            sections.append(f"## {title}\n\n" + "\n\n".join(values))

    if parsed.actions:
        lines = ["## Proposed actions"]
        for a in parsed.actions:
            lines.append(
                f"- **{a['id']} — {a['title']}** "
                f"(priority={a['priority']}, risk={a['risk']}, approval required)\n"
                f"  {a['description']}"
            )
        sections.append("\n".join(lines))

    if parsed.reply_templates:
        sections.append("## Reply template\n\n" + "\n\n".join(parsed.reply_templates))

    if parsed.confidence is not None:
        sections.append(f"## Confidence\n\n{parsed.confidence:.2f}")

    if parsed.warnings:
        sections.append("## Parser warnings\n\n" + "\n".join(f"- {w}" for w in parsed.warnings))

    return "\n\n".join(sections).strip()


# =============================================================================
# Embedding providers
# =============================================================================

class OpenAIEmbeddingProvider:
    def __init__(self, store: SQLiteStore, api_key_getter: Callable[[], str]):
        self.store = store
        self.api_key_getter = api_key_getter

    def available(self) -> bool:
        return bool(self.api_key_getter().strip())

    async def embed(self, text: str) -> list[float]:
        key = self.api_key_getter().strip()
        if not key:
            raise RuntimeError("OpenAI API key is not configured.")

        text_hash = stable_hash(text)
        cached = await self.store.embedding_get("openai", text_hash, EMBED_MODEL)
        if cached is not None:
            return cached

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, verify=True) as client:
            response = await client.post(
                EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": EMBED_MODEL,
                    "input": text,
                    "dimensions": EMBED_DIM,
                    "encoding_format": "float",
                },
            )
        response.raise_for_status()
        data = response.json()
        vector = data["data"][0]["embedding"]
        vector = l2_normalize(vector)
        await self.store.embedding_put("openai", text_hash, EMBED_MODEL, vector)
        return vector


class FacebookContrieverProvider:
    """Local Meta/Facebook dense retrieval embedding model with mean pooling."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._load_lock = threading.Lock()

    def available(self) -> bool:
        try:
            import torch  # noqa
            import transformers  # noqa
            return True
        except Exception:
            return False

    def _load_sync(self):
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(
                FB_EMBED_MODEL,
                revision=FB_EMBED_REVISION,
                trust_remote_code=False,
                local_files_only=os.getenv("QNET_HF_OFFLINE", "0") == "1",
            )
            self._model = AutoModel.from_pretrained(
                FB_EMBED_MODEL,
                revision=FB_EMBED_REVISION,
                trust_remote_code=False,
                use_safetensors=True,
                local_files_only=os.getenv("QNET_HF_OFFLINE", "0") == "1",
            )
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()

    def _embed_sync(self, text: str) -> list[float]:
        self._load_sync()
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model

        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).bool()
            token_embeddings = token_embeddings.masked_fill(~mask, 0.0)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = token_embeddings.sum(dim=1) / denom

        vector = pooled[0].detach().cpu().float().numpy()
        return l2_normalize(vector.tolist())

    async def embed(self, text: str) -> list[float]:
        text_hash = stable_hash(text)
        cached = await self.store.embedding_get("facebook", text_hash, FB_EMBED_MODEL)
        if cached is not None:
            return cached
        vector = await asyncio.to_thread(self._embed_sync, text)
        if len(vector) != FB_EMBED_DIM:
            raise RuntimeError(f"Unexpected Contriever dimension: {len(vector)}")
        await self.store.embedding_put("facebook", text_hash, FB_EMBED_MODEL, vector)
        return vector


class EmbeddingEnsemble:
    def __init__(
        self,
        store: SQLiteStore,
        api_key_getter: Callable[[], str],
        config_getter: Callable[[], AppConfig],
    ):
        self.config_getter = config_getter
        self.openai = OpenAIEmbeddingProvider(store, api_key_getter)
        self.facebook = FacebookContrieverProvider(store)

    async def embed_all(self, text: str) -> tuple[dict[str, list[float]], list[str]]:
        cfg = self.config_getter()
        vectors: dict[str, list[float]] = {}
        errors: list[str] = []

        async def run_openai():
            if cfg.use_openai_embeddings and self.openai.available():
                try:
                    vectors[VECTOR_OPENAI] = await self.openai.embed(text)
                except Exception as exc:
                    errors.append(f"OpenAI embeddings: {exc}")

        async def run_fb():
            if cfg.use_fb_embeddings and self.facebook.available():
                try:
                    vectors[VECTOR_FB] = await self.facebook.embed(text)
                except Exception as exc:
                    errors.append(f"Facebook Contriever: {exc}")
            elif cfg.use_fb_embeddings:
                errors.append("Facebook Contriever dependencies unavailable (torch/transformers).")

        await asyncio.gather(run_openai(), run_fb())
        return vectors, errors


# =============================================================================
# Async Weaviate memory surface
# =============================================================================

# =============================================================================
# Safe cybersecurity dataset manifests and simulation
# =============================================================================

@dataclass(frozen=True)
class CyberDatasetManifest:
    dataset_id: str
    name: str
    source_url: str
    category: str
    description: str
    rank: int
    kaggle_slug: Optional[str] = None
    safe_default: str = "simulate"
    feature_names: tuple[str, ...] = ()


CYBER_DATASETS: tuple[CyberDatasetManifest, ...] = (
    CyberDatasetManifest(
        dataset_id="lanl_unified_host_network",
        name="LANL Unified Host and Network Dataset",
        source_url="https://www.kaggle.com/discussions/general/335189",
        category="network+host",
        description=(
            "Enterprise host and network events collected at Los Alamos National Laboratory. "
            "The default loader simulates normalized connection/event summaries rather than "
            "fetching the very large original corpus."
        ),
        rank=1,
        feature_names=(
            "duration_ms", "src_bytes", "dst_bytes", "src_packets", "dst_packets",
            "auth_events", "process_events", "unique_destinations", "hour",
        ),
    ),
    CyberDatasetManifest(
        dataset_id="lanl_multisource_events",
        name="LANL Comprehensive Multi-Source Cyber-Security Events",
        source_url="https://www.kaggle.com/discussions/general/335189",
        category="multi-source-enterprise",
        description=(
            "De-identified multi-source enterprise security events. Simulation preserves "
            "cross-source correlation concepts without reproducing identities or raw logs."
        ),
        rank=2,
        feature_names=(
            "auth_rate", "dns_rate", "flow_rate", "process_rate", "redteam_signal",
            "unique_hosts", "service_entropy", "hour", "day_index",
        ),
    ),
    CyberDatasetManifest(
        dataset_id="lanl_user_computer_auth",
        name="LANL User-Computer Authentication Associations in Time",
        source_url="https://www.kaggle.com/discussions/general/335189",
        category="authentication",
        description=(
            "Large-scale anonymized user-to-computer authentication associations. "
            "Simulation produces aggregate authentication graph statistics only."
        ),
        rank=3,
        feature_names=(
            "auth_count", "unique_sources", "unique_targets", "failure_ratio",
            "new_edge_ratio", "off_hours_ratio", "degree_z", "burst_score",
        ),
    ),
    CyberDatasetManifest(
        dataset_id="cic_family",
        name="Canadian Institute for Cybersecurity Dataset Family",
        source_url="https://www.kaggle.com/discussions/general/335189",
        category="network-intrusion",
        description=(
            "Family of widely used intrusion-detection datasets. The simulator emits "
            "flow-like features inspired by common CIC-style traffic fields."
        ),
        rank=4,
        feature_names=(
            "flow_duration", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
            "flow_bytes_s", "flow_packets_s", "syn_count", "rst_count", "idle_mean",
        ),
    ),
    CyberDatasetManifest(
        dataset_id="kdd99",
        name="KDD Cup 1999 Intrusion Detection",
        source_url="https://www.kaggle.com/discussions/general/335189",
        category="network-intrusion",
        description=(
            "Classic intrusion-detection benchmark. The simulator uses normalized KDD-like "
            "connection features and explicit synthetic labels."
        ),
        rank=5,
        kaggle_slug="galaxyh/kdd-cup-1999-data",
        feature_names=(
            "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
            "count", "srv_count", "serror_rate", "same_srv_rate", "dst_host_count",
        ),
    ),
    CyberDatasetManifest(
        dataset_id="agung_network_malware_connection",
        name="Malware Detection in Network Traffic Data",
        source_url=(
            "https://www.kaggle.com/datasets/agungpambudi/"
            "network-malware-detection-connection-analysis/versions/1"
        ),
        category="network-malware",
        description=(
            "User-linked Kaggle network-malware/connection-analysis dataset. It is never "
            "executed; the safe importer treats it strictly as tabular data."
        ),
        rank=6,
        kaggle_slug="agungpambudi/network-malware-detection-connection-analysis",
        feature_names=(
            "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
            "service_code", "conn_state_code", "history_score", "label_code",
        ),
    ),
)


class SafeDatasetSimulator:
    """
    Generates bounded, schema-faithful *synthetic* records for cybersecurity research.

    It does not recreate original samples and does not claim statistical equivalence.
    Synthetic records are labeled simulated=True all the way into Weaviate.
    """

    LABELS = ("benign", "scan", "bruteforce", "dos", "botnet-like", "anomalous")

    def __init__(self, seed: int = 0xC01A5E):
        self.seed = seed

    @staticmethod
    def _bounded_lognormal(rng: random.Random, median: float, spread: float, cap: float) -> float:
        # random.lognormvariate expects mu in log-space.
        value = rng.lognormvariate(math.log(max(median, 1e-6)), spread)
        return float(min(max(value, 0.0), cap))

    def simulate_manifest(
        self,
        manifest: CyberDatasetManifest,
        rows: int = DATASET_SIM_ROWS_PER_SOURCE,
    ) -> list[dict[str, Any]]:
        rng = random.Random(self.seed ^ int(stable_hash(manifest.dataset_id)[:8], 16))
        records = []

        for index in range(rows):
            label_roll = rng.random()
            if label_roll < 0.78:
                label = "benign"
            elif label_roll < 0.84:
                label = "scan"
            elif label_roll < 0.90:
                label = "bruteforce"
            elif label_roll < 0.95:
                label = "dos"
            elif label_roll < 0.98:
                label = "botnet-like"
            else:
                label = "anomalous"

            attack_scale = {
                "benign": 1.0,
                "scan": 1.6,
                "bruteforce": 1.9,
                "dos": 4.2,
                "botnet-like": 2.4,
                "anomalous": 3.0,
            }[label]

            features: dict[str, float] = {}
            for fi, feature in enumerate(manifest.feature_names):
                basis = 5.0 + 11.0 * ((fi % 5) + 1)
                if any(token in feature for token in ("bytes", "duration", "count", "packets", "pkts")):
                    value = self._bounded_lognormal(
                        rng,
                        median=basis * 25.0 * attack_scale,
                        spread=0.85,
                        cap=5_000_000.0,
                    )
                elif any(token in feature for token in ("ratio", "rate", "score", "entropy")):
                    value = min(
                        1.0,
                        max(
                            0.0,
                            rng.betavariate(2.0 + attack_scale, 4.5)
                            * min(attack_scale / 1.5, 1.8),
                        ),
                    )
                elif "hour" in feature:
                    value = float(rng.randrange(24))
                else:
                    value = min(100_000.0, rng.random() * basis * attack_scale)
                features[feature] = round(float(value), 6)

            # Synthetic cross-feature distortions for non-benign records.
            if label in {"scan", "botnet-like"}:
                features["synthetic_unique_destination_pressure"] = round(
                    min(1.0, 0.45 + rng.random() * 0.55), 6
                )
            if label == "dos":
                features["synthetic_burst_pressure"] = round(
                    min(1.0, 0.72 + rng.random() * 0.28), 6
                )
            if label == "bruteforce":
                features["synthetic_auth_pressure"] = round(
                    min(1.0, 0.62 + rng.random() * 0.38), 6
                )

            record_id = stable_hash(manifest.dataset_id, str(index), compact_json(features))[:32]
            summary = (
                f"SIMULATED {manifest.name} record {index}; label={label}; "
                f"features={compact_json(features)}"
            )
            records.append(
                {
                    "record_id": record_id,
                    "dataset_id": manifest.dataset_id,
                    "dataset_name": manifest.name,
                    "source_url": manifest.source_url,
                    "source_kind": "bounded-simulation",
                    "simulated": True,
                    "label": label,
                    "category": manifest.category,
                    "summary": summary,
                    "features_json": safe_json(features),
                    "safety_json": safe_json({
                        "executed_source_code": False,
                        "deserialized_python_objects": False,
                        "contains_real_identity": False,
                        "simulation_seeded": True,
                        "claim_of_dataset_equivalence": False,
                    }),
                    "ingested_at": utc_now(),
                    "tags": ["dataset", "simulated", manifest.category, manifest.dataset_id],
                }
            )
        return records


class SafeTabularDatasetImporter:
    """
    Explicit opt-in Kaggle ingestion with a hostile-file stance.

    Accepted data: CSV, TSV, JSONL, and optional Parquet via pyarrow.
    Rejected: pickle/joblib/npy/npz executables, notebooks, scripts, binaries, PCAP,
    arbitrary archive extraction, symlinks, and path traversal.
    """

    ALLOWED_SUFFIXES = {".csv", ".tsv", ".jsonl", ".ndjson", ".parquet"}
    REJECT_SUFFIXES = {
        ".pkl", ".pickle", ".joblib", ".py", ".pyc", ".ipynb", ".exe", ".dll",
        ".so", ".dylib", ".jar", ".sh", ".bat", ".ps1", ".pcap", ".pcapng",
        ".npy", ".npz", ".pt", ".pth", ".onnx",
    }

    def __init__(self, sandbox_dir: Path = DATASET_SANDBOX_DIR):
        self.sandbox_dir = sandbox_dir

    @staticmethod
    def _safe_member_name(name: str) -> bool:
        p = Path(name.replace("\\", "/"))
        if p.is_absolute():
            return False
        return ".." not in p.parts and not str(p).startswith("~")

    def _validate_zip(self, archive: Path) -> list[zipfile.ZipInfo]:
        if archive.stat().st_size > DATASET_MAX_ARCHIVE_BYTES:
            raise RuntimeError("Dataset archive exceeds configured safety size cap.")
        accepted = []
        with zipfile.ZipFile(archive) as zf:
            total_uncompressed = 0
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not self._safe_member_name(info.filename):
                    raise RuntimeError(f"Unsafe archive path: {info.filename!r}")
                suffix = Path(info.filename).suffix.lower()
                if suffix in self.REJECT_SUFFIXES:
                    continue
                if suffix not in self.ALLOWED_SUFFIXES:
                    continue
                if info.file_size > DATASET_MAX_MEMBER_BYTES:
                    raise RuntimeError(f"Dataset member too large: {info.filename}")
                total_uncompressed += info.file_size
                if total_uncompressed > DATASET_MAX_ARCHIVE_BYTES:
                    raise RuntimeError("Archive expansion exceeds safety cap.")
                accepted.append(info)
        return accepted

    async def kaggle_download(
        self,
        manifest: CyberDatasetManifest,
        cancel: threading.Event,
        status: Callable[[str], None],
    ) -> Path:
        if not manifest.kaggle_slug:
            raise RuntimeError("This source has no configured Kaggle dataset slug.")
        exe = shutil.which("kaggle")
        if not exe:
            raise RuntimeError(
                "Kaggle CLI is not installed. Install/configure it explicitly; "
                "the app never scrapes Kaggle browser credentials."
            )

        target = self.sandbox_dir / manifest.dataset_id
        target.mkdir(parents=True, exist_ok=True)
        status(f"Kaggle download → {manifest.kaggle_slug} (tabular sandbox only)")

        cmd = [
            exe, "datasets", "download",
            "-d", manifest.kaggle_slug,
            "-p", str(target),
            "--force",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        while proc.returncode is None:
            if cancel.is_set():
                proc.terminate()
                await proc.wait()
                raise RuntimeError("Kaggle download cancelled.")
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "Kaggle CLI download failed:\n"
                + stderr.decode("utf-8", "replace")[-2000:]
            )

        archives = sorted(target.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not archives:
            # Some Kaggle downloads may materialize direct files.
            direct = [p for p in target.iterdir() if p.suffix.lower() in self.ALLOWED_SUFFIXES]
            if direct:
                return target
            raise RuntimeError("Kaggle download produced no allowed tabular artifact.")

        archive = archives[0]
        accepted = self._validate_zip(archive)
        extract_root = target / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(archive) as zf:
            for info in accepted:
                destination = (extract_root / info.filename).resolve()
                if extract_root.resolve() not in destination.parents:
                    raise RuntimeError("Archive traversal detected.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src_fh, destination.open("wb") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)
        return extract_root

    @staticmethod
    def _redact_row(row: dict[str, Any]) -> dict[str, Any]:
        redacted = {}
        sensitive_tokens = ("password", "passwd", "secret", "token", "cookie", "credential")
        for key, value in row.items():
            lk = str(key).lower()
            if any(token in lk for token in sensitive_tokens):
                redacted[key] = "<redacted>"
            else:
                text = str(value)
                redacted[key] = text[:512]
        return redacted

    async def iter_rows(self, root: Path, max_rows: int = DATASET_MAX_ROWS_REAL):
        count = 0
        files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.ALLOWED_SUFFIXES
        )

        for path in files:
            suffix = path.suffix.lower()
            if path.stat().st_size > DATASET_MAX_MEMBER_BYTES:
                continue

            if suffix in {".csv", ".tsv"}:
                delimiter = "\t" if suffix == ".tsv" else ","
                with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
                    reader = csv.DictReader(fh, delimiter=delimiter)
                    for row in reader:
                        yield path.name, self._redact_row(dict(row))
                        count += 1
                        if count >= max_rows:
                            return

            elif suffix in {".jsonl", ".ndjson"}:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, dict):
                            yield path.name, self._redact_row(obj)
                            count += 1
                            if count >= max_rows:
                                return

            elif suffix == ".parquet":
                try:
                    import pyarrow.parquet as pq
                except Exception:
                    logging.warning("Skipping parquet without pyarrow: %s", path)
                    continue

                parquet = pq.ParquetFile(path)
                for batch in parquet.iter_batches(batch_size=1024):
                    table = batch.to_pydict()
                    keys = list(table)
                    rows_n = len(table[keys[0]]) if keys else 0
                    for i in range(rows_n):
                        obj = {k: table[k][i] for k in keys}
                        yield path.name, self._redact_row(obj)
                        count += 1
                        if count >= max_rows:
                            return


# =============================================================================
# Experimental Chromatic Quantum Surface Embeddings (CQSE)
# =============================================================================

class ChromaticQuantumSurfaceEncoder:
    """
    Experimental research prototype.

    CQSE is NOT a validated replacement for semantic embeddings. It creates additional
    deterministic retrieval surfaces from:
      1) an existing semantic vector (OpenAI/Contriever when available),
      2) color-space projections (linear RGB, XYZ, OKLab, OKLCH),
      3) embedding distribution statistics and entropy,
      4) a PennyLane chromatic circuit producing a 32-state probability vector.

    The extra surfaces are used as independent named vectors and fused at ranking time.
    """

    CHROMA_DIM = 96
    QUANTUM_DIM = 32

    def __init__(self):
        self._projection_cache: dict[int, np.ndarray] = {}

    @staticmethod
    def _hashed_text_vector(text: str, dim: int = 512) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float64)
        normalized = re.sub(r"\s+", " ", text.lower()).strip()
        for n in (2, 3, 4, 5):
            for i in range(max(0, len(normalized) - n + 1)):
                gram = normalized[i:i+n]
                digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                index = value % dim
                sign = -1.0 if (value >> 7) & 1 else 1.0
                vec[index] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _projection(self, input_dim: int) -> np.ndarray:
        if input_dim in self._projection_cache:
            return self._projection_cache[input_dim]
        seed = int(stable_hash("CQSE-v1", str(input_dim))[:16], 16) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 1.0, size=(12, input_dim))
        # Normalize rows for a stable Johnson-Lindenstrauss-like projection.
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        self._projection_cache[input_dim] = matrix
        return matrix

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -18.0, 18.0)))

    @staticmethod
    def _srgb_to_linear(c: float) -> float:
        c = float(np.clip(c, 0.0, 1.0))
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    @classmethod
    def _rgb_to_xyz(cls, rgb: tuple[float, float, float]) -> tuple[float, float, float]:
        r, g, b = [cls._srgb_to_linear(x) for x in rgb]
        x = 0.4124564*r + 0.3575761*g + 0.1804375*b
        y = 0.2126729*r + 0.7151522*g + 0.0721750*b
        z = 0.0193339*r + 0.1191920*g + 0.9503041*b
        return x, y, z

    @staticmethod
    def _xyz_to_oklab(x: float, y: float, z: float) -> tuple[float, float, float]:
        l = 0.8189330101*x + 0.3618667424*y - 0.1288597137*z
        m = 0.0329845436*x + 0.9293118715*y + 0.0361456387*z
        s = 0.0482003018*x + 0.2643662691*y + 0.6338517070*z
        l_, m_, s_ = np.cbrt([l, m, s])
        L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
        a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
        b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
        return float(L), float(a), float(b)

    @staticmethod
    def _oklab_to_oklch(lab: tuple[float, float, float]) -> tuple[float, float, float]:
        L, a, b = lab
        C = math.sqrt(a*a + b*b)
        H = math.atan2(b, a) / (2.0 * math.pi)
        if H < 0:
            H += 1.0
        return L, C, H

    @staticmethod
    def _distribution_features(base: np.ndarray) -> dict[str, float]:
        x = np.asarray(base, dtype=np.float64)
        abs_x = np.abs(x)
        total = float(abs_x.sum()) + 1e-12
        p = abs_x / total
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "l1": float(abs_x.sum()),
            "l2": float(np.linalg.norm(x)),
            "min": float(x.min(initial=0.0)),
            "max": float(x.max(initial=0.0)),
            "sparsity_1e-4": float(np.mean(abs_x < 1e-4)),
            "shannon_abs_bits": entropy,
        }

    def encode(
        self,
        text: str,
        semantic_vectors: Optional[dict[str, list[float]]] = None,
        telemetry_quantum: Optional[list[float]] = None,
        gemma_palette: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        semantic_vectors = semantic_vectors or {}
        if VECTOR_OPENAI in semantic_vectors:
            base = np.asarray(semantic_vectors[VECTOR_OPENAI], dtype=np.float64)
            base_source = VECTOR_OPENAI
        elif VECTOR_FB in semantic_vectors:
            base = np.asarray(semantic_vectors[VECTOR_FB], dtype=np.float64)
            base_source = VECTOR_FB
        else:
            base = self._hashed_text_vector(text, 512)
            base_source = "hashed_ngram_512"

        base_norm = np.linalg.norm(base)
        if base_norm > 0:
            base = base / base_norm

        projected = self._projection(len(base)) @ base
        channel_values = self._sigmoid(projected)

        # Four independent pseudo-color anchors from deterministic semantic projections.
        rgbs = [
            tuple(float(x) for x in channel_values[i:i+3])
            for i in range(0, 12, 3)
        ]

        palette_hex = []
        if gemma_palette and isinstance(gemma_palette.get("palette"), list):
            for item in gemma_palette["palette"][:8]:
                if isinstance(item, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", item):
                    palette_hex.append(item.upper())

        # Mix validated Gemma palette colors as *conditioning metadata*, not as raw embeddings.
        gemma_rgb = []
        for hx in palette_hex:
            gemma_rgb.append(tuple(int(hx[i:i+2], 16)/255.0 for i in (1, 3, 5)))
        if gemma_rgb:
            for i in range(min(len(rgbs), len(gemma_rgb))):
                rgbs[i] = tuple(0.72*a + 0.28*b for a, b in zip(rgbs[i], gemma_rgb[i]))

        surfaces = []
        labs = []
        lchs = []
        for rgb in rgbs:
            xyz = self._rgb_to_xyz(rgb)
            lab = self._xyz_to_oklab(*xyz)
            lch = self._oklab_to_oklch(lab)
            labs.append(lab)
            lchs.append(lch)
            surfaces.extend([*rgb, *xyz, *lab, *lch])

        dist = self._distribution_features(base)

        # 48 raw color coordinates + distribution/telemetry/gemma signals.
        feature_seed = list(surfaces)
        feature_seed.extend([
            dist["mean"], dist["std"], dist["l1"], dist["l2"],
            dist["min"], dist["max"], dist["sparsity_1e-4"],
            dist["shannon_abs_bits"] / max(math.log2(max(len(base), 2)), 1.0),
        ])
        if telemetry_quantum and len(telemetry_quantum) == 16:
            feature_seed.extend(float(x) for x in telemetry_quantum)
        else:
            feature_seed.extend([0.0] * 16)

        # Palette semantics become stable hash coordinates.
        palette_text = safe_json(gemma_palette or {})
        palette_hash = self._hashed_text_vector(palette_text, 24)
        feature_seed.extend(palette_hash.tolist())

        seed_arr = np.asarray(feature_seed, dtype=np.float64)
        # Deterministic expansion/compression to exactly 96 dimensions.
        seed = int(stable_hash("CQSE-surface-v1", str(len(seed_arr)))[:16], 16) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        proj = rng.normal(0.0, 1.0, size=(self.CHROMA_DIM, len(seed_arr)))
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-12
        chroma = np.tanh(proj @ seed_arr)
        chroma = np.asarray(l2_normalize(chroma.tolist()), dtype=np.float64)

        quantum = self._quantum_surface(chroma, labs, lchs, dist)

        return {
            "base_source": base_source,
            "palette": palette_hex,
            "rgb_anchors": rgbs,
            "oklab_anchors": labs,
            "oklch_anchors": lchs,
            "distribution": dist,
            "chromatic_surface_96": chroma.astype(np.float32).tolist(),
            "chromatic_quantum_32": quantum,
            "experimental_notice": (
                "CQSE is an experimental retrieval feature transform, not a validated "
                "scientific embedding standard and not evidence of quantum advantage."
            ),
        }

    @staticmethod
    def _quantum_surface(
        chroma: np.ndarray,
        labs: list[tuple[float, float, float]],
        lchs: list[tuple[float, float, float]],
        dist: dict[str, float],
    ) -> list[float]:
        dev = qml.device("default.qubit", wires=5)

        # 5 qubits -> 32-dimensional probability surface.
        angles = []
        anchors = []
        for lab, lch in zip(labs, lchs):
            anchors.extend([lab[0], math.tanh(lab[1]), math.tanh(lab[2]), lch[1], lch[2]])
        anchors.extend([
            math.tanh(dist["mean"]),
            math.tanh(dist["std"]),
            np.clip(dist["sparsity_1e-4"], 0, 1),
        ])
        anchors = np.asarray(anchors, dtype=np.float64)
        for i in range(15):
            v = float(anchors[i % len(anchors)])
            c = float(chroma[(i * 7) % len(chroma)])
            angles.append(math.pi * np.clip(0.5 + 0.26*v + 0.24*c, 0.0, 1.0))

        @qml.qnode(dev)
        def circuit():
            k = 0
            for layer in range(3):
                for wire in range(5):
                    qml.RY(angles[k], wires=wire)
                    qml.RZ((angles[k] - math.pi/2) / (layer + 1), wires=wire)
                    k += 1
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 4])
                qml.CNOT(wires=[4, 0])
                if layer == 1:
                    qml.CZ(wires=[0, 3])
                    qml.CZ(wires=[1, 4])
            return qml.probs(wires=[0, 1, 2, 3, 4])

        probs = np.asarray(circuit(), dtype=np.float64)
        return probs.astype(np.float32).tolist()


# =============================================================================
# Gemma 4 E2B LiteRT-LM secure model manager / palette synthesizer
# =============================================================================

class Gemma4LiteRTManager:
    """
    Uses LiteRT-LM for the .litertlm artifact.

    Deliberately does NOT route .litertlm through llama-cpp-python: that runtime expects
    llama.cpp-compatible model formats (e.g. GGUF), not LiteRT-LM packages.
    """

    def __init__(self):
        self._download_lock = asyncio.Lock()

    @staticmethod
    def runtime_available() -> bool:
        return shutil.which("litert-lm") is not None

    async def verify(self) -> dict[str, Any]:
        if not GEMMA4_MODEL_PATH.exists():
            return {"ok": False, "reason": "missing"}

        def digest_file():
            h = hashlib.sha256()
            size = 0
            with GEMMA4_MODEL_PATH.open("rb") as fh:
                while True:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    h.update(chunk)
            return size, h.hexdigest()

        size, actual = await asyncio.to_thread(digest_file)
        ok = actual.lower() == GEMMA4_MODEL_SHA256.lower()
        receipt = {}
        if GEMMA4_RECEIPT_PATH.exists():
            try:
                receipt = json.loads(GEMMA4_RECEIPT_PATH.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "ok": ok,
            "size": size,
            "actual_sha256": actual,
            "expected_sha256": GEMMA4_MODEL_SHA256,
            "receipt": receipt,
        }

    async def download(
        self,
        cancel: threading.Event,
        progress: Callable[[float, str], None],
    ) -> dict[str, Any]:
        async with self._download_lock:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            current = await self.verify()
            if current.get("ok"):
                progress(1.0, "Gemma 4 E2B already verified.")
                return current
            if GEMMA4_MODEL_PATH.exists():
                GEMMA4_MODEL_PATH.unlink(missing_ok=True)

            offset = GEMMA4_PART_PATH.stat().st_size if GEMMA4_PART_PATH.exists() else 0
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"

            expected_total: Optional[int] = None
            started = time.monotonic()

            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=30.0, read=None, write=60.0, pool=30.0),
                follow_redirects=True,
                verify=True,
            ) as client:
                async with client.stream("GET", GEMMA4_MODEL_URL, headers=headers) as response:
                    if response.url.scheme.lower() != "https":
                        raise RuntimeError("Refusing non-HTTPS Gemma model redirect.")

                    if offset:
                        if response.status_code != 206:
                            GEMMA4_PART_PATH.unlink(missing_ok=True)
                            raise RuntimeError(
                                "Server ignored Gemma Range resume; partial file deleted."
                            )
                        content_range = response.headers.get("Content-Range", "")
                        match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
                        if not match or int(match.group(1)) != offset:
                            GEMMA4_PART_PATH.unlink(missing_ok=True)
                            raise RuntimeError(
                                "Gemma Content-Range mismatch; partial file deleted."
                            )
                        if match.group(3) != "*":
                            expected_total = int(match.group(3))
                    else:
                        response.raise_for_status()
                        content_length = response.headers.get("Content-Length")
                        if content_length and content_length.isdigit():
                            expected_total = int(content_length)

                    response.raise_for_status()
                    mode = "ab" if offset else "wb"
                    downloaded = offset

                    with GEMMA4_PART_PATH.open(mode) as fh:
                        async for chunk in response.aiter_bytes(8 * 1024 * 1024):
                            if cancel.is_set():
                                raise RuntimeError("Gemma model download cancelled.")
                            if not chunk:
                                continue
                            fh.write(chunk)
                            downloaded += len(chunk)
                            elapsed = max(time.monotonic() - started, 0.001)
                            speed = max(downloaded - offset, 0) / elapsed
                            if expected_total:
                                fraction = min(downloaded / expected_total, 1.0)
                                remaining = max(expected_total - downloaded, 0)
                                eta = remaining / speed if speed > 0 else float("inf")
                                progress(
                                    fraction,
                                    f"Gemma {human_bytes(downloaded)} / {human_bytes(expected_total)} "
                                    f"• {human_bytes(speed)}/s • ETA {eta:.0f}s",
                                )
                            else:
                                progress(
                                    0.0,
                                    f"Gemma {human_bytes(downloaded)} • {human_bytes(speed)}/s",
                                )
                        fh.flush()
                        os.fsync(fh.fileno())

            if expected_total and GEMMA4_PART_PATH.stat().st_size != expected_total:
                raise RuntimeError("Gemma transfer size does not match server-pinned total.")

            def digest_part():
                h = hashlib.sha256()
                size = 0
                with GEMMA4_PART_PATH.open("rb") as fh:
                    while True:
                        chunk = fh.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        h.update(chunk)
                return size, h.hexdigest()

            size, actual = await asyncio.to_thread(digest_part)
            if actual.lower() != GEMMA4_MODEL_SHA256.lower():
                GEMMA4_PART_PATH.unlink(missing_ok=True)
                raise RuntimeError(
                    "Gemma SHA-256 mismatch. Downloaded bytes were deleted.\n"
                    f"Expected: {GEMMA4_MODEL_SHA256}\nActual:   {actual}"
                )

            os.replace(GEMMA4_PART_PATH, GEMMA4_MODEL_PATH)
            receipt = {
                "url": GEMMA4_MODEL_URL,
                "repo": GEMMA4_REPO,
                "filename": GEMMA4_MODEL_NAME,
                "size": size,
                "sha256": actual,
                "verified_at": utc_now(),
            }
            GEMMA4_RECEIPT_PATH.write_text(
                json.dumps(receipt, indent=2),
                encoding="utf-8",
            )
            progress(1.0, "Gemma 4 E2B verified and atomically installed.")
            return {"ok": True, **receipt}

    async def palette_prompt(
        self,
        text: str,
        cancel: threading.Event,
        status: Callable[[str], None],
    ) -> dict[str, Any]:
        verified = await self.verify()
        if not verified.get("ok"):
            raise RuntimeError("Gemma 4 E2B is not SHA-256 verified.")
        if not self.runtime_available():
            raise RuntimeError(
                "LiteRT-LM CLI is not installed. Install the official `litert-lm` runtime."
            )

        prompt = (
            "You are a deterministic semantic palette adapter. Do NOT explain. "
            "For the cybersecurity text below, return STRICT JSON only with keys: "
            '"palette" (exactly 8 #RRGGBB colors), "axes" (object with keys '
            '"threat","novelty","volatility","confidence" each 0..1), '
            '"keywords" (up to 12 short strings), and "summary" (max 40 words). '
            "Colors are semantic coordinates, not visual decoration. "
            "Avoid using secrets or personal identifiers. TEXT:\n"
            + text[:12000]
        )

        status("Gemma 4 E2B → LiteRT-LM semantic palette synthesis")
        cmd = [
            shutil.which("litert-lm") or "litert-lm",
            "run",
            str(GEMMA4_MODEL_PATH),
            f"--prompt={prompt}",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        while proc.returncode is None:
            if cancel.is_set():
                proc.terminate()
                await proc.wait()
                raise RuntimeError("Gemma palette generation cancelled.")
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.2)
            except asyncio.TimeoutError:
                pass
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                "LiteRT-LM Gemma invocation failed:\n"
                + stderr.decode("utf-8", "replace")[-2000:]
            )

        raw = stdout.decode("utf-8", "replace").strip()
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise RuntimeError("Gemma did not return a JSON object.")
        obj = json.loads(match.group(0))

        palette = obj.get("palette")
        if not isinstance(palette, list) or len(palette) != 8:
            raise RuntimeError("Gemma palette JSON did not contain exactly 8 colors.")
        clean_palette = []
        for value in palette:
            if not isinstance(value, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
                raise RuntimeError(f"Invalid Gemma palette color: {value!r}")
            clean_palette.append(value.upper())
        obj["palette"] = clean_palette

        axes = obj.get("axes") if isinstance(obj.get("axes"), dict) else {}
        obj["axes"] = {
            k: float(np.clip(float(axes.get(k, 0.5)), 0.0, 1.0))
            for k in ("threat", "novelty", "volatility", "confidence")
        }
        obj["keywords"] = [
            str(x)[:48]
            for x in (obj.get("keywords") if isinstance(obj.get("keywords"), list) else [])
        ][:12]
        obj["summary"] = str(obj.get("summary") or "")[:320]
        obj["model"] = GEMMA4_MODEL_NAME
        obj["model_sha256"] = GEMMA4_MODEL_SHA256
        return obj

class WeaviateMemory:
    def __init__(
        self,
        store: SQLiteStore,
        embeddings: EmbeddingEnsemble,
        config_getter: Callable[[], AppConfig],
        weaviate_url_getter: Callable[[], str],
        weaviate_key_getter: Callable[[], str],
    ):
        self.store = store
        self.embeddings = embeddings
        self.config_getter = config_getter
        self.weaviate_url_getter = weaviate_url_getter
        self.weaviate_key_getter = weaviate_key_getter
        self.cqse = ChromaticQuantumSurfaceEncoder()

    def _client(self):
        if not WEAVIATE_IMPORT_OK:
            raise RuntimeError("weaviate-client is not installed.")

        url = self.weaviate_url_getter().strip()
        key = self.weaviate_key_getter().strip()
        if url:
            if not key:
                raise RuntimeError("WEAVIATE_URL is set but no Weaviate API key is configured.")
            return weaviate.use_async_with_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(key),
            )
        return weaviate.use_async_with_local()

    @staticmethod
    def _dtype(name: str):
        return getattr(DataType, name)

    @staticmethod
    def _vector_configs(include_quantum: bool, include_chromatic: bool = False):
        configs = [
            Configure.Vectors.self_provided(name=VECTOR_OPENAI),
            Configure.Vectors.self_provided(name=VECTOR_FB),
        ]
        if include_quantum:
            configs.append(Configure.Vectors.self_provided(name=VECTOR_QUANTUM))
        if include_chromatic:
            configs.append(Configure.Vectors.self_provided(name=VECTOR_CHROMA))
            configs.append(Configure.Vectors.self_provided(name=VECTOR_CHROMA_QUANTUM))
        return configs

    async def setup_collections(self) -> dict[str, str]:
        result: dict[str, str] = {}
        async with self._client() as client:
            for name, schema in WEAVIATE_SCHEMAS.items():
                if await client.collections.exists(name):
                    result[name] = "exists"
                    continue

                properties = [
                    Property(name=prop_name, data_type=self._dtype(dtype))
                    for prop_name, dtype in schema["properties"]
                ]
                await client.collections.create(
                    name=name,
                    vector_config=self._vector_configs(bool(schema["quantum"]), bool(schema.get("chromatic"))),
                    properties=properties,
                )
                result[name] = "created"
        return result

    async def status(self) -> dict[str, Any]:
        output: dict[str, Any] = {"ready": False, "collections": {}}
        async with self._client() as client:
            output["ready"] = bool(await client.is_ready())
            for name in WEAVIATE_SCHEMAS:
                exists = await client.collections.exists(name)
                item: dict[str, Any] = {"exists": exists}
                if exists:
                    try:
                        c = client.collections.use(name)
                        agg = await c.aggregate.over_all(total_count=True)
                        item["count"] = getattr(agg, "total_count", None)
                    except Exception as exc:
                        item["count_error"] = str(exc)
                output["collections"][name] = item
        return output

    @staticmethod
    def semantic_text(collection_name: str, properties: dict[str, Any]) -> str:
        preferred = [
            "summary", "title", "evidence", "hypothesis", "description",
            "final_text", "observations", "risks", "reply_template",
            "anomaly_type", "actor", "status", "resolution",
        ]
        parts = [collection_name]
        for key in preferred:
            value = properties.get(key)
            if value not in (None, "", [], {}):
                parts.append(f"{key}: {value}")
        if len(parts) == 1:
            parts.append(safe_json(properties))
        return "\n".join(parts)

    async def insert(
        self,
        collection_name: str,
        properties: dict[str, Any],
        source_kind: str,
        source_id: str,
        quantum_vector: Optional[list[float]] = None,
    ) -> dict[str, Any]:
        if collection_name not in WEAVIATE_SCHEMAS:
            raise KeyError(collection_name)

        semantic_text = self.semantic_text(collection_name, properties)
        vectors, embed_errors = await self.embeddings.embed_all(semantic_text)

        if WEAVIATE_SCHEMAS[collection_name].get("chromatic"):
            cqse = await asyncio.to_thread(
                self.cqse.encode,
                semantic_text,
                vectors,
                quantum_vector,
                None,
            )
            vectors[VECTOR_CHROMA] = cqse["chromatic_surface_96"]
            vectors[VECTOR_CHROMA_QUANTUM] = cqse["chromatic_quantum_32"]

        if quantum_vector is not None and WEAVIATE_SCHEMAS[collection_name]["quantum"]:
            if len(quantum_vector) != 16:
                raise ValueError("quantum vector must be 16-dimensional")
            vectors[VECTOR_QUANTUM] = [float(x) for x in quantum_vector]

        object_uuid = deterministic_uuid(collection_name, source_id)

        async with self._client() as client:
            collection = client.collections.use(collection_name)

            # Deterministic UUIDs make memory ingestion idempotent. If a snapshot
            # is synchronized twice, refresh its properties/vectors rather than
            # generating duplicate semantic memories.
            exists = await collection.data.exists(object_uuid)
            if exists:
                kwargs = {
                    "uuid": object_uuid,
                    "properties": properties,
                }
                if vectors:
                    kwargs["vector"] = vectors
                await collection.data.update(**kwargs)
            else:
                kwargs = {
                    "properties": properties,
                    "uuid": object_uuid,
                }
                if vectors:
                    kwargs["vector"] = vectors
                await collection.data.insert(**kwargs)

        await self.store.mark_synced(
            source_kind,
            source_id,
            collection_name,
            str(object_uuid),
        )
        return {
            "collection": collection_name,
            "uuid": str(object_uuid),
            "vectors": list(vectors),
            "embedding_errors": embed_errors,
        }

    async def _query_one_collection(
        self,
        client,
        collection_name: str,
        query: str,
        vectors: dict[str, list[float]],
        quantum_vector: Optional[list[float]],
        limit: int,
    ) -> list[dict[str, Any]]:
        collection = client.collections.use(collection_name)
        ranked_lists: list[tuple[str, list[Any], float]] = []

        # BM25 lexical list.
        try:
            bm = await collection.query.bm25(
                query=query,
                limit=limit,
                return_metadata=MetadataQuery(score=True),
            )
            ranked_lists.append(("bm25", list(bm.objects), 0.70))
        except Exception as exc:
            logging.debug("BM25 failed on %s: %s", collection_name, exc)

        # OpenAI semantic list.
        if VECTOR_OPENAI in vectors:
            try:
                res = await collection.query.near_vector(
                    near_vector=vectors[VECTOR_OPENAI],
                    target_vector=VECTOR_OPENAI,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
                ranked_lists.append((VECTOR_OPENAI, list(res.objects), 1.00))
            except Exception as exc:
                logging.debug("OpenAI vector query failed on %s: %s", collection_name, exc)

        # Facebook Contriever list.
        if VECTOR_FB in vectors:
            try:
                res = await collection.query.near_vector(
                    near_vector=vectors[VECTOR_FB],
                    target_vector=VECTOR_FB,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
                ranked_lists.append((VECTOR_FB, list(res.objects), 0.90))
            except Exception as exc:
                logging.debug("Facebook vector query failed on %s: %s", collection_name, exc)

        # Experimental CQSE chromatic surfaces.
        if WEAVIATE_SCHEMAS[collection_name].get("chromatic"):
            try:
                cqse_query = await asyncio.to_thread(
                    self.cqse.encode,
                    query,
                    vectors,
                    quantum_vector,
                    None,
                )
                res = await collection.query.near_vector(
                    near_vector=cqse_query["chromatic_surface_96"],
                    target_vector=VECTOR_CHROMA,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
                ranked_lists.append((VECTOR_CHROMA, list(res.objects), 0.78))
                res_q = await collection.query.near_vector(
                    near_vector=cqse_query["chromatic_quantum_32"],
                    target_vector=VECTOR_CHROMA_QUANTUM,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
                ranked_lists.append((VECTOR_CHROMA_QUANTUM, list(res_q.objects), 0.62))
            except Exception as exc:
                logging.debug("CQSE query failed on %s: %s", collection_name, exc)

        # Quantum state recall for telemetry/analysis episode collections.
        if (
            quantum_vector is not None
            and WEAVIATE_SCHEMAS[collection_name]["quantum"]
            and len(quantum_vector) == 16
        ):
            try:
                res = await collection.query.near_vector(
                    near_vector=quantum_vector,
                    target_vector=VECTOR_QUANTUM,
                    limit=limit,
                    return_metadata=MetadataQuery(distance=True),
                )
                ranked_lists.append((VECTOR_QUANTUM, list(res.objects), 0.65))
            except Exception as exc:
                logging.debug("Quantum vector query failed on %s: %s", collection_name, exc)

        cfg = self.config_getter()
        fused: dict[str, dict[str, Any]] = {}

        for source_name, objects, weight in ranked_lists:
            for rank, obj in enumerate(objects, start=1):
                object_id = str(obj.uuid)
                item = fused.setdefault(
                    object_id,
                    {
                        "uuid": object_id,
                        "collection": collection_name,
                        "properties": dict(obj.properties or {}),
                        "rrf_score": 0.0,
                        "sources": [],
                    },
                )
                item["rrf_score"] += weight / (cfg.rrf_k + rank)
                item["sources"].append({"source": source_name, "rank": rank})

        return sorted(
            fused.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:limit]

    async def search(
        self,
        query: str,
        quantum_vector: Optional[list[float]] = None,
        limit_total: int = 10,
    ) -> dict[str, Any]:
        vectors, embed_errors = await self.embeddings.embed_all(query)
        per_collection = max(4, min(12, limit_total))

        async with self._client() as client:
            tasks = [
                self._query_one_collection(
                    client,
                    name,
                    query,
                    vectors,
                    quantum_vector,
                    per_collection,
                )
                for name in WEAVIATE_SCHEMAS
                if await client.collections.exists(name)
            ]
            results_nested = await asyncio.gather(*tasks) if tasks else []

        flat = [item for group in results_nested for item in group]
        flat.sort(key=lambda x: x["rrf_score"], reverse=True)

        return {
            "query": query,
            "vector_spaces": list(vectors),
            "embedding_errors": embed_errors,
            "results": flat[:limit_total],
        }


# =============================================================================
# OpenAI / Codex providers
# =============================================================================

@dataclass
class ProviderResult:
    provider: str
    model: str
    text: str
    elapsed_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderError(RuntimeError):
    pass


class CodexDaybreakProvider:
    name = "Codex CLI / ChatGPT auth"

    def available(self) -> bool:
        return shutil.which("codex") is not None

    def version(self) -> str:
        exe = shutil.which("codex")
        if not exe:
            return "not installed"
        try:
            cp = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return (cp.stdout or cp.stderr).strip() or "installed"
        except Exception as exc:
            return f"installed; version check failed: {exc}"

    def launch_login(self):
        exe = shutil.which("codex")
        if not exe:
            raise ProviderError("Codex CLI is not installed.")
        if platform.system().lower() == "windows":
            flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            subprocess.Popen([exe, "login"], creationflags=flags)
        else:
            subprocess.Popen([exe, "login"])

    @staticmethod
    def _extract_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
        found: dict[str, Any] = {}

        def walk(value: Any):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in {
                        "input_tokens", "output_tokens", "total_tokens",
                        "cached_input_tokens", "reasoning_tokens",
                    } and isinstance(v, (int, float)):
                        found[k] = v
                    else:
                        walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)

        for e in events:
            walk(e)
        return found

    def _run_sync(
        self,
        prompt: str,
        cancel: threading.Event,
        status: Callable[[str], None],
    ) -> ProviderResult:
        exe = shutil.which("codex")
        if not exe:
            raise ProviderError("Codex CLI not installed.")

        started = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="qnet-codex-") as td:
            output_file = Path(td) / "final.txt"
            cmd = [
                exe,
                "exec",
                "--skip-git-repo-check",
                "--json",
                "--model",
                DAYBREAK_ALIAS,
                "--sandbox",
                "read-only",
                "-o",
                str(output_file),
                prompt,
            ]

            status(f"Codex → {DAYBREAK_ALIAS} • read-only sandbox")
            proc = subprocess.Popen(
                cmd,
                cwd=td,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            while proc.poll() is None:
                if cancel.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise ProviderError("Codex request cancelled.")
                time.sleep(0.1)

            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                tail = "\n".join(stderr.splitlines()[-30:]).strip()
                raise ProviderError(f"Codex CLI failed:\n{tail}")

            if not output_file.exists():
                raise ProviderError("Codex produced no final output file.")

            text = output_file.read_text(encoding="utf-8").strip()
            if not text:
                raise ProviderError("Codex returned empty output.")

            events = []
            for line in stdout.splitlines():
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass

            return ProviderResult(
                provider=self.name,
                model=DAYBREAK_ALIAS,
                text=text,
                elapsed_seconds=time.monotonic() - started,
                usage=self._extract_usage(events),
                metadata={
                    "sandbox": "read-only",
                    "credential_boundary": "Codex CLI owns authentication",
                    "event_count": len(events),
                },
            )

    async def generate(self, prompt, cancel, status):
        return await asyncio.to_thread(self._run_sync, prompt, cancel, status)


class ResponsesProvider:
    name = "OpenAI Responses API"

    def __init__(
        self,
        api_key_getter: Callable[[], str],
        model: str,
        access_program: str,
        reasoning_effort_getter: Callable[[], str],
    ):
        self.api_key_getter = api_key_getter
        self.model = model
        self.access_program = access_program
        self.reasoning_effort_getter = reasoning_effort_getter

    def available(self):
        return bool(self.api_key_getter().strip())

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        direct = data.get("output_text")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        parts = []
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "text"}:
                    text = content.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(parts).strip()

    async def generate(self, prompt, cancel, status):
        key = self.api_key_getter().strip()
        if not key:
            raise ProviderError("OpenAI API key not configured.")
        if cancel.is_set():
            raise ProviderError("Cancelled.")

        status(f"Responses API → {self.model} • cyber={self.access_program}")
        started = time.monotonic()
        payload = {
            "model": self.model,
            "input": prompt,
            "reasoning": {"effort": self.reasoning_effort_getter()},
            "max_output_tokens": 6000,
            "access_programs": {"cyber": self.access_program},
        }

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, verify=True) as client:
            response = await client.post(
                RESPONSES_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

        if response.status_code in (401, 403):
            raise ProviderError(
                f"OpenAI access denied ({response.status_code}). Check the API project, "
                "model permission, and Daybreak provisioning."
            )
        response.raise_for_status()
        data = response.json()
        text = self._extract_text(data)
        if not text:
            raise ProviderError("Responses API returned no output text.")

        return ProviderResult(
            provider=self.name,
            model=str(data.get("model") or self.model),
            text=text,
            elapsed_seconds=time.monotonic() - started,
            usage=data.get("usage") if isinstance(data.get("usage"), dict) else {},
            metadata={
                "response_id": data.get("id"),
                "access_programs": data.get("access_programs"),
            },
        )


class LegacyGGMLManager:
    """Optional exact historical GGML model with fail-closed SHA-256 verification."""

    async def verify(self) -> bool:
        if not LEGACY_MODEL_PATH.exists():
            return False
        if LEGACY_MODEL_PATH.stat().st_size != LEGACY_MODEL_SIZE:
            return False

        def digest():
            h = hashlib.sha256()
            with LEGACY_MODEL_PATH.open("rb") as fh:
                while True:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()

        return (await asyncio.to_thread(digest)).lower() == LEGACY_MODEL_SHA256

    async def download(
        self,
        cancel: threading.Event,
        progress: Callable[[float, str], None],
    ):
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        if LEGACY_MODEL_PATH.exists():
            if await self.verify():
                progress(1.0, "Legacy GGML model already verified.")
                return
            LEGACY_MODEL_PATH.unlink(missing_ok=True)
            raise ProviderError("Installed legacy model failed SHA-256 and was deleted.")

        offset = LEGACY_PART_PATH.stat().st_size if LEGACY_PART_PATH.exists() else 0
        if offset > LEGACY_MODEL_SIZE:
            LEGACY_PART_PATH.unlink(missing_ok=True)
            offset = 0

        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30, read=None, write=60, pool=30),
            verify=True,
            follow_redirects=True,
        ) as client:
            async with client.stream("GET", LEGACY_MODEL_URL, headers=headers) as response:
                if response.url.scheme.lower() != "https":
                    raise ProviderError("Refusing non-HTTPS model redirect.")
                if offset:
                    if response.status_code != 206:
                        LEGACY_PART_PATH.unlink(missing_ok=True)
                        raise ProviderError(
                            "Server ignored the HTTP Range resume request. "
                            "Partial file was deleted; start the download again."
                        )
                    content_range = response.headers.get("Content-Range", "")
                    match = re.fullmatch(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", content_range.strip())
                    if not match or int(match.group(1)) != offset:
                        LEGACY_PART_PATH.unlink(missing_ok=True)
                        raise ProviderError(
                            "Resume Content-Range did not match the requested offset. "
                            "Partial file was deleted."
                        )
                response.raise_for_status()

                mode = "ab" if offset else "wb"
                downloaded = offset
                with LEGACY_PART_PATH.open(mode) as fh:
                    async for chunk in response.aiter_bytes(8 * 1024 * 1024):
                        if cancel.is_set():
                            raise ProviderError("Legacy model download cancelled.")
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if downloaded > LEGACY_MODEL_SIZE:
                            fh.close()
                            LEGACY_PART_PATH.unlink(missing_ok=True)
                            raise ProviderError("Legacy transfer exceeded pinned size.")
                        progress(
                            min(downloaded / LEGACY_MODEL_SIZE, 1.0),
                            f"Legacy model {human_bytes(downloaded)} / {human_bytes(LEGACY_MODEL_SIZE)}",
                        )
                    fh.flush()
                    os.fsync(fh.fileno())

        if LEGACY_PART_PATH.stat().st_size != LEGACY_MODEL_SIZE:
            raise ProviderError("Legacy model transfer incomplete.")

        def hash_part():
            h = hashlib.sha256()
            with LEGACY_PART_PATH.open("rb") as fh:
                while True:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()

        actual = await asyncio.to_thread(hash_part)
        if actual.lower() != LEGACY_MODEL_SHA256:
            LEGACY_PART_PATH.unlink(missing_ok=True)
            raise ProviderError(
                "Legacy model SHA-256 mismatch. Corrupt data deleted.\n"
                f"Expected: {LEGACY_MODEL_SHA256}\nActual:   {actual}"
            )
        os.replace(LEGACY_PART_PATH, LEGACY_MODEL_PATH)
        progress(1.0, "Legacy GGML model verified and atomically installed.")


class LegacyLlamaProvider:
    name = "Local Llama-2 GGMLv3"

    def __init__(self, manager: LegacyGGMLManager):
        self.manager = manager
        self._llm = None
        self._lock = threading.Lock()

    async def available(self) -> bool:
        try:
            import llama_cpp  # noqa
        except Exception:
            return False
        return await self.manager.verify()

    def _generate_sync(self, prompt: str):
        with self._lock:
            if self._llm is None:
                from llama_cpp import Llama
                self._llm = Llama(
                    model_path=str(LEGACY_MODEL_PATH),
                    n_ctx=4096,
                    n_threads=max(1, (os.cpu_count() or 4) - 1),
                    verbose=False,
                )
            wrapped = (
                "<s>[INST] <<SYS>>\n"
                "You are a defensive network analysis assistant. Use only supplied evidence. "
                "Do not invent malware, devices, outages, or causes.\n"
                "<</SYS>>\n\n"
                + prompt
                + " [/INST]"
            )
            out = self._llm(
                wrapped,
                max_tokens=900,
                temperature=0.2,
                top_p=0.9,
                stop=["</s>", "[INST]"],
                echo=False,
            )
            return out["choices"][0]["text"].strip()

    async def generate(self, prompt, cancel, status):
        if not await self.available():
            raise ProviderError("Verified local legacy model/runtime unavailable.")
        if cancel.is_set():
            raise ProviderError("Cancelled.")
        status("Local Llama-2 fallback • verified GGMLv3")
        started = time.monotonic()
        text = await asyncio.to_thread(self._generate_sync, prompt)
        return ProviderResult(
            provider=self.name,
            model=LEGACY_MODEL_NAME,
            text=text,
            elapsed_seconds=time.monotonic() - started,
            metadata={"sha256": LEGACY_MODEL_SHA256},
        )


class ProviderOrchestrator:
    def __init__(
        self,
        api_key_getter: Callable[[], str],
        reasoning_effort_getter: Callable[[], str],
        legacy_manager: LegacyGGMLManager,
    ):
        self.codex = CodexDaybreakProvider()
        self.daybreak_api = ResponsesProvider(
            api_key_getter,
            DAYBREAK_ALIAS,
            "daybreak_blue",
            reasoning_effort_getter,
        )
        self.standard_api = ResponsesProvider(
            api_key_getter,
            STANDARD_MODEL,
            "standard",
            reasoning_effort_getter,
        )
        self.local = LegacyLlamaProvider(legacy_manager)

    async def generate(
        self,
        prompt: str,
        cancel: threading.Event,
        status: Callable[[str], None],
        preference: str = "Auto",
    ) -> tuple[ProviderResult, list[str]]:
        chain: list[Any]
        if preference == "Codex Daybreak":
            chain = [self.codex]
        elif preference == "API Daybreak":
            chain = [self.daybreak_api]
        elif preference == "API GPT-6 Sol":
            chain = [self.standard_api]
        elif preference == "Local Llama":
            chain = [self.local]
        else:
            chain = [self.codex, self.daybreak_api, self.standard_api, self.local]

        errors = []
        for provider in chain:
            try:
                if provider is self.codex and not self.codex.available():
                    errors.append("Codex CLI: not installed")
                    continue
                if isinstance(provider, ResponsesProvider) and not provider.available():
                    errors.append(f"{provider.model}: API key not configured")
                    continue
                if provider is self.local and not await self.local.available():
                    errors.append("Local Llama: verified model/runtime unavailable")
                    continue

                return await provider.generate(prompt, cancel, status), errors
            except Exception as exc:
                errors.append(f"{getattr(provider, 'name', type(provider).__name__)}: {exc}")
                if preference != "Auto":
                    raise

        raise ProviderError("No provider completed the request.\n\n" + "\n".join(errors))


# =============================================================================
# Prompt builder
# =============================================================================

ANALYSIS_MODES = [
    "Deep Network Analysis",
    "Predictive Risk",
    "Adaptive Circuit Audit",
    "Malware / Traffic Triage",
    "Bandwidth Consumers",
    "Incident Commander",
    "Memory Correlation",
    "Temporal Hyperfield Analysis",
]


def build_prompt(
    mode: str,
    analytics: dict[str, Any],
    memory_hits: list[dict[str, Any]],
    temporal_hyperfield: Optional[dict[str, Any]] = None,
) -> str:
    tasks = {
        "Deep Network Analysis": (
            "Assess current condition, bottleneck hypotheses, robust anomalies, temporal "
            "patterns, forecast risks, and safe optimizations."
        ),
        "Predictive Risk": (
            "Focus on trend direction, next-sample forecasts, time-of-day recurrence, "
            "capacity risk, and evidence that would falsify each forecast."
        ),
        "Adaptive Circuit Audit": (
            "Audit the PennyLane feature map: robust scaling, normalized inputs, RY/RZ "
            "angles, probability distribution, Pauli-Z expectations, entropy, saturation, "
            "and whether the feature transform preserves useful separation. Do not claim "
            "quantum speedup."
        ),
        "Malware / Traffic Triage": (
            "Use only flow evidence for security claims. Triage statistical flow anomalies "
            "and missing evidence. Never label malware as confirmed without packet/DNS/IDS/"
            "EDR/endpoint evidence."
        ),
        "Bandwidth Consumers": (
            "Analyze top talkers, concentration, resource contention, and safe QoS/scheduling "
            "or investigation options. Treat src_ip as a proxy when device identity is absent."
        ),
        "Incident Commander": (
            "Produce an operator handoff: what changed, impact, immediate safe checks, "
            "escalation criteria, and a stakeholder-ready status template."
        ),
        "Memory Correlation": (
            "Correlate the current telemetry with retrieved historical incidents, analyses, "
            "actions, and telemetry memories. Separate genuine recurrence from superficial "
            "semantic similarity."
        ),
        "Temporal Hyperfield Analysis": (
            "Analyze the supplied TCQH temporal worldline. Focus on attractor regimes, "
            "regime transitions, multivariate boundary energy, OKLab trajectory geometry, "
            "velocity/acceleration/curvature, CQSE quantum entropy, Jensen-Shannon quantum "
            "state divergence, and cross-regime memory resonance. Treat these as experimental "
            "feature-space structures rather than literal physical quantum dynamics. "
            "Identify defensible operational hypotheses and safe verification actions."
        ),
    }

    compact_memory = []
    for hit in memory_hits[:12]:
        compact_memory.append({
            "collection": hit.get("collection"),
            "rrf_score": hit.get("rrf_score"),
            "sources": hit.get("sources"),
            "properties": hit.get("properties"),
        })

    return (
        "You are the defensive analysis engine of a local network observability console.\n"
        "The current telemetry and statistics were computed locally and are the source of truth.\n"
        "Do not invent measurements, causes, devices, packet captures, malware signatures, or outages.\n"
        "Do not request or expose credentials. Proposed actions are advisory and require human approval.\n"
        "Historical vector-memory matches are context, not proof of causality.\n\n"
        f"TASK:\n{tasks[mode]}\n\n"
        f"OUTPUT PROTOCOL:\n{PROTOCOL_INSTRUCTIONS}\n\n"
        "CURRENT DETERMINISTIC ANALYTICS:\n"
        + json.dumps(analytics, indent=2, default=str)
        + "\n\nRETRIEVED MEMORY:\n"
        + json.dumps(compact_memory, indent=2, default=str)
        + "\n\nTEMPORAL HYPERFIELD:\n"
        + json.dumps(
            (
                {
                    "generated_at": temporal_hyperfield.get("generated_at"),
                    "point_count": temporal_hyperfield.get("point_count"),
                    "regimes": temporal_hyperfield.get("regimes"),
                    "boundaries": temporal_hyperfield.get("boundaries"),
                    "experimental_notice": temporal_hyperfield.get("experimental_notice"),
                }
                if temporal_hyperfield
                else {"available": False}
            ),
            indent=2,
            default=str,
        )
    )


# =============================================================================
# Memory ingestion helpers
# =============================================================================

def telemetry_memory_properties(analytics: dict[str, Any]) -> dict[str, Any]:
    c = analytics["current"]
    trends = {
        metric: analytics["metrics"][metric]["trend"]
        for metric in QuantumFeatureEngine.METRICS
    }
    anomaly_count = len(analytics["current_anomalies"])
    health = "anomalous" if anomaly_count else "nominal"
    summary = (
        f"Telemetry {c['timestamp']}: ping {c['ping']:.2f} ms, jitter {c['jitter']:.2f} ms, "
        f"download {c['download_speed']:.2f} Mbps, upload {c['upload_speed']:.2f} Mbps; "
        f"{anomaly_count} current robust anomalies."
    )
    return {
        "event_id": stable_hash(str(c["timestamp"]), compact_json(c))[:32],
        "timestamp": str(c["timestamp"]),
        "summary": summary,
        "ping_ms": float(c["ping"]),
        "jitter_ms": float(c["jitter"]),
        "download_mbps": float(c["download_speed"]),
        "upload_mbps": float(c["upload_speed"]),
        "anomaly_count": anomaly_count,
        "health_state": health,
        "trends_json": safe_json(trends),
        "quantum_json": safe_json(analytics["quantum"]),
        "tags": ["telemetry", health],
    }


# =============================================================================
# GUI
# =============================================================================

class GuiLogHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    def emit(self, record):
        try:
            self.sink(self.format(record))
        except Exception:
            pass


class MetricCard(ctk.CTkFrame):
    def __init__(self, master, title: str, unit: str):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text=title, anchor="w").grid(
            row=0, column=0, padx=12, pady=(10, 1), sticky="ew"
        )
        self.value = ctk.CTkLabel(
            self,
            text="—",
            font=ctk.CTkFont(size=25, weight="bold"),
            anchor="w",
        )
        self.value.grid(row=1, column=0, padx=12, pady=(0, 1), sticky="ew")
        self.detail = ctk.CTkLabel(self, text=unit, anchor="w")
        self.detail.grid(row=2, column=0, padx=12, pady=(0, 10), sticky="ew")

    def set(self, value: str, detail: Optional[str] = None):
        self.value.configure(text=value)
        if detail is not None:
            self.detail.configure(text=detail)


class QuantumNetworkHyperConsole(ctk.CTk):
    def __init__(self):
        super().__init__()
        configure_logging()

        self.cfg = AppConfig.load()
        self.runtime = AsyncRuntime()
        self.cancel_event = threading.Event()

        self.session_openai_key = os.getenv("OPENAI_API_KEY", "")
        self.session_weaviate_url = os.getenv("WEAVIATE_URL", "")
        self.session_weaviate_key = os.getenv("WEAVIATE_API_KEY", "")

        self.store = SQLiteStore(Path(self.cfg.database_path).expanduser())
        self.quantum = QuantumFeatureEngine()
        self.legacy_manager = LegacyGGMLManager()
        self.gemma4 = Gemma4LiteRTManager()
        self.dataset_simulator = SafeDatasetSimulator()
        self.dataset_importer = SafeTabularDatasetImporter()
        self.cqse_encoder = ChromaticQuantumSurfaceEncoder()
        self.temporal_hyperfield = TemporalChromaticHyperfieldEngine(self.cqse_encoder)
        self.last_hyperfield: Optional[dict[str, Any]] = None
        self.hyperfield_replay_index: int = 0
        self.hyperfield_replay_job = None
        self.hyperfield_playing = False

        self.embeddings = EmbeddingEnsemble(
            self.store,
            lambda: self.session_openai_key,
            lambda: self.cfg,
        )
        self.memory = WeaviateMemory(
            self.store,
            self.embeddings,
            lambda: self.cfg,
            lambda: self.session_weaviate_url,
            lambda: self.session_weaviate_key,
        )
        self.providers = ProviderOrchestrator(
            lambda: self.session_openai_key,
            lambda: self.cfg.reasoning_effort,
            self.legacy_manager,
        )

        self.last_rows: list[dict[str, Any]] = []
        self.last_analytics: Optional[dict[str, Any]] = None
        self.last_memory_hits: list[dict[str, Any]] = []
        self.last_provider_result: Optional[ProviderResult] = None
        self.last_parsed: Optional[ParsedProtocol] = None
        self.last_analysis_id: Optional[str] = None
        self.last_cqse_surface: Optional[dict[str, Any]] = None
        self.last_gemma_palette: Optional[dict[str, Any]] = None
        self.hyper_surface_state: dict[str, Any] = {}
        self._busy = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1500x940")
        self.minsize(1160, 760)

        self._build_ui()
        logging.getLogger().addHandler(GuiLogHandler(self._append_log))
        self.protocol("WM_DELETE_WINDOW", self._shutdown)

        self._submit(
            self._bootstrap(),
            on_success=lambda _: self.refresh_telemetry(),
            on_error=lambda exc: self._show_error("Bootstrap", exc),
        )
        self.after(1000, self._system_tick)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="QUANTUM NETWORK // DAYBREAK // CQSE // TEMPORAL HYPERFIELD // VECTOR MEMORY",
            font=ctk.CTkFont(size=23, weight="bold"),
        ).grid(row=0, column=0, padx=18, pady=13, sticky="w")

        self.system_label = ctk.CTkLabel(header, text="CPU — | RAM —")
        self.system_label.grid(row=0, column=1, padx=18, pady=13, sticky="e")

        status = ctk.CTkFrame(self)
        status.grid(row=1, column=0, padx=16, pady=(10, 0), sticky="ew")
        status.grid_columnconfigure(0, weight=1)
        self.progress = ctk.CTkProgressBar(status)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, padx=10, pady=(8, 3), sticky="ew")
        self.status_label = ctk.CTkLabel(status, text="Initializing...", anchor="w")
        self.status_label.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=2, column=0, padx=16, pady=12, sticky="nsew")
        for name in (
            "Observatory",
            "Daybreak AI",
            "Vector Memory",
            "Datasets / CQSE",
            "Temporal Hyperfield",
            "Protocol / Actions",
            "Auth / Models",
            "Settings",
            "Logs",
        ):
            self.tabs.add(name)

        self._build_observatory()
        self._build_ai()
        self._build_memory()
        self._build_dataset_cqse()
        self._build_temporal_hyperfield()
        self._build_protocol()
        self._build_auth()
        self._build_settings()
        self._build_logs()

    def _build_observatory(self):
        tab = self.tabs.tab("Observatory")
        tab.grid_columnconfigure((0, 1, 2, 3), weight=1)
        tab.grid_rowconfigure(2, weight=1)

        self.cards = {}
        for col, (key, title, unit) in enumerate([
            ("ping", "PING", "ms"),
            ("jitter", "JITTER", "ms"),
            ("download_speed", "DOWNLOAD", "Mbps"),
            ("upload_speed", "UPLOAD", "Mbps"),
        ]):
            card = MetricCard(tab, title, unit)
            card.grid(row=0, column=col, padx=6, pady=6, sticky="ew")
            self.cards[key] = card

        controls = ctk.CTkFrame(tab)
        controls.grid(row=1, column=0, columnspan=4, padx=6, pady=6, sticky="ew")
        controls.grid_columnconfigure(5, weight=1)

        ctk.CTkButton(
            controls, text="Refresh telemetry", command=self.refresh_telemetry
        ).grid(row=0, column=0, padx=8, pady=8)

        ctk.CTkButton(
            controls, text="Sync snapshot → Weaviate", command=self.sync_current_snapshot
        ).grid(row=0, column=1, padx=8, pady=8)

        ctk.CTkButton(
            controls, text="Export CSV", command=self.export_csv
        ).grid(row=0, column=2, padx=8, pady=8)

        self.sample_label = ctk.CTkLabel(controls, text="samples: —")
        self.sample_label.grid(row=0, column=3, padx=8, pady=8)
        self.anomaly_label = ctk.CTkLabel(controls, text="anomalies: —")
        self.anomaly_label.grid(row=0, column=4, padx=8, pady=8)
        self.db_label = ctk.CTkLabel(controls, text=self.cfg.database_path, anchor="e")
        self.db_label.grid(row=0, column=5, padx=8, pady=8, sticky="e")

        chart = ctk.CTkFrame(tab)
        chart.grid(row=2, column=0, columnspan=4, padx=6, pady=6, sticky="nsew")
        chart.grid_columnconfigure(0, weight=1)
        chart.grid_rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_latency = self.fig.add_subplot(211)
        self.ax_speed = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_ai(self):
        tab = self.tabs.tab("Daybreak AI")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)

        bar = ctk.CTkFrame(tab)
        bar.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        bar.grid_columnconfigure(5, weight=1)

        self.mode_menu = ctk.CTkOptionMenu(bar, values=ANALYSIS_MODES)
        self.mode_menu.set("Deep Network Analysis")
        self.mode_menu.grid(row=0, column=0, padx=8, pady=8)

        self.provider_menu = ctk.CTkOptionMenu(
            bar,
            values=["Auto", "Codex Daybreak", "API Daybreak", "API GPT-6 Sol", "Local Llama"],
        )
        self.provider_menu.set(self.cfg.provider_preference)
        self.provider_menu.grid(row=0, column=1, padx=8, pady=8)

        self.run_ai_btn = ctk.CTkButton(bar, text="Analyze", command=self.run_analysis)
        self.run_ai_btn.grid(row=0, column=2, padx=8, pady=8)

        ctk.CTkButton(bar, text="Export report", command=self.export_report).grid(
            row=0, column=3, padx=8, pady=8
        )

        self.cancel_btn = ctk.CTkButton(
            bar, text="Cancel", command=self.cancel_current, state="disabled"
        )
        self.cancel_btn.grid(row=0, column=4, padx=8, pady=8)

        self.provider_badge = ctk.CTkLabel(
            bar, text=f"preferred: {DAYBREAK_ALIAS}", anchor="e"
        )
        self.provider_badge.grid(row=0, column=5, padx=8, pady=8, sticky="e")

        self.ai_meta = ctk.CTkTextbox(tab, height=145, wrap="word")
        self.ai_meta.grid(row=1, column=0, padx=6, pady=6, sticky="ew")

        self.ai_output = ctk.CTkTextbox(tab, wrap="word")
        self.ai_output.grid(row=2, column=0, padx=6, pady=6, sticky="nsew")

    def _build_memory(self):
        tab = self.tabs.tab("Vector Memory")
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(3, weight=1)

        top = ctk.CTkFrame(tab)
        top.grid(row=0, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        top.grid_columnconfigure(5, weight=1)

        ctk.CTkButton(
            top, text=f"Setup {len(WEAVIATE_SCHEMAS)} collections", command=self.setup_weaviate
        ).grid(row=0, column=0, padx=8, pady=8)

        ctk.CTkButton(
            top, text="Weaviate status", command=self.refresh_weaviate_status
        ).grid(row=0, column=1, padx=8, pady=8)

        self.memory_query = ctk.CTkEntry(
            top,
            placeholder_text="semantic + BM25 + quantum memory search...",
        )
        self.memory_query.grid(row=0, column=2, columnspan=3, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(
            top, text="Search memory", command=self.search_memory
        ).grid(row=0, column=5, padx=8, pady=8)

        self.memory_status = ctk.CTkTextbox(tab, height=135, wrap="word")
        self.memory_status.grid(row=1, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(
            tab,
            text=(
                f"Named vectors: {VECTOR_OPENAI} ({EMBED_MODEL}, {EMBED_DIM}D)  •  "
                f"{VECTOR_FB} ({FB_EMBED_MODEL}, {FB_EMBED_DIM}D)  •  "
                f"{VECTOR_QUANTUM} (PennyLane, 16D)  •  "
                f"{VECTOR_CHROMA} (CQSE, 96D)  •  "
                f"{VECTOR_CHROMA_QUANTUM} (PennyLane CQSE, 32D)"
            ),
            anchor="w",
        ).grid(row=2, column=0, columnspan=2, padx=8, pady=4, sticky="ew")

        left = ctk.CTkFrame(tab)
        right = ctk.CTkFrame(tab)
        left.grid(row=3, column=0, padx=(6, 3), pady=6, sticky="nsew")
        right.grid(row=3, column=1, padx=(3, 6), pady=6, sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Schema surface").grid(
            row=0, column=0, padx=8, pady=6, sticky="w"
        )
        self.schema_box = ctk.CTkTextbox(left, wrap="word")
        self.schema_box.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")
        self.schema_box.insert("1.0", json.dumps(WEAVIATE_SCHEMAS, indent=2))
        self.schema_box.configure(state="disabled")

        ctk.CTkLabel(right, text="Retrieved memories").grid(
            row=0, column=0, padx=8, pady=6, sticky="w"
        )
        self.memory_results = ctk.CTkTextbox(right, wrap="word")
        self.memory_results.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

    def _build_dataset_cqse(self):
        """
        Hyper Surface workbench.

        The complete setup/dependency matrix lives above the vector visualizer so
        an operator can inspect exactly which semantic, chromatic, quantum, memory,
        and dataset layers contributed to the current surface.
        """
        tab = self.tabs.tab("Datasets / CQSE")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # ------------------------------------------------------------------
        # TOP: complete hyper-surface setup matrix
        # ------------------------------------------------------------------
        setup = ctk.CTkFrame(tab)
        setup.grid(row=0, column=0, padx=6, pady=(6, 3), sticky="ew")
        setup.grid_columnconfigure(8, weight=1)

        ctk.CTkLabel(
            setup,
            text="HYPER SURFACE SETUP MATRIX",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=9, padx=10, pady=(9, 3), sticky="ew")

        self.hyper_status_labels: dict[str, ctk.CTkLabel] = {}
        status_defs = [
            ("sqlite", "SQLite/WAL"),
            ("weaviate", "Weaviate"),
            ("openai", "OpenAI 3072D"),
            ("contriever", "Contriever 768D"),
            ("gemma", "Gemma 4 E2B"),
            ("pennylane", "PennyLane"),
            ("datasets", "Dataset Sandbox"),
            ("cqse", "CQSE 96D+32D"),
        ]
        for col, (key, title) in enumerate(status_defs):
            frame = ctk.CTkFrame(setup)
            frame.grid(row=1, column=col, padx=4, pady=5, sticky="ew")
            frame.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                frame,
                text=title,
                font=ctk.CTkFont(size=11, weight="bold"),
            ).grid(row=0, column=0, padx=7, pady=(7, 1), sticky="ew")
            lbl = ctk.CTkLabel(frame, text="UNKNOWN", font=ctk.CTkFont(size=11))
            lbl.grid(row=1, column=0, padx=7, pady=(0, 7), sticky="ew")
            self.hyper_status_labels[key] = lbl

        actions = ctk.CTkFrame(setup)
        actions.grid(row=2, column=0, columnspan=9, padx=6, pady=5, sticky="ew")
        actions.grid_columnconfigure(10, weight=1)

        ctk.CTkButton(
            actions,
            text="Initialize Hyper Surface",
            command=self.initialize_hyper_surface,
            width=165,
        ).grid(row=0, column=0, padx=5, pady=6)

        ctk.CTkButton(
            actions,
            text=f"Setup {len(WEAVIATE_SCHEMAS)} Collections",
            command=self.setup_weaviate,
        ).grid(row=0, column=1, padx=5, pady=6)

        ctk.CTkButton(
            actions,
            text="Seed Safe Top 5",
            command=self.seed_top5_datasets,
        ).grid(row=0, column=2, padx=5, pady=6)

        ctk.CTkButton(
            actions,
            text="Verify Gemma",
            command=self.verify_gemma4_gui,
        ).grid(row=0, column=3, padx=5, pady=6)

        ctk.CTkButton(
            actions,
            text="Download Gemma",
            command=self.download_gemma4,
        ).grid(row=0, column=4, padx=5, pady=6)

        ctk.CTkButton(
            actions,
            text="Refresh Surface State",
            command=self.refresh_hyper_surface_state,
        ).grid(row=0, column=5, padx=5, pady=6)

        self.cqse_semantic_source = ctk.CTkOptionMenu(
            actions,
            values=["Auto Ensemble", "OpenAI Preferred", "Contriever Preferred", "Hashed Offline"],
            width=155,
        )
        self.cqse_semantic_source.set("Auto Ensemble")
        self.cqse_semantic_source.grid(row=0, column=6, padx=5, pady=6)

        self.cqse_palette_mode = ctk.CTkOptionMenu(
            actions,
            values=["Deterministic", "Gemma Conditioned"],
            width=150,
        )
        self.cqse_palette_mode.set("Deterministic")
        self.cqse_palette_mode.grid(row=0, column=7, padx=5, pady=6)

        self.cqse_auto_store = ctk.CTkSwitch(actions, text="Auto-store surface")
        self.cqse_auto_store.select()
        self.cqse_auto_store.grid(row=0, column=8, padx=7, pady=6)

        self.cqse_auto_recall = ctk.CTkSwitch(actions, text="Auto-neighborhood")
        self.cqse_auto_recall.select()
        self.cqse_auto_recall.grid(row=0, column=9, padx=7, pady=6)

        self.hyper_setup_summary = ctk.CTkTextbox(setup, height=94, wrap="word")
        self.hyper_setup_summary.grid(
            row=3, column=0, columnspan=9, padx=8, pady=(4, 8), sticky="ew"
        )
        self.hyper_setup_summary.insert(
            "1.0",
            (
                "Pipeline: network telemetry → robust statistics → semantic ensemble "
                "(OpenAI/Contriever/offline hash) → RGB/XYZ/OKLab/OKLCH projection → "
                "distribution entropy → PennyLane 5-qubit CQSE → 96-D chromatic vector + "
                "32-D quantum probability vector → Weaviate named-vector neighborhoods → "
                "weighted RRF retrieval → Daybreak analysis.\n"
                "CQSE is explicitly experimental and is treated as an additional retrieval "
                "surface, not a validated replacement for standard semantic embeddings."
            ),
        )

        # ------------------------------------------------------------------
        # MIDDLE: generation controls + live scalar metrics
        # ------------------------------------------------------------------
        command = ctk.CTkFrame(tab)
        command.grid(row=1, column=0, padx=6, pady=3, sticky="ew")
        command.grid_columnconfigure(8, weight=1)

        ctk.CTkButton(
            command,
            text="Build Current CQSE",
            command=self.build_current_cqse,
        ).grid(row=0, column=0, padx=5, pady=7)

        ctk.CTkButton(
            command,
            text="Build Gemma-CQSE",
            command=self.build_gemma_cqse,
        ).grid(row=0, column=1, padx=5, pady=7)

        ctk.CTkButton(
            command,
            text="Nearest Surface Memories",
            command=self.search_current_cqse_neighbors,
        ).grid(row=0, column=2, padx=5, pady=7)

        ctk.CTkButton(
            command,
            text="Simulate Linked Malware Set",
            command=self.seed_linked_malware_dataset,
        ).grid(row=0, column=3, padx=5, pady=7)

        ctk.CTkButton(
            command,
            text="Real Kaggle Import",
            command=self.real_kaggle_linked_import,
        ).grid(row=0, column=4, padx=5, pady=7)

        self.cqse_query_entry = ctk.CTkEntry(
            command,
            placeholder_text="optional semantic probe for a custom CQSE surface...",
        )
        self.cqse_query_entry.grid(row=0, column=5, columnspan=3, padx=6, pady=7, sticky="ew")

        ctk.CTkButton(
            command,
            text="Probe Text",
            command=self.build_text_probe_cqse,
        ).grid(row=0, column=8, padx=5, pady=7)

        metric_strip = ctk.CTkFrame(tab)
        metric_strip.grid(row=2, column=0, padx=6, pady=3, sticky="ew")
        metric_strip.grid_columnconfigure((0, 1, 2, 3, 4, 5), weight=1)

        metric_defs = [
            ("base", "Semantic Base"),
            ("entropy", "Embedding Entropy"),
            ("chroma_norm", "CQSE-96 L2"),
            ("quantum_entropy", "CQSE-Q Entropy"),
            ("palette", "Palette Mode"),
            ("neighbors", "Memory Neighbors"),
        ]
        self.cqse_metric_labels = {}
        for col, (key, title) in enumerate(metric_defs):
            card = ctk.CTkFrame(metric_strip)
            card.grid(row=0, column=col, padx=4, pady=5, sticky="ew")
            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(size=10, weight="bold"),
            ).grid(row=0, column=0, padx=8, pady=(7, 1))
            value = ctk.CTkLabel(
                card,
                text="—",
                font=ctk.CTkFont(size=15, weight="bold"),
            )
            value.grid(row=1, column=0, padx=8, pady=(0, 7))
            self.cqse_metric_labels[key] = value

        # ------------------------------------------------------------------
        # BOTTOM: actual colorspace/vector workbench
        # ------------------------------------------------------------------
        workbench = ctk.CTkTabview(tab)
        workbench.grid(row=3, column=0, padx=6, pady=(3, 6), sticky="nsew")
        for name in (
            "Surface Visualizer",
            "Palette / Coordinates",
            "Quantum State",
            "Retrieval Neighborhood",
            "Dataset Corpus",
            "Raw CQSE",
        ):
            workbench.add(name)

        # Surface Visualizer: one figure with four coordinated views.
        visual = workbench.tab("Surface Visualizer")
        visual.grid_columnconfigure(0, weight=1)
        visual.grid_rowconfigure(0, weight=1)

        self.cqse_fig = Figure(figsize=(11.5, 6.3), dpi=100)
        self.cqse_ax_palette = self.cqse_fig.add_subplot(221)
        self.cqse_ax_lab = self.cqse_fig.add_subplot(222)
        self.cqse_ax_vector = self.cqse_fig.add_subplot(223)
        self.cqse_ax_quantum = self.cqse_fig.add_subplot(224)
        self.cqse_fig.tight_layout(pad=2.0)

        self.cqse_canvas = FigureCanvasTkAgg(self.cqse_fig, master=visual)
        self.cqse_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Palette and coordinates.
        palette_tab = workbench.tab("Palette / Coordinates")
        palette_tab.grid_columnconfigure((0, 1), weight=1)
        palette_tab.grid_rowconfigure(1, weight=1)

        self.cqse_palette_frame = ctk.CTkFrame(palette_tab)
        self.cqse_palette_frame.grid(
            row=0, column=0, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        self.cqse_palette_frame.grid_columnconfigure(tuple(range(8)), weight=1)
        self.cqse_palette_swatches: list[ctk.CTkFrame] = []
        self.cqse_palette_labels: list[ctk.CTkLabel] = []
        for i in range(8):
            sw = ctk.CTkFrame(self.cqse_palette_frame, height=64)
            sw.grid(row=0, column=i, padx=3, pady=5, sticky="ew")
            sw.grid_propagate(False)
            label = ctk.CTkLabel(sw, text=f"P{i+1}\n—")
            label.place(relx=0.5, rely=0.5, anchor="center")
            self.cqse_palette_swatches.append(sw)
            self.cqse_palette_labels.append(label)

        self.cqse_coordinate_box = ctk.CTkTextbox(palette_tab, wrap="word")
        self.cqse_coordinate_box.grid(
            row=1, column=0, padx=(8, 4), pady=8, sticky="nsew"
        )
        self.cqse_axis_box = ctk.CTkTextbox(palette_tab, wrap="word")
        self.cqse_axis_box.grid(
            row=1, column=1, padx=(4, 8), pady=8, sticky="nsew"
        )

        # Quantum state surface.
        quantum_tab = workbench.tab("Quantum State")
        quantum_tab.grid_columnconfigure((0, 1), weight=1)
        quantum_tab.grid_rowconfigure(1, weight=1)

        self.cqse_quantum_summary = ctk.CTkTextbox(
            quantum_tab, height=130, wrap="word"
        )
        self.cqse_quantum_summary.grid(
            row=0, column=0, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        self.cqse_quantum_probability_box = ctk.CTkTextbox(
            quantum_tab, wrap="none"
        )
        self.cqse_quantum_probability_box.grid(
            row=1, column=0, padx=(8, 4), pady=8, sticky="nsew"
        )
        self.cqse_quantum_interpretation = ctk.CTkTextbox(
            quantum_tab, wrap="word"
        )
        self.cqse_quantum_interpretation.grid(
            row=1, column=1, padx=(4, 8), pady=8, sticky="nsew"
        )

        # Retrieval neighborhood.
        neighborhood = workbench.tab("Retrieval Neighborhood")
        neighborhood.grid_columnconfigure(0, weight=1)
        neighborhood.grid_rowconfigure(1, weight=1)
        self.cqse_neighbor_meta = ctk.CTkLabel(
            neighborhood,
            text="No CQSE neighborhood retrieved yet.",
            anchor="w",
        )
        self.cqse_neighbor_meta.grid(row=0, column=0, padx=8, pady=6, sticky="ew")
        self.cqse_neighbor_tree = ttk.Treeview(
            neighborhood,
            columns=("rank", "collection", "score", "sources", "summary"),
            show="headings",
        )
        for col, width in (
            ("rank", 55),
            ("collection", 210),
            ("score", 95),
            ("sources", 220),
            ("summary", 620),
        ):
            self.cqse_neighbor_tree.heading(col, text=col.upper())
            self.cqse_neighbor_tree.column(col, width=width, anchor="w")
        self.cqse_neighbor_tree.grid(
            row=1, column=0, padx=8, pady=8, sticky="nsew"
        )

        # Dataset corpus.
        dataset_tab = workbench.tab("Dataset Corpus")
        dataset_tab.grid_columnconfigure((0, 1), weight=1)
        dataset_tab.grid_rowconfigure(1, weight=1)

        self.dataset_manifest_box = ctk.CTkTextbox(
            dataset_tab, height=170, wrap="word"
        )
        self.dataset_manifest_box.grid(
            row=0, column=0, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        self.dataset_manifest_box.insert(
            "1.0",
            json.dumps(
                [
                    {
                        "rank": m.rank,
                        "dataset_id": m.dataset_id,
                        "name": m.name,
                        "category": m.category,
                        "safe_default": m.safe_default,
                        "source_url": m.source_url,
                        "kaggle_slug": m.kaggle_slug,
                    }
                    for m in CYBER_DATASETS
                ],
                indent=2,
            ),
        )
        self.dataset_manifest_box.configure(state="disabled")

        self.dataset_box = ctk.CTkTextbox(dataset_tab, wrap="word")
        self.dataset_box.grid(
            row=1, column=0, padx=(8, 4), pady=8, sticky="nsew"
        )
        self.dataset_box.insert(
            "1.0",
            (
                "Default ingestion is bounded simulation. Real import is an explicit action "
                "and only accepts sanitized CSV/TSV/JSONL/Parquet.\n"
            ),
        )

        self.dataset_security_box = ctk.CTkTextbox(dataset_tab, wrap="word")
        self.dataset_security_box.grid(
            row=1, column=1, padx=(4, 8), pady=8, sticky="nsew"
        )
        self.dataset_security_box.insert(
            "1.0",
            (
                "Rejected: pickle/joblib/notebooks/scripts/binaries/PCAP/NumPy object "
                "archives/model files. ZIP traversal and expansion are checked. "
                "Rows and member sizes are capped. Secret-like columns are redacted."
            ),
        )

        # Raw CQSE.
        raw = workbench.tab("Raw CQSE")
        raw.grid_columnconfigure(0, weight=1)
        raw.grid_rowconfigure(0, weight=1)
        self.cqse_box = ctk.CTkTextbox(raw, wrap="word")
        self.cqse_box.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.cqse_box.insert(
            "1.0",
            (
                "No surface generated yet.\n\n"
                "CQSE is an experimental retrieval feature transform. "
                "Generate Current CQSE or Gemma-CQSE to populate this surface."
            ),
        )

    async def _inspect_hyper_surface_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}

        # SQLite/WAL
        try:
            await self.store.initialize()
            async with self.store.connection() as db:
                cur = await db.execute("PRAGMA journal_mode")
                row = await cur.fetchone()
                journal = str(row[0]).lower() if row else "unknown"
            state["sqlite"] = {
                "ok": journal == "wal",
                "detail": f"WAL={journal}",
            }
        except Exception as exc:
            state["sqlite"] = {"ok": False, "detail": str(exc)}

        # Weaviate
        try:
            w = await self.memory.status()
            existing = sum(
                1 for v in w.get("collections", {}).values() if v.get("exists")
            )
            state["weaviate"] = {
                "ok": bool(w.get("ready")) and existing == len(WEAVIATE_SCHEMAS),
                "detail": f"{existing}/{len(WEAVIATE_SCHEMAS)} collections",
                "raw": w,
            }
        except Exception as exc:
            state["weaviate"] = {"ok": False, "detail": str(exc)}

        # OpenAI semantic vector provider
        state["openai"] = {
            "ok": self.embeddings.openai.available() and self.cfg.use_openai_embeddings,
            "detail": (
                f"{EMBED_MODEL} / {EMBED_DIM}D"
                if self.embeddings.openai.available()
                else "no session API key"
            ),
        }

        # Contriever
        fb_ok = self.embeddings.facebook.available() and self.cfg.use_fb_embeddings
        state["contriever"] = {
            "ok": fb_ok,
            "detail": f"{FB_EMBED_MODEL} / {FB_EMBED_DIM}D" if fb_ok else "runtime unavailable/disabled",
        }

        # Gemma
        try:
            verify = await self.gemma4.verify()
            runtime_ok = self.gemma4.runtime_available()
            state["gemma"] = {
                "ok": bool(verify.get("ok")) and runtime_ok,
                "detail": (
                    f"verified + LiteRT-LM"
                    if verify.get("ok") and runtime_ok
                    else f"model={'verified' if verify.get('ok') else 'missing/unverified'}, "
                         f"runtime={'yes' if runtime_ok else 'no'}"
                ),
                "raw": verify,
            }
        except Exception as exc:
            state["gemma"] = {"ok": False, "detail": str(exc)}

        # PennyLane runtime and deterministic dimension check.
        try:
            dev = qml.device("default.qubit", wires=5)

            @qml.qnode(dev)
            def _probe():
                for wire in range(5):
                    qml.Hadamard(wires=wire)
                return qml.probs(wires=range(5))

            probs = await asyncio.to_thread(_probe)
            state["pennylane"] = {
                "ok": len(probs) == 32,
                "detail": f"default.qubit / {len(probs)}D probability surface",
            }
        except Exception as exc:
            state["pennylane"] = {"ok": False, "detail": str(exc)}

        # Dataset sandbox.
        try:
            DATASET_SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
            writable = os.access(DATASET_SANDBOX_DIR, os.W_OK)
            state["datasets"] = {
                "ok": writable,
                "detail": f"{len(CYBER_DATASETS)} manifests / sandbox {'writable' if writable else 'read-only'}",
            }
        except Exception as exc:
            state["datasets"] = {"ok": False, "detail": str(exc)}

        # CQSE deterministic self-test.
        try:
            probe = await asyncio.to_thread(
                self.cqse_encoder.encode,
                "CQSE self-test: defensive network anomaly surface",
                {},
                None,
                None,
            )
            state["cqse"] = {
                "ok": (
                    len(probe["chromatic_surface_96"]) == 96
                    and len(probe["chromatic_quantum_32"]) == 32
                ),
                "detail": (
                    f"{len(probe['chromatic_surface_96'])}D + "
                    f"{len(probe['chromatic_quantum_32'])}D"
                ),
            }
        except Exception as exc:
            state["cqse"] = {"ok": False, "detail": str(exc)}

        return state

    def _render_hyper_surface_state(self, state: dict[str, Any]):
        self.hyper_surface_state = state
        ready = 0
        for key, label in self.hyper_status_labels.items():
            item = state.get(key, {})
            ok = bool(item.get("ok"))
            ready += int(ok)
            label.configure(
                text=("READY" if ok else "OFFLINE") + "\n" + str(item.get("detail", ""))[:45]
            )

        self.hyper_setup_summary.delete("1.0", "end")
        self.hyper_setup_summary.insert(
            "1.0",
            (
                f"Hyper Surface readiness: {ready}/{len(self.hyper_status_labels)} layers active.\n\n"
                + json.dumps(state, indent=2, default=str)
            ),
        )

    def refresh_hyper_surface_state(self):
        self.set_status(0.05, "Inspecting complete CQSE dependency graph...")
        self._submit(
            self._inspect_hyper_surface_state(),
            on_success=lambda state: (
                self._render_hyper_surface_state(state),
                self.set_status(1.0, "Hyper Surface dependency state refreshed."),
            ),
            on_error=lambda e: self._show_error("Hyper Surface state", e),
        )

    def initialize_hyper_surface(self):
        """
        One-click initialization:
        - initialize SQLite/WAL
        - create all Weaviate schemas when available
        - verify Gemma (never auto-downloads a multi-GB model)
        - run PennyLane/CQSE self-tests
        - inspect embedding providers
        """
        self.cancel_event.clear()
        self.set_status(0.02, "Initializing Hyper Surface...")

        async def work():
            steps = []

            await self.store.initialize()
            steps.append("SQLite/WAL initialized")
            self.set_status_threadsafe(0.18, "SQLite/WAL ready...")

            try:
                result = await self.memory.setup_collections()
                steps.append(f"Weaviate: {result}")
            except Exception as exc:
                steps.append(f"Weaviate unavailable: {exc}")
            self.set_status_threadsafe(0.42, "Vector-memory layer inspected...")

            gemma = await self.gemma4.verify()
            steps.append(
                "Gemma: verified" if gemma.get("ok") else "Gemma: not verified (download remains explicit)"
            )
            self.set_status_threadsafe(0.58, "Gemma integrity state inspected...")

            probe = await asyncio.to_thread(
                self.cqse_encoder.encode,
                "Initialize Hyper Surface defensive cyber probe",
                {},
                self.last_analytics["quantum"]["probabilities"] if self.last_analytics else None,
                None,
            )
            steps.append(
                f"CQSE self-test: {len(probe['chromatic_surface_96'])}D + "
                f"{len(probe['chromatic_quantum_32'])}D"
            )
            self.set_status_threadsafe(0.78, "PennyLane CQSE self-test complete...")

            state = await self._inspect_hyper_surface_state()
            return {"steps": steps, "state": state}

        def success(result):
            self._render_hyper_surface_state(result["state"])
            self.set_status(1.0, "Hyper Surface initialized.")
            self._append_dataset_log(
                "Initialize Hyper Surface:\n" + json.dumps(result["steps"], indent=2)
            )

        self._submit(
            work(),
            on_success=success,
            on_error=lambda e: self._show_error("Initialize Hyper Surface", e),
        )

    def verify_gemma4_gui(self):
        self.set_status(0.10, "Verifying Gemma 4 E2B SHA-256...")
        self._submit(
            self.gemma4.verify(),
            on_success=lambda r: (
                self._append_dataset_log("Gemma verification:\n" + json.dumps(r, indent=2)),
                self.set_status(1.0 if r.get("ok") else 0.0, "Gemma verification complete."),
                self.refresh_hyper_surface_state(),
            ),
            on_error=lambda e: self._show_error("Gemma verification", e),
        )

    @staticmethod
    def _hex_from_rgb(rgb: tuple[float, float, float]) -> str:
        values = [
            max(0, min(255, int(round(float(c) * 255.0))))
            for c in rgb
        ]
        return "#{:02X}{:02X}{:02X}".format(*values)

    @staticmethod
    def _quantum_entropy(probs: list[float]) -> float:
        arr = np.asarray(probs, dtype=np.float64)
        return float(-np.sum(arr * np.log2(arr + 1e-12)))

    def _render_cqse_surface(
        self,
        surface: dict[str, Any],
        palette_meta: Optional[dict[str, Any]] = None,
        neighbors: Optional[list[dict[str, Any]]] = None,
    ):
        self.last_cqse_surface = surface
        if palette_meta is not None:
            self.last_gemma_palette = palette_meta

        rgb = [tuple(float(x) for x in row) for row in surface.get("rgb_anchors", [])]
        labs = [tuple(float(x) for x in row) for row in surface.get("oklab_anchors", [])]
        lchs = [tuple(float(x) for x in row) for row in surface.get("oklch_anchors", [])]
        vec96 = [float(x) for x in surface.get("chromatic_surface_96", [])]
        q32 = [float(x) for x in surface.get("chromatic_quantum_32", [])]

        palette = list(surface.get("palette") or [])
        if not palette:
            palette = [self._hex_from_rgb(x) for x in rgb]
        while len(palette) < 8:
            palette.append("#30343B")

        # Swatches.
        for i, (frame, label) in enumerate(
            zip(self.cqse_palette_swatches, self.cqse_palette_labels)
        ):
            hx = palette[i] if i < len(palette) else "#30343B"
            try:
                frame.configure(fg_color=hx)
            except Exception:
                pass
            label.configure(text=f"P{i+1}\n{hx}")

        # Scalar metrics.
        self.cqse_metric_labels["base"].configure(
            text=str(surface.get("base_source", "—"))[:22]
        )
        dist = surface.get("distribution", {})
        self.cqse_metric_labels["entropy"].configure(
            text=f"{float(dist.get('shannon_abs_bits', 0.0)):.3f} bits"
        )
        self.cqse_metric_labels["chroma_norm"].configure(
            text=f"{float(np.linalg.norm(np.asarray(vec96, dtype=float))):.4f}"
            if vec96 else "—"
        )
        self.cqse_metric_labels["quantum_entropy"].configure(
            text=f"{self._quantum_entropy(q32):.3f} bits" if q32 else "—"
        )
        self.cqse_metric_labels["palette"].configure(
            text="Gemma" if palette_meta else "Deterministic"
        )
        self.cqse_metric_labels["neighbors"].configure(
            text=str(len(neighbors or []))
        )

        # Coordinated visualizer.
        self.cqse_ax_palette.clear()
        self.cqse_ax_lab.clear()
        self.cqse_ax_vector.clear()
        self.cqse_ax_quantum.clear()

        # Palette bars.
        if palette:
            self.cqse_ax_palette.barh(
                list(range(len(palette))),
                [1.0] * len(palette),
                color=palette,
            )
            self.cqse_ax_palette.set_yticks(range(len(palette)))
            self.cqse_ax_palette.set_yticklabels([f"P{i+1}" for i in range(len(palette))])
            self.cqse_ax_palette.set_xlim(0, 1)
            self.cqse_ax_palette.set_title("Semantic palette coordinates")
            self.cqse_ax_palette.set_xticks([])

        # OKLab a/b plane.
        if labs:
            a_vals = [x[1] for x in labs]
            b_vals = [x[2] for x in labs]
            point_colors = [
                self._hex_from_rgb(rgb[i]) if i < len(rgb) else palette[i % len(palette)]
                for i in range(len(labs))
            ]
            self.cqse_ax_lab.scatter(a_vals, b_vals, c=point_colors, s=120)
            for i, (a, b) in enumerate(zip(a_vals, b_vals), start=1):
                self.cqse_ax_lab.annotate(f"P{i}", (a, b))
            self.cqse_ax_lab.axhline(0, linewidth=0.6, alpha=0.4)
            self.cqse_ax_lab.axvline(0, linewidth=0.6, alpha=0.4)
            self.cqse_ax_lab.set_xlabel("OKLab a")
            self.cqse_ax_lab.set_ylabel("OKLab b")
            self.cqse_ax_lab.set_title("OKLab chromatic manifold")

        # 96-D surface as 8x12 heatmap.
        if len(vec96) == 96:
            matrix = np.asarray(vec96, dtype=np.float64).reshape(8, 12)
            self.cqse_ax_vector.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                cmap="turbo",
                vmin=-max(abs(matrix.min()), abs(matrix.max()), 1e-6),
                vmax=max(abs(matrix.min()), abs(matrix.max()), 1e-6),
            )
            self.cqse_ax_vector.set_title("CQSE 96-D chromatic surface")
            self.cqse_ax_vector.set_xlabel("surface coordinate")
            self.cqse_ax_vector.set_ylabel("semantic band")

        # 32-state probability vector.
        if len(q32) == 32:
            bar_colors = []
            for i, probability in enumerate(q32):
                # Encode basis-state probability into HSV, then render as a true colorized quantum surface.
                hue = (i / 32.0 + 0.08 * probability) % 1.0
                sat = min(1.0, 0.45 + probability * 5.0)
                val = min(1.0, 0.55 + probability * 4.5)
                bar_colors.append(colorsys.hsv_to_rgb(hue, sat, val))
            self.cqse_ax_quantum.bar(
                list(range(32)),
                q32,
                color=bar_colors,
                width=0.88,
            )
            self.cqse_ax_quantum.set_title("5-qubit CQSE probability surface")
            self.cqse_ax_quantum.set_xlabel("computational basis state")
            self.cqse_ax_quantum.set_ylabel("probability")

        self.cqse_fig.tight_layout(pad=1.7)
        self.cqse_canvas.draw_idle()

        # Coordinate / axes text.
        self.cqse_coordinate_box.delete("1.0", "end")
        self.cqse_coordinate_box.insert(
            "1.0",
            json.dumps(
                {
                    "rgb": rgb,
                    "oklab": labs,
                    "oklch": lchs,
                    "distribution": dist,
                },
                indent=2,
                default=str,
            ),
        )

        self.cqse_axis_box.delete("1.0", "end")
        self.cqse_axis_box.insert(
            "1.0",
            json.dumps(
                {
                    "gemma_palette": palette_meta,
                    "experimental_notice": surface.get("experimental_notice"),
                    "base_source": surface.get("base_source"),
                },
                indent=2,
                default=str,
            ),
        )

        # Quantum details.
        self.cqse_quantum_summary.delete("1.0", "end")
        self.cqse_quantum_summary.insert(
            "1.0",
            json.dumps(
                {
                    "states": len(q32),
                    "probability_sum": float(sum(q32)) if q32 else 0.0,
                    "entropy_bits": self._quantum_entropy(q32) if q32 else None,
                    "max_probability": max(q32) if q32 else None,
                    "argmax_state": int(np.argmax(q32)) if q32 else None,
                    "telemetry_quantum_present": bool(
                        self.last_analytics
                        and self.last_analytics.get("quantum", {}).get("probabilities")
                    ),
                },
                indent=2,
            ),
        )

        self.cqse_quantum_probability_box.delete("1.0", "end")
        self.cqse_quantum_probability_box.insert(
            "1.0",
            "\n".join(
                f"|{i:05b}>  {p:.10f}"
                for i, p in enumerate(q32)
            ),
        )

        interpretation = (
            "The 32-dimensional vector is the probability distribution of a five-qubit "
            "feature circuit. Similarity in this space means similar CQSE circuit output, "
            "not evidence of quantum speedup or causal equivalence. Entropy near 5 bits is "
            "diffuse; lower entropy indicates concentration into fewer basis states. "
            "The surface is deliberately ranked independently from semantic embeddings."
        )
        self.cqse_quantum_interpretation.delete("1.0", "end")
        self.cqse_quantum_interpretation.insert("1.0", interpretation)

        self.cqse_box.delete("1.0", "end")
        self.cqse_box.insert(
            "1.0",
            json.dumps(
                {
                    "surface": surface,
                    "palette_meta": palette_meta,
                    "neighbors": neighbors or [],
                },
                indent=2,
                default=str,
            ),
        )

        if neighbors is not None:
            self._render_cqse_neighbors(neighbors)

    def _render_cqse_neighbors(self, neighbors: list[dict[str, Any]]):
        for item in self.cqse_neighbor_tree.get_children():
            self.cqse_neighbor_tree.delete(item)

        for rank, hit in enumerate(neighbors, start=1):
            props = hit.get("properties", {})
            summary = (
                props.get("summary")
                or props.get("final_text")
                or props.get("title")
                or props.get("description")
                or safe_json(props)
            )
            source_names = ",".join(
                str(x.get("source")) for x in hit.get("sources", [])
            )
            self.cqse_neighbor_tree.insert(
                "",
                "end",
                values=(
                    rank,
                    hit.get("collection"),
                    f"{float(hit.get('rrf_score', 0.0)):.6f}",
                    source_names,
                    str(summary)[:240],
                ),
            )

        self.cqse_neighbor_meta.configure(
            text=f"{len(neighbors)} fused neighborhood memories"
        )
        self.cqse_metric_labels["neighbors"].configure(text=str(len(neighbors)))

    async def _generate_cqse(
        self,
        text: str,
        gemma_conditioned: bool,
        telemetry_quantum: Optional[list[float]],
    ) -> dict[str, Any]:
        semantic, errors = await self.embeddings.embed_all(text)

        # Operator-selected semantic base.
        preference = self.cqse_semantic_source.get()
        selected = dict(semantic)
        if preference == "OpenAI Preferred" and VECTOR_OPENAI in semantic:
            selected = {VECTOR_OPENAI: semantic[VECTOR_OPENAI]}
        elif preference == "Contriever Preferred" and VECTOR_FB in semantic:
            selected = {VECTOR_FB: semantic[VECTOR_FB]}
        elif preference == "Hashed Offline":
            selected = {}

        palette = None
        if gemma_conditioned:
            palette = await self.gemma4.palette_prompt(
                text,
                self.cancel_event,
                lambda msg: self.set_status_threadsafe(0.42, msg),
            )

        surface = await asyncio.to_thread(
            self.cqse_encoder.encode,
            text,
            selected,
            telemetry_quantum,
            palette,
        )

        neighbors = []
        if self.cqse_auto_recall.get():
            try:
                recall = await self.memory.search(
                    text,
                    telemetry_quantum,
                    self.cfg.recall_limit,
                )
                neighbors = recall["results"]
                errors.extend(recall.get("embedding_errors", []))
            except Exception as exc:
                errors.append(f"Memory recall: {exc}")

        if self.cqse_auto_store.get():
            try:
                surface_id = str(uuid.uuid4())
                props = {
                    "surface_id": surface_id,
                    "created_at": utc_now(),
                    "source_kind": "gemma-cqse" if palette else "deterministic-cqse",
                    "source_id": stable_hash(text)[:32],
                    "summary": text[:900],
                    "palette_json": safe_json(palette or surface.get("palette", [])),
                    "colorspace_json": safe_json({
                        "rgb": surface["rgb_anchors"],
                        "oklab": surface["oklab_anchors"],
                        "oklch": surface["oklch_anchors"],
                    }),
                    "entropy_json": safe_json(surface["distribution"]),
                    "quantum_json": safe_json({
                        "chromatic_quantum_32": surface["chromatic_quantum_32"],
                        "telemetry_quantum_16": telemetry_quantum,
                    }),
                    "experimental_notice": surface["experimental_notice"],
                    "tags": [
                        "cqse",
                        "gemma-conditioned" if palette else "deterministic",
                        "experimental",
                    ],
                }
                await self.memory.insert(
                    "QNChromaticSurfaceMemoryV1",
                    props,
                    "cqse",
                    surface_id,
                    telemetry_quantum,
                )
            except Exception as exc:
                errors.append(f"Surface store: {exc}")

        return {
            "text": text,
            "surface": surface,
            "palette": palette,
            "neighbors": neighbors,
            "errors": errors,
        }

    def build_text_probe_cqse(self):
        text = self.cqse_query_entry.get().strip()
        if not text:
            return
        self.cancel_event.clear()
        gemma_conditioned = self.cqse_palette_mode.get() == "Gemma Conditioned"
        telemetry_quantum = (
            self.last_analytics["quantum"]["probabilities"]
            if self.last_analytics
            else None
        )
        self.set_status(0.05, "Generating custom CQSE text probe...")
        self._submit(
            self._generate_cqse(text, gemma_conditioned, telemetry_quantum),
            on_success=lambda result: (
                self._render_cqse_surface(
                    result["surface"],
                    result["palette"],
                    result["neighbors"],
                ),
                self._append_dataset_log(
                    "CQSE text probe:\n" + json.dumps(
                        {"errors": result["errors"]},
                        indent=2,
                    )
                ),
                self.set_status(1.0, "Custom CQSE probe complete."),
            ),
            on_error=lambda e: self._show_error("CQSE text probe", e),
        )

    def search_current_cqse_neighbors(self):
        if not self.last_cqse_surface:
            messagebox.showinfo("No CQSE surface", "Generate a CQSE surface first.")
            return

        query = self.cqse_query_entry.get().strip()
        if not query:
            query = (
                "Current chromatic quantum cyber surface; "
                + str(self.last_cqse_surface.get("base_source", "semantic"))
            )
        qvec = (
            self.last_analytics["quantum"]["probabilities"]
            if self.last_analytics
            else None
        )
        self.set_status(0.10, "Searching fused CQSE neighborhood...")
        self._submit(
            self.memory.search(query, qvec, self.cfg.recall_limit),
            on_success=lambda result: (
                self._render_cqse_neighbors(result["results"]),
                self.set_status(1.0, "CQSE neighborhood retrieved."),
            ),
            on_error=lambda e: self._show_error("CQSE neighborhood", e),
        )

    def _append_dataset_log(self, value: Any):
        self.dataset_box.insert("end", str(value) + "\n")
        self.dataset_box.see("end")

    def seed_top5_datasets(self):
        self._seed_manifests(list(CYBER_DATASETS[:5]))

    def seed_linked_malware_dataset(self):
        linked = next(m for m in CYBER_DATASETS if m.dataset_id == "agung_network_malware_connection")
        self._seed_manifests([linked])

    def _seed_manifests(self, manifests: list[CyberDatasetManifest]):
        self.cancel_event.clear()
        self.set_status(0.02, "Preparing bounded simulated dataset corpus...")

        async def work():
            # Ensure schema exists first.
            await self.memory.setup_collections()
            total = len(manifests) * DATASET_SIM_ROWS_PER_SOURCE
            done = 0
            summaries = []

            for manifest in manifests:
                records = await asyncio.to_thread(
                    self.dataset_simulator.simulate_manifest,
                    manifest,
                    DATASET_SIM_ROWS_PER_SOURCE,
                )
                label_counts: dict[str, int] = {}
                for record in records:
                    if self.cancel_event.is_set():
                        raise RuntimeError("Dataset simulation cancelled.")
                    await self.memory.insert(
                        "QNDatasetCorpusMemoryV1",
                        record,
                        "dataset-sim",
                        record["record_id"],
                        None,
                    )
                    label_counts[record["label"]] = label_counts.get(record["label"], 0) + 1
                    done += 1
                    if done % 20 == 0 or done == total:
                        self.set_status_threadsafe(
                            done / total,
                            f"Dataset memory seed {done}/{total} • {manifest.name}",
                        )
                summaries.append({
                    "dataset": manifest.name,
                    "dataset_id": manifest.dataset_id,
                    "records": len(records),
                    "simulated": True,
                    "label_counts": label_counts,
                })
            return summaries

        self._submit(
            work(),
            on_success=lambda r: (
                self._append_dataset_log(json.dumps(r, indent=2)),
                self.set_status(1.0, "Simulated dataset corpus loaded into Weaviate."),
            ),
            on_error=lambda e: self._show_error("Dataset simulation", e),
        )

    def real_kaggle_linked_import(self):
        linked = next(m for m in CYBER_DATASETS if m.dataset_id == "agung_network_malware_connection")
        if not messagebox.askyesno(
            "Explicit real-data import",
            (
                "This will invoke your configured Kaggle CLI and download the linked dataset "
                "into a local sandbox. Only CSV/TSV/JSONL/Parquet will be parsed; executable, "
                "pickle, notebook, PCAP and model files are ignored/rejected. Continue?"
            ),
        ):
            return

        self.cancel_event.clear()
        self.set_status(0.02, "Starting explicit Kaggle tabular import...")

        async def work():
            await self.memory.setup_collections()
            root = await self.dataset_importer.kaggle_download(
                linked,
                self.cancel_event,
                lambda msg: self.set_status_threadsafe(0.10, msg),
            )
            count = 0
            label_counts: dict[str, int] = {}
            async for filename, row in self.dataset_importer.iter_rows(root):
                if self.cancel_event.is_set():
                    raise RuntimeError("Kaggle import cancelled.")
                # Row becomes text-only structured memory; no source code is executed.
                label = str(
                    row.get("label")
                    or row.get("Label")
                    or row.get("class")
                    or row.get("Class")
                    or row.get("attack_cat")
                    or "unknown"
                )[:96]
                record_id = stable_hash(linked.dataset_id, filename, str(count), compact_json(row))[:32]
                props = {
                    "record_id": record_id,
                    "dataset_id": linked.dataset_id,
                    "dataset_name": linked.name,
                    "source_url": linked.source_url,
                    "source_kind": "kaggle-tabular-explicit",
                    "simulated": False,
                    "label": label,
                    "category": linked.category,
                    "summary": f"{filename} row {count}: {compact_json(row)[:3500]}",
                    "features_json": safe_json(row),
                    "safety_json": safe_json({
                        "executed_source_code": False,
                        "deserialized_python_objects": False,
                        "sandboxed_tabular_parser": True,
                        "row_cap": DATASET_MAX_ROWS_REAL,
                    }),
                    "ingested_at": utc_now(),
                    "tags": ["dataset", "real-tabular", linked.category, linked.dataset_id],
                }
                await self.memory.insert(
                    "QNDatasetCorpusMemoryV1",
                    props,
                    "dataset-real",
                    record_id,
                    None,
                )
                count += 1
                label_counts[label] = label_counts.get(label, 0) + 1
                if count % 100 == 0:
                    self.set_status_threadsafe(
                        min(0.95, 0.15 + count / max(DATASET_MAX_ROWS_REAL, 1) * 0.8),
                        f"Imported {count:,} sanitized rows → Weaviate",
                    )
            return {
                "dataset": linked.name,
                "rows_ingested": count,
                "label_counts_top": sorted(
                    label_counts.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:30],
                "sandbox": str(root),
            }

        self._submit(
            work(),
            on_success=lambda r: (
                self._append_dataset_log(json.dumps(r, indent=2)),
                self.set_status(1.0, "Explicit Kaggle import completed."),
            ),
            on_error=lambda e: self._show_error("Kaggle import", e),
        )

    def download_gemma4(self):
        self.cancel_event.clear()
        self.set_status(0.0, "Preparing SHA-256-pinned Gemma 4 E2B download...")

        self._submit(
            self.gemma4.download(
                self.cancel_event,
                lambda v, msg: self.set_status_threadsafe(v, msg),
            ),
            on_success=lambda r: (
                self._append_dataset_log("Gemma verification receipt:\n" + json.dumps(r, indent=2)),
                self.set_status(1.0, "Gemma 4 E2B verified for LiteRT-LM."),
            ),
            on_error=lambda e: self._show_error("Gemma 4 E2B", e),
        )

    def build_current_cqse(self):
        if not self.last_analytics:
            messagebox.showinfo("No telemetry", "Refresh telemetry first.")
            return

        self.cancel_event.clear()
        text = (
            "Current defensive cybersecurity telemetry, robust statistics, flow state, "
            "and telemetry quantum feature map:\n"
            + json.dumps(self.last_analytics, indent=2, default=str)
        )
        telemetry_quantum = self.last_analytics["quantum"]["probabilities"]
        gemma_conditioned = self.cqse_palette_mode.get() == "Gemma Conditioned"

        self.set_status(0.05, "Generating current telemetry CQSE surface...")
        self._submit(
            self._generate_cqse(text, gemma_conditioned, telemetry_quantum),
            on_success=lambda result: (
                self._render_cqse_surface(
                    result["surface"],
                    result["palette"],
                    result["neighbors"],
                ),
                self._append_dataset_log(
                    "Current CQSE completed:\n"
                    + json.dumps({"errors": result["errors"]}, indent=2)
                ),
                self.set_status(1.0, "Current CQSE surface generated."),
            ),
            on_error=lambda e: self._show_error("Current CQSE", e),
        )

    def build_gemma_cqse(self):
        if not self.last_analytics:
            messagebox.showinfo("No telemetry", "Refresh telemetry first.")
            return

        self.cancel_event.clear()
        text = (
            "Defensive cyber telemetry for Gemma palette-conditioned CQSE:\n"
            + json.dumps(
                {
                    "current": self.last_analytics["current"],
                    "anomalies": self.last_analytics["current_anomalies"],
                    "trends": {
                        key: value["trend"]
                        for key, value in self.last_analytics["metrics"].items()
                    },
                    "flows": self.last_analytics["flow_analysis"],
                    "telemetry_quantum": self.last_analytics["quantum"],
                },
                indent=2,
                default=str,
            )
        )
        telemetry_quantum = self.last_analytics["quantum"]["probabilities"]

        self.set_status(0.05, "Generating Gemma-conditioned CQSE surface...")
        self._submit(
            self._generate_cqse(text, True, telemetry_quantum),
            on_success=lambda result: (
                self._render_cqse_surface(
                    result["surface"],
                    result["palette"],
                    result["neighbors"],
                ),
                self._append_dataset_log(
                    "Gemma-CQSE completed:\n"
                    + json.dumps({"errors": result["errors"]}, indent=2)
                ),
                self.set_status(1.0, "Gemma-conditioned CQSE complete."),
            ),
            on_error=lambda e: self._show_error("Gemma CQSE", e),
        )

    def _build_temporal_hyperfield(self):
        tab = self.tabs.tab("Temporal Hyperfield")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(3, weight=1)

        # Setup/control band.
        top = ctk.CTkFrame(tab)
        top.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        top.grid_columnconfigure(9, weight=1)

        ctk.CTkLabel(
            top,
            text="TEMPORAL CHROMATIC QUANTUM HYPERFIELD // 4D CYBER WORLDLINE",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, columnspan=10, padx=9, pady=(8, 3), sticky="w")

        ctk.CTkButton(
            top,
            text="Build Hyperfield",
            command=self.build_temporal_hyperfield,
        ).grid(row=1, column=0, padx=5, pady=7)

        ctk.CTkButton(
            top,
            text="Correlate Regimes → Weaviate",
            command=self.correlate_hyperfield_regimes,
        ).grid(row=1, column=1, padx=5, pady=7)

        ctk.CTkButton(
            top,
            text="Store Worldline Snapshot",
            command=self.store_hyperfield_snapshot,
        ).grid(row=1, column=2, padx=5, pady=7)

        self.hyperfield_tail_entry = ctk.CTkEntry(top, width=85)
        self.hyperfield_tail_entry.insert(0, "120")
        self.hyperfield_tail_entry.grid(row=1, column=3, padx=(12, 3), pady=7)
        ctk.CTkLabel(top, text="trail samples").grid(
            row=1, column=4, padx=(0, 7), pady=7
        )

        self.hyperfield_boundary_switch = ctk.CTkSwitch(top, text="Boundary markers")
        self.hyperfield_boundary_switch.select()
        self.hyperfield_boundary_switch.grid(row=1, column=5, padx=7, pady=7)

        self.hyperfield_attractor_switch = ctk.CTkSwitch(top, text="Attractor centroids")
        self.hyperfield_attractor_switch.select()
        self.hyperfield_attractor_switch.grid(row=1, column=6, padx=7, pady=7)

        self.hyperfield_quantum_switch = ctk.CTkSwitch(top, text="Quantum divergence")
        self.hyperfield_quantum_switch.select()
        self.hyperfield_quantum_switch.grid(row=1, column=7, padx=7, pady=7)

        self.hyperfield_state_label = ctk.CTkLabel(
            top, text="No worldline built.", anchor="e"
        )
        self.hyperfield_state_label.grid(row=1, column=9, padx=8, pady=7, sticky="e")

        # Replay band.
        replay = ctk.CTkFrame(tab)
        replay.grid(row=1, column=0, padx=6, pady=3, sticky="ew")
        replay.grid_columnconfigure(2, weight=1)

        self.hyperfield_play_btn = ctk.CTkButton(
            replay, text="▶ Replay", command=self.toggle_hyperfield_replay, width=95
        )
        self.hyperfield_play_btn.grid(row=0, column=0, padx=6, pady=6)

        ctk.CTkButton(
            replay, text="⟲ Reset", command=self.reset_hyperfield_replay, width=85
        ).grid(row=0, column=1, padx=6, pady=6)

        self.hyperfield_slider = ctk.CTkSlider(
            replay,
            from_=0,
            to=1,
            number_of_steps=1,
            command=self._hyperfield_slider_changed,
        )
        self.hyperfield_slider.set(0)
        self.hyperfield_slider.grid(row=0, column=2, padx=8, pady=6, sticky="ew")

        self.hyperfield_replay_label = ctk.CTkLabel(
            replay, text="t=— | regime=— | anomaly=—", anchor="e"
        )
        self.hyperfield_replay_label.grid(row=0, column=3, padx=8, pady=6, sticky="e")

        # Metrics band.
        metrics = ctk.CTkFrame(tab)
        metrics.grid(row=2, column=0, padx=6, pady=3, sticky="ew")
        metrics.grid_columnconfigure((0, 1, 2, 3, 4, 5, 6), weight=1)

        metric_defs = [
            ("points", "Worldline Points"),
            ("regimes", "Attractors"),
            ("boundaries", "Boundaries"),
            ("velocity", "Mean Velocity"),
            ("curvature", "Mean Curvature"),
            ("qjs", "Mean Q-JS"),
            ("entropy", "Mean Q Entropy"),
        ]
        self.hyperfield_metrics = {}
        for col, (key, title) in enumerate(metric_defs):
            card = ctk.CTkFrame(metrics)
            card.grid(row=0, column=col, padx=4, pady=5, sticky="ew")
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=10, weight="bold")
            ).grid(row=0, column=0, padx=7, pady=(6, 1))
            value = ctk.CTkLabel(
                card, text="—", font=ctk.CTkFont(size=15, weight="bold")
            )
            value.grid(row=1, column=0, padx=7, pady=(0, 6))
            self.hyperfield_metrics[key] = value

        # Workbench tabs.
        workbench = ctk.CTkTabview(tab)
        workbench.grid(row=3, column=0, padx=6, pady=(3, 6), sticky="nsew")
        for name in (
            "4D Worldline",
            "Regime Field",
            "Boundary Matrix",
            "Quantum Divergence",
            "Memory Resonance",
            "Raw Hyperfield",
        ):
            workbench.add(name)

        # 4D worldline (XYZ=Lab, 4th dimension time via marker size/color).
        world = workbench.tab("4D Worldline")
        world.grid_columnconfigure(0, weight=1)
        world.grid_rowconfigure(0, weight=1)
        self.hyperfield_fig = Figure(figsize=(11.5, 6.5), dpi=100)
        self.hyperfield_ax_world = self.hyperfield_fig.add_subplot(221, projection="3d")
        self.hyperfield_ax_motion = self.hyperfield_fig.add_subplot(222)
        self.hyperfield_ax_entropy = self.hyperfield_fig.add_subplot(223)
        self.hyperfield_ax_regime = self.hyperfield_fig.add_subplot(224)
        self.hyperfield_fig.tight_layout(pad=2.0)
        self.hyperfield_canvas = FigureCanvasTkAgg(self.hyperfield_fig, master=world)
        self.hyperfield_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Regime field.
        regime_tab = workbench.tab("Regime Field")
        regime_tab.grid_columnconfigure((0, 1), weight=1)
        regime_tab.grid_rowconfigure(1, weight=1)
        self.hyperfield_regime_summary = ctk.CTkTextbox(
            regime_tab, height=150, wrap="word"
        )
        self.hyperfield_regime_summary.grid(
            row=0, column=0, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        self.hyperfield_regime_tree = ttk.Treeview(
            regime_tab,
            columns=("regime", "samples", "ping", "jitter", "download", "qentropy", "anomaly"),
            show="headings",
        )
        for col, width in (
            ("regime", 70), ("samples", 70), ("ping", 90), ("jitter", 90),
            ("download", 100), ("qentropy", 100), ("anomaly", 100),
        ):
            self.hyperfield_regime_tree.heading(col, text=col.upper())
            self.hyperfield_regime_tree.column(col, width=width, anchor="w")
        self.hyperfield_regime_tree.grid(
            row=1, column=0, columnspan=2, padx=8, pady=8, sticky="nsew"
        )

        # Boundary matrix.
        boundary_tab = workbench.tab("Boundary Matrix")
        boundary_tab.grid_columnconfigure(0, weight=1)
        boundary_tab.grid_rowconfigure(0, weight=1)
        self.hyperfield_boundary_tree = ttk.Treeview(
            boundary_tab,
            columns=("index", "time", "from", "to", "change", "anomaly", "qjs", "reason"),
            show="headings",
        )
        for col, width in (
            ("index", 60), ("time", 180), ("from", 60), ("to", 60),
            ("change", 90), ("anomaly", 90), ("qjs", 90), ("reason", 320),
        ):
            self.hyperfield_boundary_tree.heading(col, text=col.upper())
            self.hyperfield_boundary_tree.column(col, width=width, anchor="w")
        self.hyperfield_boundary_tree.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        # Quantum divergence tab.
        qtab = workbench.tab("Quantum Divergence")
        qtab.grid_columnconfigure((0, 1), weight=1)
        qtab.grid_rowconfigure(1, weight=1)
        self.hyperfield_quantum_summary = ctk.CTkTextbox(qtab, height=135, wrap="word")
        self.hyperfield_quantum_summary.grid(
            row=0, column=0, columnspan=2, padx=8, pady=8, sticky="ew"
        )
        self.hyperfield_quantum_box = ctk.CTkTextbox(qtab, wrap="none")
        self.hyperfield_quantum_box.grid(
            row=1, column=0, padx=(8, 4), pady=8, sticky="nsew"
        )
        self.hyperfield_motion_box = ctk.CTkTextbox(qtab, wrap="word")
        self.hyperfield_motion_box.grid(
            row=1, column=1, padx=(4, 8), pady=8, sticky="nsew"
        )

        # Memory resonance.
        memory_tab = workbench.tab("Memory Resonance")
        memory_tab.grid_columnconfigure(0, weight=1)
        memory_tab.grid_rowconfigure(1, weight=1)
        self.hyperfield_memory_label = ctk.CTkLabel(
            memory_tab, text="No regime-memory correlation run yet.", anchor="w"
        )
        self.hyperfield_memory_label.grid(row=0, column=0, padx=8, pady=6, sticky="ew")
        self.hyperfield_memory_tree = ttk.Treeview(
            memory_tab,
            columns=("regime", "rank", "collection", "score", "sources", "summary"),
            show="headings",
        )
        for col, width in (
            ("regime", 70), ("rank", 55), ("collection", 200),
            ("score", 90), ("sources", 220), ("summary", 600),
        ):
            self.hyperfield_memory_tree.heading(col, text=col.upper())
            self.hyperfield_memory_tree.column(col, width=width, anchor="w")
        self.hyperfield_memory_tree.grid(
            row=1, column=0, padx=8, pady=8, sticky="nsew"
        )

        # Raw.
        raw_tab = workbench.tab("Raw Hyperfield")
        raw_tab.grid_columnconfigure(0, weight=1)
        raw_tab.grid_rowconfigure(0, weight=1)
        self.hyperfield_raw_box = ctk.CTkTextbox(raw_tab, wrap="word")
        self.hyperfield_raw_box.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.hyperfield_raw_box.insert(
            "1.0",
            (
                "Build a Temporal Hyperfield to populate the replayable cyber worldline.\n\n"
                "Interpretation note: time, colorspace, semantic entropy and PennyLane "
                "probabilities are computational feature surfaces. They are not literal "
                "physical spacetime or evidence of quantum causality."
            ),
        )

    def build_temporal_hyperfield(self):
        if len(self.last_rows) < 2:
            messagebox.showinfo("Insufficient telemetry", "Refresh telemetry with at least two samples.")
            return

        self.cancel_event.clear()
        self.set_status(0.05, "Constructing temporal chromatic quantum worldline...")

        async def work():
            return await asyncio.to_thread(
                self.temporal_hyperfield.build,
                self.last_rows,
            )

        self._submit(
            work(),
            on_success=lambda field: (
                setattr(self, "last_hyperfield", field),
                self._render_temporal_hyperfield(field),
                self.set_status(1.0, "Temporal Hyperfield constructed."),
            ),
            on_error=lambda e: self._show_error("Temporal Hyperfield", e),
        )

    def _render_temporal_hyperfield(self, field: dict[str, Any]):
        points = field["points"]
        if not points:
            return

        self.hyperfield_replay_index = len(points) - 1
        self.hyperfield_slider.configure(
            from_=0,
            to=max(1, len(points) - 1),
            number_of_steps=max(1, len(points) - 1),
        )
        self.hyperfield_slider.set(self.hyperfield_replay_index)

        velocities = np.asarray([p["velocity"] for p in points], dtype=float)
        curvatures = np.asarray([p["curvature"] for p in points], dtype=float)
        qjs = np.asarray([p["quantum_js_divergence"] for p in points], dtype=float)
        qentropy = np.asarray([p["quantum_entropy"] for p in points], dtype=float)

        self.hyperfield_metrics["points"].configure(text=str(len(points)))
        self.hyperfield_metrics["regimes"].configure(text=str(len(field["regimes"])))
        self.hyperfield_metrics["boundaries"].configure(text=str(len(field["boundaries"])))
        self.hyperfield_metrics["velocity"].configure(text=f"{velocities.mean():.3f}")
        self.hyperfield_metrics["curvature"].configure(text=f"{curvatures.mean():.3f}")
        self.hyperfield_metrics["qjs"].configure(text=f"{qjs.mean():.4f}")
        self.hyperfield_metrics["entropy"].configure(text=f"{qentropy.mean():.3f}")

        self.hyperfield_state_label.configure(
            text=(
                f"{len(points)} points • {len(field['regimes'])} attractors • "
                f"{len(field['boundaries'])} boundaries"
            )
        )

        # Regime table.
        for item in self.hyperfield_regime_tree.get_children():
            self.hyperfield_regime_tree.delete(item)
        for regime in field["regimes"]:
            self.hyperfield_regime_tree.insert(
                "",
                "end",
                values=(
                    regime["regime_id"],
                    regime["samples"],
                    f"{regime['mean_ping']:.2f}",
                    f"{regime['mean_jitter']:.2f}",
                    f"{regime['mean_download']:.2f}",
                    f"{regime['mean_quantum_entropy']:.3f}",
                    f"{regime['max_anomaly_score']:.3f}",
                ),
            )

        self.hyperfield_regime_summary.delete("1.0", "end")
        self.hyperfield_regime_summary.insert(
            "1.0",
            json.dumps(
                {
                    "experimental_notice": field["experimental_notice"],
                    "regime_count": len(field["regimes"]),
                    "scales": field["scales"],
                },
                indent=2,
            ),
        )

        # Boundary table.
        for item in self.hyperfield_boundary_tree.get_children():
            self.hyperfield_boundary_tree.delete(item)
        for boundary in field["boundaries"]:
            self.hyperfield_boundary_tree.insert(
                "",
                "end",
                values=(
                    boundary["index"],
                    boundary["timestamp"],
                    boundary["regime_from"],
                    boundary["regime_to"],
                    f"{boundary['change_score']:.3f}",
                    f"{boundary['anomaly_score']:.3f}",
                    f"{boundary['quantum_js_divergence']:.4f}",
                    ",".join(boundary["reason"]),
                ),
            )

        # Raw quantum/motion summaries.
        self.hyperfield_quantum_summary.delete("1.0", "end")
        self.hyperfield_quantum_summary.insert(
            "1.0",
            json.dumps(
                {
                    "mean_q32_entropy_bits": float(qentropy.mean()),
                    "max_q32_entropy_bits": float(qentropy.max()),
                    "mean_js_divergence": float(qjs.mean()),
                    "max_js_divergence": float(qjs.max()),
                    "max_change_score": max(p["change_score"] for p in points),
                    "max_anomaly_score": max(p["anomaly_score"] for p in points),
                },
                indent=2,
            ),
        )

        self.hyperfield_quantum_box.delete("1.0", "end")
        self.hyperfield_quantum_box.insert(
            "1.0",
            "\n".join(
                (
                    f"{p['index']:04d}  R={p['regime_id']}  "
                    f"QH={p['quantum_entropy']:.5f}  "
                    f"QJS={p['quantum_js_divergence']:.6f}  "
                    f"Δ={p['change_score']:.4f}"
                )
                for p in points
            ),
        )

        self.hyperfield_motion_box.delete("1.0", "end")
        self.hyperfield_motion_box.insert(
            "1.0",
            "\n".join(
                (
                    f"{p['index']:04d} | v={p['velocity']:.5f} | "
                    f"a={p['acceleration']:.5f} | κ={p['curvature']:.5f} | "
                    f"attractor_d={p['attractor_distance']:.5f} | "
                    f"anomaly={p['anomaly_score']:.5f}"
                )
                for p in points
            ),
        )

        self.hyperfield_raw_box.delete("1.0", "end")
        self.hyperfield_raw_box.insert("1.0", json.dumps(field, indent=2, default=str))

        self._render_hyperfield_replay_frame(self.hyperfield_replay_index)

    def _render_hyperfield_replay_frame(self, index: int):
        if not self.last_hyperfield:
            return
        points = self.last_hyperfield["points"]
        if not points:
            return

        index = max(0, min(int(index), len(points) - 1))
        self.hyperfield_replay_index = index

        try:
            trail = max(8, int(self.hyperfield_tail_entry.get()))
        except Exception:
            trail = 120

        start = max(0, index - trail + 1)
        visible = points[start:index + 1]

        self.hyperfield_ax_world.clear()
        self.hyperfield_ax_motion.clear()
        self.hyperfield_ax_entropy.clear()
        self.hyperfield_ax_regime.clear()

        L = np.asarray([p["lab_l"] for p in visible], dtype=float)
        A = np.asarray([p["lab_a"] for p in visible], dtype=float)
        B = np.asarray([p["lab_b"] for p in visible], dtype=float)
        T = np.asarray([p["t_norm"] for p in visible], dtype=float)
        anomaly = np.asarray([p["anomaly_score"] for p in visible], dtype=float)
        qentropy = np.asarray([p["quantum_entropy"] for p in visible], dtype=float)

        # Time acts as the fourth visual coordinate through color and marker size.
        sizes = 30.0 + 70.0 * (T - T.min()) / max(float(np.ptp(T)), 1e-9)
        scatter = self.hyperfield_ax_world.scatter(
            A, B, L,
            c=T,
            s=sizes,
            cmap="turbo",
            alpha=0.88,
        )
        self.hyperfield_ax_world.plot(A, B, L, alpha=0.45, linewidth=1.2)
        self.hyperfield_ax_world.set_xlabel("OKLab a")
        self.hyperfield_ax_world.set_ylabel("OKLab b")
        self.hyperfield_ax_world.set_zlabel("OKLab L")
        self.hyperfield_ax_world.set_title("4D worldline: Lab + normalized time")

        if self.hyperfield_attractor_switch.get():
            for regime in self.last_hyperfield["regimes"]:
                lab = regime["mean_lab"]
                self.hyperfield_ax_world.scatter(
                    [lab[1]], [lab[2]], [lab[0]],
                    marker="X",
                    s=180,
                    edgecolors="black",
                    linewidths=0.8,
                )
                self.hyperfield_ax_world.text(
                    lab[1], lab[2], lab[0], f" R{regime['regime_id']}"
                )

        if self.hyperfield_boundary_switch.get():
            boundary_indexes = {
                b["index"] for b in self.last_hyperfield["boundaries"]
                if start <= b["index"] <= index
            }
            for global_idx in sorted(boundary_indexes):
                p = points[global_idx]
                self.hyperfield_ax_world.scatter(
                    [p["lab_a"]], [p["lab_b"]], [p["lab_l"]],
                    marker="D",
                    s=95,
                    edgecolors="black",
                    linewidths=0.8,
                )

        x = np.asarray([p["index"] for p in visible], dtype=int)
        self.hyperfield_ax_motion.plot(x, [p["velocity"] for p in visible], label="velocity")
        self.hyperfield_ax_motion.plot(x, [p["acceleration"] for p in visible], label="acceleration")
        self.hyperfield_ax_motion.plot(x, [p["curvature"] for p in visible], label="curvature")
        self.hyperfield_ax_motion.set_title("Hyperfield motion")
        self.hyperfield_ax_motion.legend(loc="upper left")
        self.hyperfield_ax_motion.grid(True, alpha=0.22)

        self.hyperfield_ax_entropy.plot(
            x, qentropy, label="CQSE quantum entropy"
        )
        if self.hyperfield_quantum_switch.get():
            self.hyperfield_ax_entropy.plot(
                x,
                [p["quantum_js_divergence"] for p in visible],
                label="Q Jensen-Shannon divergence",
            )
        self.hyperfield_ax_entropy.set_title("Quantum-state evolution")
        self.hyperfield_ax_entropy.legend(loc="upper left")
        self.hyperfield_ax_entropy.grid(True, alpha=0.22)

        regimes = np.asarray([p["regime_id"] for p in visible], dtype=float)
        self.hyperfield_ax_regime.step(x, regimes, where="mid", label="regime")
        self.hyperfield_ax_regime.plot(
            x,
            [p["change_score"] for p in visible],
            label="change score",
        )
        self.hyperfield_ax_regime.plot(
            x,
            anomaly,
            label="anomaly score",
        )
        self.hyperfield_ax_regime.set_title("Regime / boundary energy")
        self.hyperfield_ax_regime.legend(loc="upper left")
        self.hyperfield_ax_regime.grid(True, alpha=0.22)

        self.hyperfield_fig.tight_layout(pad=1.6)
        self.hyperfield_canvas.draw_idle()

        current = points[index]
        self.hyperfield_replay_label.configure(
            text=(
                f"t={index}/{len(points)-1} | R={current['regime_id']} | "
                f"v={current['velocity']:.3f} | κ={current['curvature']:.3f} | "
                f"QJS={current['quantum_js_divergence']:.4f} | "
                f"anom={current['anomaly_score']:.3f}"
            )
        )

    def _hyperfield_slider_changed(self, value):
        if not self.last_hyperfield:
            return
        self._render_hyperfield_replay_frame(int(round(float(value))))

    def toggle_hyperfield_replay(self):
        if not self.last_hyperfield:
            messagebox.showinfo("No Hyperfield", "Build the Temporal Hyperfield first.")
            return
        self.hyperfield_playing = not self.hyperfield_playing
        self.hyperfield_play_btn.configure(
            text="⏸ Pause" if self.hyperfield_playing else "▶ Replay"
        )
        if self.hyperfield_playing:
            if self.hyperfield_replay_index >= len(self.last_hyperfield["points"]) - 1:
                self.hyperfield_replay_index = 0
                self.hyperfield_slider.set(0)
            self._hyperfield_replay_tick()

    def _hyperfield_replay_tick(self):
        if not self.hyperfield_playing or not self.last_hyperfield:
            return
        points = self.last_hyperfield["points"]
        if self.hyperfield_replay_index >= len(points) - 1:
            self.hyperfield_playing = False
            self.hyperfield_play_btn.configure(text="▶ Replay")
            return
        self.hyperfield_replay_index += 1
        self.hyperfield_slider.set(self.hyperfield_replay_index)
        self._render_hyperfield_replay_frame(self.hyperfield_replay_index)
        self.hyperfield_replay_job = self.after(90, self._hyperfield_replay_tick)

    def reset_hyperfield_replay(self):
        self.hyperfield_playing = False
        self.hyperfield_play_btn.configure(text="▶ Replay")
        if self.hyperfield_replay_job is not None:
            try:
                self.after_cancel(self.hyperfield_replay_job)
            except Exception:
                pass
            self.hyperfield_replay_job = None
        self.hyperfield_replay_index = 0
        self.hyperfield_slider.set(0)
        self._render_hyperfield_replay_frame(0)

    def correlate_hyperfield_regimes(self):
        if not self.last_hyperfield:
            messagebox.showinfo("No Hyperfield", "Build the Temporal Hyperfield first.")
            return

        self.set_status(0.05, "Correlating hyperfield attractors against vector memory...")

        async def work():
            output = []
            for regime in self.last_hyperfield["regimes"]:
                query = (
                    f"Temporal cyber regime {regime['regime_id']} with "
                    f"mean ping {regime['mean_ping']:.2f} ms, "
                    f"mean jitter {regime['mean_jitter']:.2f} ms, "
                    f"download {regime['mean_download']:.2f} Mbps, "
                    f"quantum entropy {regime['mean_quantum_entropy']:.4f}, "
                    f"change score {regime['mean_change_score']:.4f}, "
                    f"maximum anomaly {regime['max_anomaly_score']:.4f}."
                )
                try:
                    recall = await self.memory.search(
                        query,
                        None,
                        min(6, self.cfg.recall_limit),
                    )
                    for rank, hit in enumerate(recall["results"], start=1):
                        output.append({
                            "regime": regime["regime_id"],
                            "rank": rank,
                            **hit,
                        })
                except Exception as exc:
                    output.append({
                        "regime": regime["regime_id"],
                        "rank": 0,
                        "collection": "ERROR",
                        "rrf_score": 0.0,
                        "sources": [],
                        "properties": {"summary": str(exc)},
                    })
            return output

        self._submit(
            work(),
            on_success=lambda rows: (
                self._render_hyperfield_memory(rows),
                self.set_status(1.0, "Hyperfield regime-memory resonance complete."),
            ),
            on_error=lambda e: self._show_error("Hyperfield memory correlation", e),
        )

    def _render_hyperfield_memory(self, rows: list[dict[str, Any]]):
        for item in self.hyperfield_memory_tree.get_children():
            self.hyperfield_memory_tree.delete(item)

        for row in rows:
            props = row.get("properties", {})
            summary = (
                props.get("summary")
                or props.get("final_text")
                or props.get("title")
                or safe_json(props)
            )
            sources = ",".join(
                str(x.get("source")) for x in row.get("sources", [])
            )
            self.hyperfield_memory_tree.insert(
                "",
                "end",
                values=(
                    row.get("regime"),
                    row.get("rank"),
                    row.get("collection"),
                    f"{float(row.get('rrf_score', 0.0)):.6f}",
                    sources,
                    str(summary)[:260],
                ),
            )

        self.hyperfield_memory_label.configure(
            text=f"{len(rows)} cross-regime memory resonances"
        )

    def store_hyperfield_snapshot(self):
        if not self.last_hyperfield:
            messagebox.showinfo("No Hyperfield", "Build the Temporal Hyperfield first.")
            return

        field = self.last_hyperfield
        points = field["points"]
        summary = (
            f"Temporal hyperfield with {field['point_count']} worldline points, "
            f"{len(field['regimes'])} attractor regimes, "
            f"{len(field['boundaries'])} transition boundaries, "
            f"mean quantum entropy {np.mean([p['quantum_entropy'] for p in points]):.4f}, "
            f"mean quantum JS divergence {np.mean([p['quantum_js_divergence'] for p in points]):.5f}."
        )
        current_q16 = (
            self.last_analytics["quantum"]["probabilities"]
            if self.last_analytics else None
        )

        async def work():
            await self.memory.setup_collections()
            worldline_id = str(uuid.uuid4())
            props = {
                "worldline_id": worldline_id,
                "created_at": utc_now(),
                "summary": summary,
                "point_count": int(field["point_count"]),
                "regime_count": int(len(field["regimes"])),
                "boundary_count": int(len(field["boundaries"])),
                "mean_velocity": float(np.mean([p["velocity"] for p in points])),
                "mean_curvature": float(np.mean([p["curvature"] for p in points])),
                "mean_quantum_entropy": float(np.mean([p["quantum_entropy"] for p in points])),
                "mean_quantum_js": float(np.mean([p["quantum_js_divergence"] for p in points])),
                "max_anomaly_score": float(max(p["anomaly_score"] for p in points)),
                "regimes_json": safe_json(field["regimes"]),
                "boundaries_json": safe_json(field["boundaries"]),
                "scales_json": safe_json(field["scales"]),
                "experimental_notice": field["experimental_notice"],
                "tags": ["tcqh", "temporal-worldline", "attractor-field", "experimental"],
            }
            result = await self.memory.insert(
                "QNTemporalHyperfieldMemoryV1",
                props,
                "temporal-hyperfield",
                worldline_id,
                current_q16,
            )
            return result

        self._submit(
            work(),
            on_success=lambda result: self.set_status(
                1.0,
                f"Worldline stored in {result['collection']}.",
            ),
            on_error=lambda e: self._show_error("Store Hyperfield", e),
        )

    def _build_protocol(self):
        tab = self.tabs.tab("Protocol / Actions")
        tab.grid_columnconfigure((0, 1), weight=1)
        tab.grid_rowconfigure(1, weight=1)

        left = ctk.CTkFrame(tab)
        right = ctk.CTkFrame(tab)
        left.grid(row=0, column=0, padx=(6, 3), pady=6, sticky="nsew", rowspan=2)
        right.grid(row=0, column=1, padx=(3, 6), pady=6, sticky="nsew", rowspan=2)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left, text="Structured blocks").grid(
            row=0, column=0, padx=8, pady=6, sticky="w"
        )
        self.protocol_box = ctk.CTkTextbox(left, wrap="word")
        self.protocol_box.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        ctk.CTkLabel(right, text="Human approval queue").grid(
            row=0, column=0, padx=8, pady=6, sticky="w"
        )

        self.action_tree = ttk.Treeview(
            right,
            columns=("priority", "risk", "state", "title"),
            show="headings",
            height=14,
        )
        for col, width in [("priority", 80), ("risk", 80), ("state", 95), ("title", 360)]:
            self.action_tree.heading(col, text=col.upper())
            self.action_tree.column(col, width=width, anchor="w")
        self.action_tree.grid(row=1, column=0, padx=8, pady=8, sticky="nsew")

        action_bar = ctk.CTkFrame(right)
        action_bar.grid(row=2, column=0, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(
            action_bar, text="Approve selected", command=lambda: self.set_selected_action("approved")
        ).grid(row=0, column=0, padx=6, pady=6)
        ctk.CTkButton(
            action_bar, text="Reject selected", command=lambda: self.set_selected_action("rejected")
        ).grid(row=0, column=1, padx=6, pady=6)
        ctk.CTkButton(
            action_bar, text="Refresh queue", command=self.refresh_action_queue
        ).grid(row=0, column=2, padx=6, pady=6)

    def _build_auth(self):
        tab = self.tabs.tab("Auth / Models")
        tab.grid_columnconfigure(1, weight=1)

        info = [
            ("Codex", self.providers.codex.version()),
            ("Daybreak alias", DAYBREAK_ALIAS),
            ("Standard API fallback", STANDARD_MODEL),
            ("OpenAI embedding", f"{EMBED_MODEL} / {EMBED_DIM}D"),
            ("Facebook embedding", f"{FB_EMBED_MODEL}@{FB_EMBED_REVISION[:12]} / {FB_EMBED_DIM}D / safetensors"),
            ("Legacy local model", LEGACY_MODEL_NAME),
            ("Gemma 4 LiteRT model", f"{GEMMA4_MODEL_NAME} / SHA {GEMMA4_MODEL_SHA256[:12]}…"),
        ]
        for row, (label, value) in enumerate(info):
            ctk.CTkLabel(tab, text=label, anchor="w").grid(
                row=row, column=0, padx=12, pady=6, sticky="w"
            )
            ctk.CTkLabel(tab, text=value, anchor="w").grid(
                row=row, column=1, padx=12, pady=6, sticky="ew"
            )

        ctk.CTkButton(
            tab, text="Launch official Codex login", command=self.launch_codex_login
        ).grid(row=7, column=0, padx=12, pady=10, sticky="w")

        ctk.CTkLabel(
            tab,
            text=(
                "The app invokes `codex login`; it never reads Codex/browser token storage."
            ),
            anchor="w",
        ).grid(row=7, column=1, padx=12, pady=10, sticky="ew")

        ctk.CTkLabel(tab, text="OpenAI API key (session only)").grid(
            row=8, column=0, padx=12, pady=6, sticky="w"
        )
        self.api_key_entry = ctk.CTkEntry(tab, show="•")
        if self.session_openai_key:
            self.api_key_entry.insert(0, self.session_openai_key)
        self.api_key_entry.grid(row=8, column=1, padx=12, pady=6, sticky="ew")

        ctk.CTkButton(
            tab, text="Apply session secrets", command=self.apply_session_secrets
        ).grid(row=9, column=0, padx=12, pady=10, sticky="w")

        ctk.CTkButton(
            tab, text="Download / verify legacy Llama", command=self.download_legacy_model
        ).grid(row=10, column=0, padx=12, pady=10, sticky="w")

        self.auth_info = ctk.CTkTextbox(tab, height=260, wrap="word")
        self.auth_info.grid(row=11, column=0, columnspan=2, padx=12, pady=12, sticky="ew")
        self.auth_info.insert(
            "1.0",
            (
                "Daybreak Blue requires approved organization/workspace/project provisioning.\n"
                "The model/program selection does not grant access.\n\n"
                "Provider order in Auto mode:\n"
                "1. Codex CLI + ChatGPT sign-in → gpt-daybreak-blue-latest\n"
                "2. Responses API → gpt-daybreak-blue-latest / daybreak_blue\n"
                "3. Responses API → gpt-6-sol / standard\n"
                "4. Optional verified local Llama-2 GGMLv3 fallback\n\n"
                "Secrets entered in this tab live only in process memory."
            ),
        )
        self.auth_info.configure(state="disabled")

    def _build_settings(self):
        tab = self.tabs.tab("Settings")
        tab.grid_columnconfigure(1, weight=1)

        settings = [
            ("Database path", "database_path", self.cfg.database_path),
            ("History samples", "history_limit", str(self.cfg.history_limit)),
            ("Flow samples", "flow_limit", str(self.cfg.flow_limit)),
            ("Refresh seconds", "refresh_seconds", str(self.cfg.refresh_seconds)),
            ("Reasoning effort", "reasoning_effort", self.cfg.reasoning_effort),
            ("Recall limit", "recall_limit", str(self.cfg.recall_limit)),
            ("RRF k", "rrf_k", str(self.cfg.rrf_k)),
        ]
        self.setting_entries = {}
        for row, (label, key, value) in enumerate(settings):
            ctk.CTkLabel(tab, text=label, anchor="w").grid(
                row=row, column=0, padx=12, pady=6, sticky="w"
            )
            entry = ctk.CTkEntry(tab)
            entry.insert(0, value)
            entry.grid(row=row, column=1, padx=12, pady=6, sticky="ew")
            self.setting_entries[key] = entry

        ctk.CTkLabel(tab, text="Weaviate URL (session only)").grid(
            row=8, column=0, padx=12, pady=6, sticky="w"
        )
        self.weaviate_url_entry = ctk.CTkEntry(tab)
        self.weaviate_url_entry.insert(0, self.session_weaviate_url)
        self.weaviate_url_entry.grid(row=8, column=1, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(tab, text="Weaviate API key (session only)").grid(
            row=9, column=0, padx=12, pady=6, sticky="w"
        )
        self.weaviate_key_entry = ctk.CTkEntry(tab, show="•")
        if self.session_weaviate_key:
            self.weaviate_key_entry.insert(0, self.session_weaviate_key)
        self.weaviate_key_entry.grid(row=9, column=1, padx=12, pady=6, sticky="ew")

        self.openai_embed_switch = ctk.CTkSwitch(tab, text="OpenAI embeddings")
        if self.cfg.use_openai_embeddings:
            self.openai_embed_switch.select()
        self.openai_embed_switch.grid(row=10, column=0, padx=12, pady=8, sticky="w")

        self.fb_embed_switch = ctk.CTkSwitch(tab, text="Facebook Contriever embeddings")
        if self.cfg.use_fb_embeddings:
            self.fb_embed_switch.select()
        self.fb_embed_switch.grid(row=10, column=1, padx=12, pady=8, sticky="w")

        ctk.CTkButton(
            tab, text="Choose database", command=self.choose_database
        ).grid(row=11, column=0, padx=12, pady=12, sticky="w")
        ctk.CTkButton(
            tab, text="Save settings", command=self.save_settings
        ).grid(row=11, column=1, padx=12, pady=12, sticky="w")

    def _build_logs(self):
        tab = self.tabs.tab("Logs")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        self.log_box = ctk.CTkTextbox(tab, wrap="none")
        self.log_box.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

    # ------------------------------------------------------------------
    # Tk / async helpers
    # ------------------------------------------------------------------

    def _submit(self, coro, on_success=None, on_error=None):
        fut = self.runtime.submit(coro)

        def done(f):
            try:
                value = f.result()
                if on_success:
                    self.after(0, lambda: on_success(value))
            except Exception as exc:
                logging.exception("Async operation failed")
                if on_error:
                    self.after(0, lambda exc=exc: on_error(exc))
                else:
                    self.after(0, lambda exc=exc: self._show_error("Operation failed", exc))
        fut.add_done_callback(done)
        return fut

    def _show_error(self, title: str, exc: Exception):
        self.set_status(0.0, f"{title}: {exc}")
        messagebox.showerror(title, str(exc))

    def set_status(self, value: float, text: str):
        self.progress.set(float(np.clip(value, 0.0, 1.0)))
        self.status_label.configure(text=text)

    def set_status_threadsafe(self, value: float, text: str):
        self.after(0, lambda: self.set_status(value, text))

    def _append_log(self, text: str):
        if not hasattr(self, "log_box"):
            return
        self.after(0, lambda: (self.log_box.insert("end", text + "\n"), self.log_box.see("end")))

    def _set_busy(self, busy: bool):
        self._busy = busy
        self.run_ai_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")

    def cancel_current(self):
        self.cancel_event.set()
        self.status_label.configure(text="Cancellation requested...")

    # ------------------------------------------------------------------
    # Async application operations
    # ------------------------------------------------------------------

    async def _bootstrap(self):
        self.set_status_threadsafe(0.05, "Initializing aiosqlite WAL database...")
        await self.store.initialize()
        self.set_status_threadsafe(1.0, "Database ready.")

    def refresh_telemetry(self):
        if self._busy:
            return
        self._set_busy(True)
        self.cancel_event.clear()
        self.set_status(0.05, "Loading telemetry asynchronously...")

        async def work():
            rows, flows = await asyncio.gather(
                self.store.fetch_metrics(self.cfg.history_limit),
                self.store.fetch_flows(self.cfg.flow_limit),
            )
            if not rows:
                raise RuntimeError(
                    "network_logs contains no data. Insert measurements before analysis."
                )
            self.set_status_threadsafe(0.40, "Computing robust statistics + PennyLane circuit...")
            analytics = await asyncio.to_thread(self.quantum.analyze, rows, flows)
            return rows, analytics

        def success(result):
            rows, analytics = result
            self.last_rows = rows
            self.last_analytics = analytics
            self._render_telemetry()
            self.set_status(1.0, "Telemetry synchronized.")
            self._set_busy(False)

        def error(exc):
            self._set_busy(False)
            self._show_error("Telemetry refresh", exc)

        self._submit(work(), success, error)

    def _render_telemetry(self):
        if not self.last_analytics:
            return
        c = self.last_analytics["current"]
        self.cards["ping"].set(f"{c['ping']:.1f}", "ms")
        self.cards["jitter"].set(f"{c['jitter']:.1f}", "ms")
        self.cards["download_speed"].set(f"{c['download_speed']:.1f}", "Mbps")
        self.cards["upload_speed"].set(f"{c['upload_speed']:.1f}", "Mbps")
        self.sample_label.configure(text=f"samples: {self.last_analytics['sample_count']}")
        self.anomaly_label.configure(
            text=f"anomalies: {len(self.last_analytics['current_anomalies'])}"
        )

        chronological = self.last_rows[::-1]
        x = list(range(len(chronological)))

        self.ax_latency.clear()
        self.ax_latency.plot(x, [r["ping"] for r in chronological], label="Ping")
        self.ax_latency.plot(x, [r["jitter"] for r in chronological], label="Jitter")
        self.ax_latency.set_ylabel("ms")
        self.ax_latency.grid(True, alpha=0.25)
        self.ax_latency.legend(loc="upper left")

        self.ax_speed.clear()
        self.ax_speed.plot(x, [r["download_speed"] for r in chronological], label="Download")
        self.ax_speed.plot(x, [r["upload_speed"] for r in chronological], label="Upload")
        self.ax_speed.set_ylabel("Mbps")
        self.ax_speed.set_xlabel("oldest → newest sample")
        self.ax_speed.grid(True, alpha=0.25)
        self.ax_speed.legend(loc="upper left")

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

        q = self.last_analytics["quantum"]
        self.ai_meta.delete("1.0", "end")
        self.ai_meta.insert(
            "1.0",
            json.dumps(
                {
                    "current_anomalies": self.last_analytics["current_anomalies"],
                    "quantum_entropy_bits": q["shannon_entropy_bits"],
                    "quantum_saturation": q["saturation"],
                    "quantum_z": q["pauli_z_expectations"],
                },
                indent=2,
            ),
        )

    def setup_weaviate(self):
        self.set_status(0.05, "Creating/verifying Weaviate memory schemas...")
        self._submit(
            self.memory.setup_collections(),
            on_success=lambda r: (
                self.memory_status.delete("1.0", "end"),
                self.memory_status.insert("1.0", json.dumps(r, indent=2)),
                self.set_status(1.0, f"{len(WEAVIATE_SCHEMAS)} Weaviate collections ready."),
            ),
            on_error=lambda e: self._show_error("Weaviate setup", e),
        )

    def refresh_weaviate_status(self):
        self.set_status(0.05, "Checking Weaviate...")
        self._submit(
            self.memory.status(),
            on_success=lambda r: (
                self.memory_status.delete("1.0", "end"),
                self.memory_status.insert("1.0", json.dumps(r, indent=2)),
                self.set_status(1.0, "Weaviate status updated."),
            ),
            on_error=lambda e: self._show_error("Weaviate status", e),
        )

    def sync_current_snapshot(self):
        if not self.last_analytics:
            messagebox.showinfo("No telemetry", "Refresh telemetry first.")
            return

        analytics = self.last_analytics
        props = telemetry_memory_properties(analytics)
        qvec = analytics["quantum"]["probabilities"]

        async def work():
            return await self.memory.insert(
                "QNTelemetryMemoryV1",
                props,
                "telemetry",
                props["event_id"],
                qvec,
            )

        self.set_status(0.10, "Embedding + syncing telemetry snapshot...")
        self._submit(
            work(),
            on_success=lambda r: (
                self.set_status(1.0, f"Telemetry remembered in {r['collection']}."),
                self.refresh_weaviate_status(),
            ),
            on_error=lambda e: self._show_error("Memory sync", e),
        )

    def search_memory(self):
        query = self.memory_query.get().strip()
        if not query:
            return

        qvec = None
        if self.last_analytics:
            qvec = self.last_analytics["quantum"]["probabilities"]

        self.set_status(0.10, "Fusing BM25 + OpenAI + Facebook + quantum memory recall...")
        self._submit(
            self.memory.search(query, qvec, self.cfg.recall_limit),
            on_success=self._render_memory_search,
            on_error=lambda e: self._show_error("Memory search", e),
        )

    def _render_memory_search(self, result):
        self.last_memory_hits = result["results"]
        self.memory_results.delete("1.0", "end")
        self.memory_results.insert("1.0", json.dumps(result, indent=2, default=str))
        self.set_status(1.0, f"Retrieved {len(self.last_memory_hits)} fused memories.")

    def run_analysis(self):
        if self._busy:
            return
        if not self.last_analytics:
            messagebox.showinfo("No telemetry", "Refresh telemetry first.")
            return

        mode = self.mode_menu.get()
        if mode in {"Malware / Traffic Triage", "Bandwidth Consumers"}:
            if not self.last_analytics["flow_analysis"].get("available"):
                messagebox.showwarning(
                    "Flow data required",
                    (
                        "This mode requires network_flows. Throughput/latency alone cannot "
                        "identify malware or individual devices."
                    ),
                )
                return

        self._set_busy(True)
        self.cancel_event.clear()
        self.ai_output.delete("1.0", "end")
        self.protocol_box.delete("1.0", "end")
        self.set_status(0.05, "Building analysis context...")

        async def work():
            memory_hits: list[dict[str, Any]] = []
            memory_errors = []
            if self.cfg.auto_memory_recall:
                try:
                    c = self.last_analytics["current"]
                    query = (
                        f"{mode}; ping {c['ping']} jitter {c['jitter']} "
                        f"download {c['download_speed']} upload {c['upload_speed']} "
                        f"anomalies {self.last_analytics['current_anomalies']}"
                    )
                    recall = await self.memory.search(
                        query,
                        self.last_analytics["quantum"]["probabilities"],
                        self.cfg.recall_limit,
                    )
                    memory_hits = recall["results"]
                    memory_errors = recall["embedding_errors"]
                except Exception as exc:
                    memory_errors = [f"Memory recall unavailable: {exc}"]

            if mode == "Temporal Hyperfield Analysis" and not self.last_hyperfield:
                raise RuntimeError(
                    "Temporal Hyperfield Analysis requires a built worldline. "
                    "Open Temporal Hyperfield and click Build Hyperfield first."
                )

            prompt = build_prompt(
                mode,
                self.last_analytics,
                memory_hits,
                self.last_hyperfield,
            )
            result, fallback_errors = await self.providers.generate(
                prompt,
                self.cancel_event,
                lambda msg: self.set_status_threadsafe(0.55, msg),
                self.provider_menu.get(),
            )
            parsed = parse_protocol(result.text)
            analysis_id = str(uuid.uuid4())

            provider_metadata = {
                "usage": result.usage,
                "metadata": result.metadata,
                "fallback_errors": fallback_errors,
                "memory_errors": memory_errors,
                "memory_hit_count": len(memory_hits),
            }

            await self.store.save_analysis(
                analysis_id,
                mode,
                result.provider,
                result.model,
                prompt,
                result.text,
                parsed.as_dict(),
                self.last_analytics,
                provider_metadata,
            )
            await self.store.save_actions(analysis_id, parsed.actions)

            # Persist this analysis episode into Weaviate when available.
            try:
                props = {
                    "analysis_id": analysis_id,
                    "created_at": utc_now(),
                    "mode": mode,
                    "provider": result.provider,
                    "model": result.model,
                    "confidence": float(parsed.confidence or 0.0),
                    "final_text": parsed.final_text,
                    "observations": "\n\n".join(parsed.blocks.get("observation", [])),
                    "risks": "\n\n".join(parsed.blocks.get("risk", [])),
                    "actions_json": safe_json(parsed.actions),
                    "reply_template": "\n\n".join(parsed.reply_templates),
                    "raw_protocol": result.text,
                    "telemetry_json": safe_json(self.last_analytics["current"]),
                    "quantum_json": safe_json(self.last_analytics["quantum"]),
                    "tags": ["analysis", mode.lower().replace(" ", "-")],
                }
                await self.memory.insert(
                    "QNAnalysisEpisodeMemoryV1",
                    props,
                    "analysis",
                    analysis_id,
                    self.last_analytics["quantum"]["probabilities"],
                )
            except Exception as exc:
                provider_metadata.setdefault("memory_write_errors", []).append(str(exc))

            return analysis_id, result, parsed, memory_hits, provider_metadata

        def success(payload):
            analysis_id, result, parsed, memory_hits, meta = payload
            self.last_analysis_id = analysis_id
            self.last_provider_result = result
            self.last_parsed = parsed
            self.last_memory_hits = memory_hits

            self.ai_output.insert("1.0", protocol_markdown(parsed) or result.text)
            self.ai_meta.delete("1.0", "end")
            self.ai_meta.insert(
                "1.0",
                json.dumps(
                    {
                        "analysis_id": analysis_id,
                        "provider": result.provider,
                        "model": result.model,
                        "elapsed_seconds": round(result.elapsed_seconds, 2),
                        **meta,
                    },
                    indent=2,
                    default=str,
                ),
            )
            self.provider_badge.configure(text=f"{result.provider} • {result.model}")
            self._render_protocol(parsed)
            self.refresh_action_queue()
            self.set_status(1.0, f"Analysis complete via {result.provider}.")
            self._set_busy(False)

        def error(exc):
            self._set_busy(False)
            self._show_error("Analysis", exc)

        self._submit(work(), success, error)

    def _render_protocol(self, parsed: ParsedProtocol):
        self.protocol_box.delete("1.0", "end")
        blocks = []
        for tag in ("status", "observation", "evidence", "risk", "recommendation", "confidence", "final"):
            for value in parsed.blocks.get(tag, []):
                blocks.append(f"[{tag}]\n{value}\n[/{tag}]")
        for template in parsed.reply_templates:
            blocks.append(f"[replytemplate]\n{template}\n[/replytemplate]")
        self.protocol_box.insert("1.0", "\n\n".join(blocks))

    def refresh_action_queue(self):
        self._submit(
            self.store.list_actions(),
            on_success=self._render_action_queue,
            on_error=lambda e: self._show_error("Action queue", e),
        )

    def _render_action_queue(self, rows):
        for item in self.action_tree.get_children():
            self.action_tree.delete(item)
        for row in rows:
            self.action_tree.insert(
                "",
                "end",
                iid=row["action_id"],
                values=(
                    row["priority"],
                    row["risk"],
                    row["approval_state"],
                    row["title"],
                ),
            )

    def set_selected_action(self, state: str):
        selected = self.action_tree.selection()
        if not selected:
            return
        action_id = selected[0]
        self._submit(
            self.store.update_action_state(action_id, state),
            on_success=lambda _: self.refresh_action_queue(),
            on_error=lambda e: self._show_error("Action state", e),
        )

    def launch_codex_login(self):
        try:
            self.providers.codex.launch_login()
            self.status_label.configure(
                text="Official Codex login launched; complete authentication in that flow."
            )
        except Exception as exc:
            self._show_error("Codex login", exc)

    def apply_session_secrets(self):
        self.session_openai_key = self.api_key_entry.get().strip()
        self.session_weaviate_url = self.weaviate_url_entry.get().strip()
        self.session_weaviate_key = self.weaviate_key_entry.get().strip()
        self.status_label.configure(text="Session secrets applied; nothing persisted to config.")

    def download_legacy_model(self):
        self.cancel_event.clear()
        self.set_status(0.0, "Preparing secure legacy model download...")

        async def work():
            await self.legacy_manager.download(
                self.cancel_event,
                lambda value, msg: self.set_status_threadsafe(value, msg),
            )

        self._submit(
            work(),
            on_success=lambda _: self.set_status(1.0, "Legacy model verified."),
            on_error=lambda e: self._show_error("Legacy model", e),
        )

    # ------------------------------------------------------------------
    # Settings / exports
    # ------------------------------------------------------------------

    def choose_database(self):
        path = filedialog.askopenfilename(
            title="Choose SQLite database",
            filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
        )
        if path:
            e = self.setting_entries["database_path"]
            e.delete(0, "end")
            e.insert(0, path)

    def save_settings(self):
        try:
            self.cfg.database_path = self.setting_entries["database_path"].get().strip()
            self.cfg.history_limit = max(20, int(self.setting_entries["history_limit"].get()))
            self.cfg.flow_limit = max(100, int(self.setting_entries["flow_limit"].get()))
            self.cfg.refresh_seconds = max(2, int(self.setting_entries["refresh_seconds"].get()))
            effort = self.setting_entries["reasoning_effort"].get().strip().lower()
            if effort not in {"low", "medium", "high", "xhigh", "max"}:
                raise ValueError("reasoning_effort must be low, medium, high, xhigh, or max")
            self.cfg.reasoning_effort = effort
            self.cfg.recall_limit = max(1, min(50, int(self.setting_entries["recall_limit"].get())))
            self.cfg.rrf_k = max(10, int(self.setting_entries["rrf_k"].get()))
            self.cfg.provider_preference = self.provider_menu.get()
            self.cfg.use_openai_embeddings = bool(self.openai_embed_switch.get())
            self.cfg.use_fb_embeddings = bool(self.fb_embed_switch.get())
            self.cfg.save()

            self.session_weaviate_url = self.weaviate_url_entry.get().strip()
            self.session_weaviate_key = self.weaviate_key_entry.get().strip()

            self.store = SQLiteStore(Path(self.cfg.database_path).expanduser())
            self.embeddings = EmbeddingEnsemble(
                self.store,
                lambda: self.session_openai_key,
                lambda: self.cfg,
            )
            self.memory = WeaviateMemory(
                self.store,
                self.embeddings,
                lambda: self.cfg,
                lambda: self.session_weaviate_url,
                lambda: self.session_weaviate_key,
            )
            self.db_label.configure(text=self.cfg.database_path)

            self._submit(
                self.store.initialize(),
                on_success=lambda _: self.set_status(1.0, "Settings saved; database initialized."),
                on_error=lambda e: self._show_error("Settings", e),
            )
        except Exception as exc:
            self._show_error("Settings", exc)

    def export_csv(self):
        if not self.last_rows:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "timestamp", "ping", "jitter", "download_speed", "upload_speed"
                ],
            )
            writer.writeheader()
            writer.writerows(self.last_rows)
        self.status_label.configure(text=f"CSV exported: {path}")

    def export_report(self):
        if not self.last_analytics:
            return
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        default = f"hyperconsole_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        path = filedialog.asksaveasfilename(
            initialdir=str(REPORT_DIR),
            initialfile=default,
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
        )
        if not path:
            return

        text = (
            f"# {APP_NAME} Report\n\n"
            f"Generated: {utc_now()}\n\n"
            f"## Deterministic analytics\n\n```json\n"
            f"{json.dumps(self.last_analytics, indent=2, default=str)}\n```\n\n"
            f"## Retrieved memory\n\n```json\n"
            f"{json.dumps(self.last_memory_hits, indent=2, default=str)}\n```\n\n"
            f"## AI analysis\n\n"
            f"{protocol_markdown(self.last_parsed) if self.last_parsed else 'No AI analysis.'}\n"
        )
        Path(path).write_text(text, encoding="utf-8")
        self.status_label.configure(text=f"Report exported: {path}")

    # ------------------------------------------------------------------
    # System / lifecycle
    # ------------------------------------------------------------------

    def _system_tick(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self.system_label.configure(
                text=(
                    f"CPU {cpu:.0f}% | RAM {mem.percent:.0f}% | "
                    f"{human_bytes(mem.available)} free | {datetime.now().strftime('%H:%M:%S')}"
                )
            )
        finally:
            self.after(1500, self._system_tick)

    def Continue(self):
        """Chainable compatibility entry point: Continue().run()."""
        return self

    def run(self):
        self.mainloop()
        return self

    def _shutdown(self):
        self.cancel_event.set()
        self.hyperfield_playing = False
        if self.hyperfield_replay_job is not None:
            try:
                self.after_cancel(self.hyperfield_replay_job)
            except Exception:
                pass
            self.hyperfield_replay_job = None
        self.runtime.stop()
        self.destroy()


# =============================================================================
# CLI / Continue().run()
# =============================================================================

_APP_INSTANCE: Optional[QuantumNetworkHyperConsole] = None


def Continue() -> QuantumNetworkHyperConsole:
    global _APP_INSTANCE
    if _APP_INSTANCE is None:
        _APP_INSTANCE = QuantumNetworkHyperConsole()
    return _APP_INSTANCE


def run():
    return Continue().run()


def print_schemas():
    print(json.dumps(WEAVIATE_SCHEMAS, indent=2))


def print_requirements():
    print(
        "Required:\n"
        "  customtkinter\n"
        "  aiosqlite\n"
        "  httpx\n"
        "  numpy\n"
        "  pennylane\n"
        "  matplotlib\n"
        "  psutil\n"
        "  weaviate-client>=4.16.4\n"
        "  torch\n"
        "  transformers\n"
        "  pyarrow  # optional real parquet ingestion\n"
        "  litert-lm  # Gemma 4 E2B .litertlm runtime\n"
        "  kaggle  # optional explicit real dataset download\n\n"
        "Optional legacy local fallback:\n"
        "  llama-cpp-python==0.1.78\n"
    )


async def cli_init_db(path: str):
    store = SQLiteStore(Path(path).expanduser())
    await store.initialize()
    print(f"Initialized: {store.path}")


async def cli_setup_weaviate():
    cfg = AppConfig.load()
    store = SQLiteStore(Path(cfg.database_path).expanduser())
    await store.initialize()
    api_key = os.getenv("OPENAI_API_KEY", "")
    w_url = os.getenv("WEAVIATE_URL", "")
    w_key = os.getenv("WEAVIATE_API_KEY", "")
    embeddings = EmbeddingEnsemble(store, lambda: api_key, lambda: cfg)
    memory = WeaviateMemory(
        store,
        embeddings,
        lambda: cfg,
        lambda: w_url,
        lambda: w_key,
    )
    result = await memory.setup_collections()
    print(json.dumps(result, indent=2))



async def cli_seed_safe_datasets():
    cfg = AppConfig.load()
    store = SQLiteStore(Path(cfg.database_path).expanduser())
    await store.initialize()
    api_key = os.getenv("OPENAI_API_KEY", "")
    w_url = os.getenv("WEAVIATE_URL", "")
    w_key = os.getenv("WEAVIATE_API_KEY", "")
    embeddings = EmbeddingEnsemble(store, lambda: api_key, lambda: cfg)
    memory = WeaviateMemory(
        store,
        embeddings,
        lambda: cfg,
        lambda: w_url,
        lambda: w_key,
    )
    await memory.setup_collections()
    simulator = SafeDatasetSimulator()
    total = 0
    for manifest in CYBER_DATASETS[:5]:
        records = simulator.simulate_manifest(manifest, DATASET_SIM_ROWS_PER_SOURCE)
        for record in records:
            await memory.insert(
                "QNDatasetCorpusMemoryV1",
                record,
                "dataset-sim",
                record["record_id"],
                None,
            )
            total += 1
        print(f"{manifest.name}: {len(records)} simulated records")
    print(f"Seeded {total} simulated records across the top five manifests.")


async def cli_verify_gemma4():
    result = await Gemma4LiteRTManager().verify()
    print(json.dumps(result, indent=2))


async def cli_download_gemma4():
    manager = Gemma4LiteRTManager()
    cancel = threading.Event()
    result = await manager.download(
        cancel,
        lambda value, msg: print(f"[{value:0.3f}] {msg}"),
    )
    print(json.dumps(result, indent=2))


def cli_cqse_demo(text: str):
    encoder = ChromaticQuantumSurfaceEncoder()
    result = encoder.encode(text)
    print(json.dumps(result, indent=2, default=str))

def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--print-schemas", action="store_true")
    parser.add_argument("--print-requirements", action="store_true")
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--setup-weaviate", action="store_true")
    parser.add_argument("--seed-safe-datasets", action="store_true")
    parser.add_argument("--verify-gemma4", action="store_true")
    parser.add_argument("--download-gemma4", action="store_true")
    parser.add_argument("--cqse-demo", type=str, default=None)
    args = parser.parse_args()

    if args.print_schemas:
        print_schemas()
        return
    if args.print_requirements:
        print_requirements()
        return
    if args.init_db:
        asyncio.run(cli_init_db(str(DEFAULT_DB_PATH)))
        return
    if args.setup_weaviate:
        asyncio.run(cli_setup_weaviate())
        return
    if args.seed_safe_datasets:
        asyncio.run(cli_seed_safe_datasets())
        return
    if args.verify_gemma4:
        asyncio.run(cli_verify_gemma4())
        return
    if args.download_gemma4:
        asyncio.run(cli_download_gemma4())
        return
    if args.cqse_demo is not None:
        cli_cqse_demo(args.cqse_demo)
        return

    run()


if __name__ == "__main__":
    main()
