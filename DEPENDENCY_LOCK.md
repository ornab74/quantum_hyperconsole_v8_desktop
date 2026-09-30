# Dependency Lock / Security Baseline

Reviewed: 2026-09-29

This package now uses exact direct dependency pins instead of broad ranges.

## Core decisions

- `torch==2.14.0`
- `transformers==5.17.0`
- `safetensors==0.8.0`
- `numpy==2.4.6` to retain Python 3.11 + 3.12 support.
- `httpx==0.28.1`; the 1.0 line is still development/pre-release.
- `customtkinter==5.2.2` is intentionally retained on the desktop build rather
  than jumping to the 6.0.0 major API without a complete GUI regression run.
- `pennylane==0.45.1`
- `weaviate-client==4.23.1`
- `matplotlib==3.11.2`
- `psutil==7.2.2`
- `aiosqlite==0.22.1`

## Contriever supply-chain policy

Repository:
`facebook/contriever-msmarco`

Pinned revision:
`287db3b2d7662e688c051c351a0739de4ef2d6d7`

The upstream `main` branch contains `pytorch_model.bin`, which is a pickle-based
PyTorch serialization artifact. This package instead pins the immutable
safetensors conversion revision above and forces:

- `trust_remote_code=False`
- `use_safetensors=True`

Expected safetensors LFS object SHA-256:
`185c2c743bbb9ec15727575a931dd95fdf035e1fc377dd846a9a4f9fa89dd54b`

No fallback to the upstream pickle weight is intentionally configured.

## Meaning of "locked"

The files in this package exact-pin all direct application dependencies and
optional feature dependencies.

This is not presented as a complete platform-specific transitive wheel hash
lock, because those hashes differ by Python version, operating system, CPU
architecture, and PyTorch wheel index. For maximum reproducibility in CI,
generate a platform-specific `--require-hashes` lock from these exact direct
pins and retain it alongside your built image.

Always run:

```bash
python -m pip check
```

after installation.
