# Deployment attempt: Streamlit Community Cloud

This folder holds a self-contained copy of `app.py` + a CPU-only `requirements.txt`, prepared for public deployment to Streamlit Community Cloud. **Deployment was not completed** — kept here as a documented record of what was tried and why it was set aside, not as something currently in use. The app itself runs locally; see the main [README](../README.md) for instructions.

## What happened

Two failed deploy attempts, each with a distinct, diagnosed cause:

1. **Wrong dependency file used.** Streamlit Cloud auto-detected multiple dependency files in the repo (`uv.lock`, `requirements.txt`, `pyproject.toml`) and defaulted to `uv.lock` — the local dev lockfile, pointing at a CUDA-specific `torch` build (`torch==2.6.0+cu124`) plus a full set of `nvidia-cu12-*` packages, entirely unnecessary on a GPU-less server. Fixed by isolating `app.py` and a CPU-only `requirements.txt` in this dedicated folder, with no `uv.lock`/`pyproject.toml` present to compete with it.
2. **Silent death during model loading**, confirmed after the dependency fix — the build log cut off mid-operation (`Fetching 2 files: 0%`) with no Python traceback, consistent with a process getting killed from outside Python entirely (an OOM kill, or a platform-enforced startup timeout — both produce this exact symptom). The likely mechanism: loading a 4-bit quantized model transiently requires holding the original, unquantized weights in memory during conversion — for Phi-3-mini, that's roughly 7.6GB in fp16 plus quantization overhead, well above the ~2GB *final* quantized size. Free-tier memory ceilings on this platform likely can't accommodate that peak, even though the eventual model would easily fit once quantized.

## Decision

Rather than continue chasing this specific free platform's memory ceiling, the app runs locally instead. The core functionality (model loading, caching, side-by-side baseline vs. fine-tuned comparison) is fully verified working — see the main README for a demo recording/screenshot.

A viable path to public deployment, if revisited: a platform with a genuinely larger free-tier memory allowance, or adding `low_cpu_mem_usage=True` to `from_pretrained()` to reduce peak memory during loading (untested here, since the decision to stop was made before trying it).
