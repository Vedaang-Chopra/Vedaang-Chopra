Title: Uses / Skills
Slug: uses
Status: published

# Skills — Every Claim, Cited

Skills are grouped by how well I can back them. Strong skills are used in a system I built or shipped; the qualified group is coursework or offline research and says so. Fortinet figures are my own reported numbers.

## 🤖 Agentic AI & LLM Systems

| Skill | Where I used it |
|---|---|
| **LangGraph** | CAD repair loop ([CORE Lab research](resume.html)) · ATHENA Production V1 14-node staged pipeline · SHASTRA orchestration |
| **LangChain** | [Fortinet](resume.html) agentic RAG diagnostics (production) · CAD RAG module |
| **Agentic RAG** | Fortinet — LLM plans multi-step tool calls over telemetry and verifies hypotheses with structured function calling; hackathon prototype to production, ~70% faster mean resolution (self-reported) |
| **Tool / function calling** | Fortinet (structured function calling) · CAD execution feedback · ATHENA research tools |
| **Multi-agent orchestration** | ATHENA (team) — deterministic staged workflow with typed handoffs · SHASTRA (team) |
| **Structured generation (Pydantic)** | ATHENA typed artifact contracts · ARTEMIS serving contract · CAD |
| **LLM / VLM-as-judge** | CAD visual judgment and ATHENA scene evaluation, used as advisory signals, not correctness oracles |

**Stack:** LangGraph, LangChain, Pydantic

## 👁️ Vision-Language Models & Retrieval

| Skill | Where I used it |
|---|---|
| **CLIP / SBERT encoders** | [CERBERUS](https://github.com/Vedaang-Chopra/CEREBRUS) — frozen CLIP ViT-L/14 and MPNet alignment |
| **Cross-modal retrieval** | CERBERUS — R@5 77.7% / 77.5% (image-to-text / text-to-image) at 4096-d on PixMo validation |
| **Matryoshka embeddings** | CERBERUS — nested prefixes down to 128-d (R@5 62.9% / 64.0%) |
| **LoRA fine-tuning** | CERBERUS only — Qwen-2.5 7B, r=64 on q/k/v/o, image-QA SFT; training loss only, no held-out QA metric yet |
| **VLM routing & evaluation** | [ARTEMIS](https://github.com/Vedaang-Chopra/ARTEMIS) — offline research; leakage-corrected 67.7K-question benchmark, oracle 65.4% vs 49.1% best single model, learned router did not beat lookup (negative result) |

**Stack:** PyTorch, Hugging Face Transformers, CLIP, SBERT, PEFT

## ⚙️ Production ML & Inference Systems

| Skill | Where I used it |
|---|---|
| **Inference optimization (ONNX Runtime)** | [Fortinet](resume.html) — quantized CPU-only edge inference on access points and switches, ~40% latency reduction (self-reported) |
| **OpenSearch at scale** | Fortinet — ingestion re-architecture (async I/O + Go), under 50 to 2,000+ events/sec |
| **SLA forecasting** | Fortinet — 60+ classifiers, 4 categories, automated retraining, 7-day horizon |
| **Anomaly detection** | Fortinet — DBSCAN on SD-WAN telemetry · patented unsupervised wireless thresholding |
| **Backend ML integration (Python/Go)** | Fortinet — data-science models into Go and Python services, CPU-only |
| **Serving APIs** | ATHENA — FastAPI with SSE streaming |

**Stack:** Python, Go, ONNX Runtime, OpenSearch, FastAPI, Docker, Redis, PostgreSQL, Azure

## 🔬 Evaluation & Research Rigor

| Skill | Where I used it |
|---|---|
| **Automated evaluation harnesses** | CAD — ~40-metric STL evaluation stack, pytest harness, paired bootstrap confidence intervals |
| **Geometric verification** | CAD — voxel IoU, Chamfer and Hausdorff distances on STL meshes |
| **Ablations and negative results** | CERBERUS Perceiver ablation · CAD feedback matrix and failed pre-registered gate · ARTEMIS router vs lookup |
| **Benchmark construction** | ARTEMIS — leakage audit that voided 3,909 cross-split image hashes |
| **Experiment tracking** | Weights & Biases, TensorBoard |

**Stack:** pytest, Weights & Biases

## 🛡️ Security & Memory Forensics

| Skill | Where I used it |
|---|---|
| **Memory forensics (Volatility 2)** | [Malware_Analysis](https://github.com/Vedaang-Chopra/Malware_Analysis) — plugin orchestration for feature extraction |
| **ML for security** | Malware_Analysis — nine scikit-learn classifiers over 976 memory windows (best 88.74%, exploratory); paper accepted at IEEE ICAIA 2026 · Neural-Malware-Shield GraphSAGE on MalNet-tiny |
| **Adversarial ML (coursework)** | [AI-Security](https://github.com/Vedaang-Chopra/AI-Security) — PGD, embedding poisoning, membership inference |
| **Dataset curation** | Hugging Face Hub — 33 GB analysis outputs plus ~470 GB raw memory dumps |
| **Security auditing** | [Audit_Script_Development](https://github.com/Vedaang-Chopra/Audit_Script_Development) |

**Stack:** Volatility, YARA (vendored rules), scikit-learn, Python

## 🐍 Core Engineering

| Skill | Where I used it |
|---|---|
| **Python** | Every system above · 4.5 years in production at Fortinet |
| **Go** | Fortinet — OpenSearch scaling, backend services |
| **PyTorch** | CERBERUS training · ARTEMIS router · RL Soccer · Neural-Malware-Shield |
| **scikit-learn** | Fortinet classifiers · Malware_Analysis |
| **SQL / PostgreSQL** | ARTEMIS profile store · independent financial-analytics project |

**Stack:** Python, Go, PyTorch, SQL/PostgreSQL, Linux, Git, Bash

## 🎓 Academic (explicitly qualified)

| Skill | Where I used it |
|---|---|
| **PPO / DQN, curriculum learning (Ray/RLlib)** | [RL_Soccer_project](https://github.com/Vedaang-Chopra/RL_Soccer_project) — academic team project, small-N evaluation |
| **Distributed training (DDP, mixed precision)** | Coursework code paths only; no multi-node LLM training |
| **Post-training literature (RLHF, DPO family, reward models)** | Coursework and self-study reading; no hands-on post-training of my own |

---

## 💻 Daily Drivers

*   **Compute:** Georgia Tech HICE GPU cluster
*   **Editor / Terminal:** VS Code · Zsh + tmux
*   **Infra:** Docker, Linux (daily driver), Git, Azure
