Title: Uses / Skills
Slug: uses
Status: published

# Skills — Every Claim, Cited

Every skill below is backed by a real system I built. Evidence links point to my code, repos, or the specific project where it was used. Nothing on this page is aspirational.

## 🤖 Agentic AI & LLM Systems

| Skill | Where I used it |
|---|---|
| **LangGraph** (repair loops, supervisor-worker, planners) | CAD pipeline repair loop ([CORE Lab research](resume.html)) · ATHENA supervisor-worker routing · SHASTRA planner |
| **LangChain** | [Fortinet](resume.html) agentic RAG diagnostics (production) · CAD RAG module |
| **Agentic RAG** | Fortinet agentic RAG diagnostics — LLM plans multi-step tool calls over telemetry, verifies hypotheses via structured function calling; hackathon prototype → production, ~70% mean resolution-time reduction |
| **Tool / function calling** | Fortinet (structured function calling) · CAD execution feedback · ATHENA tools.py |
| **Multi-agent orchestration** | ATHENA — 5 specialized agents + supervisor-worker via LangGraph Command, reflection critique |
| **Structured generation (Pydantic)** | ATHENA (OrchestrationRouter / ReflectionRouter) · SHASTRA · CAD |
| **LLM-as-a-judge** | CAD VLM-based visual judgment · ATHENA reflection loops |
| **Planning & reasoning** | SHASTRA trace-to-graph · CAD multi-step repair · ATHENA |

**Stack:** LangGraph, LangChain, Pydantic

## 👁️ Vision-Language Models & Retrieval

| Skill | Where I used it |
|---|---|
| **vLLM serving** | [ARTEMIS](https://github.com/Vedaang-Chopra/ARTEMIS) — 5 VLMs across 10 endpoints (Gemma 3 27B, Qwen3-VL, Qwen2.5-VL, DeepSeek OCR) · [CERBERUS](https://github.com/Vedaang-Chopra/CEREBRUS) |
| **VLM routing & evaluation** | ARTEMIS — trained neural multi-task router, SLA-aware load balancing (simulation-validated), 5 routing modes, ~340K profiles / ~68K queries, 90.3% oracle-utility recovery (balanced) |
| **CLIP / SBERT / FAISS** | CERBERUS — frozen-encoder alignment + retrieval |
| **Cross-modal retrieval** | CERBERUS — R@5 ~78% on PixMo; 4096→128-dim Matryoshka compression |
| **Matryoshka Representation Learning** | CERBERUS — `src/encoders/mrl.py`, prefix-sliced projector |
| **LoRA fine-tuning of Qwen2.5 decoders** | CERBERUS — PEFT r=32 α=64, Qwen2.5-7B/3B/1.5B on GT HICE cluster |

**Stack:** PyTorch, Hugging Face Transformers, vLLM, CLIP, SBERT, FAISS, PEFT

## ⚙️ Production ML & Inference Systems

| Skill | Where I used it |
|---|---|
| **Inference optimization (ONNX Runtime)** | [Fortinet](resume.html) — edge deployment on network appliances, ~40% latency reduction |
| **OpenSearch / Elasticsearch at scale** | Fortinet — full ingestion re-architecture (async I/O + Golang), 50 → 2,000 events/sec (40x) |
| **SLA forecasting** | Fortinet — 60+ classifiers, 4 categories, automated retraining, 7-day horizon |
| **Anomaly detection** | Fortinet — DBSCAN on SD-WAN telemetry (reportedly prevented >50% of potential outages) · unsupervised wireless thresholding (patent) |
| **Backend ML integration (Python/Go)** | Fortinet — distributed telemetry systems, CPU-only pickle model serving |
| **ML serving APIs** | [ARTEMIS](https://github.com/Vedaang-Chopra/ARTEMIS) — FastAPI inference stack over Postgres-backed profiles |

**Stack:** Python, Go, ONNX Runtime, OpenSearch, FastAPI, Docker, Redis, PostgreSQL, Azure

## 🔬 Evaluation & Research Rigor

| Skill | Where I used it |
|---|---|
| **Automated evaluation harnesses** | CAD — pytest harness, compile + geometric gates on every run |
| **Geometric verification** | CAD — Chamfer/Hausdorff on STL meshes, precision/recall/F1, normal consistency |
| **Ablation study design** | CERBERUS — Perceiver Resampler ablation (honest negative result) · CAD repair-loop components |
| **Cross-modal evaluation** | ATHENA — BLEU/ROUGE-L/METEOR/BERTScore + CLIPScore/SSIM/PSNR on 100 reference videos |
| **Experiment tracking** | CAD + ARTEMIS + CERBERUS — Weights & Biases, TensorBoard |
| **Benchmark construction** | ARTEMIS — 5 evaluation suites (VQA, OCR, captioning, reasoning) · CAD on CADPrompt benchmark |

**Stack:** pytest, Weights & Biases, TensorBoard

## 🛡️ Security & Memory Forensics

| Skill | Where I used it |
|---|---|
| **Memory forensics (Volatility3)** | [Malware_Analysis](https://github.com/Vedaang-Chopra/Malware_Analysis) — automated orchestration of malfind, pslist, vadinfo, yarascan |
| **YARA rule development** | Malware_Analysis — 100+ rules for ransomware family classification |
| **ML for security** | Malware_Analysis — scikit-learn pipeline for malicious-process identification; **published at IEEE ICAIA 2026** |
| **Adversarial robustness** | [AI-Security](https://github.com/Vedaang-Chopra/AI-Security) — PGD attacks, embedding poisoning, blind backdoors, model extraction, membership inference, watermarking (coursework) |
| **Dataset curation at scale** | Hugging Face Hub — 33 GB analysis dataset + ~470 GB raw dumps + 3,384 code files |
| **Security auditing** | [Audit_Script_Development](https://github.com/Vedaang-Chopra/Audit_Script_Development) — automated Linux posture-audit tooling |

**Stack:** Volatility3, YARA, scikit-learn, Python

## 🐍 Core Engineering

| Skill | Where I used it |
|---|---|
| **Python** (expert) | Every system above · 4.5 years production at Fortinet |
| **Go** | Fortinet — OpenSearch scaling, backend services |
| **PyTorch (DDP, mixed precision)** | CERBERUS — distributed training on H100/A100 · ARTEMIS router training |
| **Scikit-learn** | Fortinet 60+ classifiers · Malware_Analysis pipeline |
| **Distributed training** | CERBERUS — PyTorch DDP + mixed precision, GT HICE cluster (H100/A100) |
| **HPC / Slurm** | CAD multi-GPU runs · ARTEMIS/CERBERUS cluster jobs |

**Stack:** Python, Go, PyTorch, SQL/PostgreSQL, Linux, Git, Slurm, Bash, C/C++ (coursework)

## 🎓 Academic (explicitly qualified)

| Skill | Where I used it |
|---|---|
| **PPO / DQN, reward shaping, curriculum learning, self-play (Ray/RLlib)** | [RL_Soccer_project](https://github.com/Vedaang-Chopra/RL_Soccer_project) — 2v2 soccer agents *(academic project, team)* |
| **Post-training literature (RLHF/DPO-family, process-vs-outcome reward models)** | Coursework/self-study notes tied to CAD verifiable-reward design — *literature familiarity only, no hands-on production post-training* |

---

## 💻 Daily Drivers

*   **Compute:** Georgia Tech HPC (Slurm), H100/A100 clusters
*   **Editor / Terminal:** VS Code · Zsh + tmux
*   **Infra:** Docker, Linux (daily driver), Git, Azure
