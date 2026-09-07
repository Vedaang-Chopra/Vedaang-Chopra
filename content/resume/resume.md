Title: Resume
Slug: resume
Save_as: resume.html
Date: 2026-09-07

<div class="resume-header">
    <a href="/resume/Vedaang_Chopra_Prof_CV.pdf" class="btn-pill" target="_blank">Download PDF ⬇</a>
</div>

# Vedaang Chopra
**Applied AI / ML / Research Engineer — Production ML & Agentic AI Systems**  
Atlanta, GA | +1 404-740-9905 | [vedaangchopra@gatech.edu](mailto:vedaangchopra@gatech.edu) | [LinkedIn](https://linkedin.com/in/vedaang-chopra) | [GitHub](https://github.com/Vedaang-Chopra)

---

## Summary

MS Computer Science (Machine Learning) student at Georgia Tech (GPA 4.0/4.0, graduating Dec 2026) with 4+ years of production ML/AI engineering experience at Fortinet and active graduate research in agentic AI systems, LLM evaluation, and cost-aware VLM routing. Published researcher (IEEE ICAIA 2026) and inventor on a filed patent.

---

## Technical Skills

- **Agentic AI & LLM Systems:** LangGraph, LangChain, agentic RAG, tool/function calling, multi-agent orchestration (supervisor-worker), structured generation (Pydantic), LLM-as-a-judge
- **AI/ML Core:** Python (expert), PyTorch, Hugging Face Transformers, Scikit-Learn, CNNs/Vision Transformers, NLP & Computer Vision
- **LLMs / VLMs / Inference:** vLLM serving, VLM routing & evaluation, model benchmarking, inference optimization, CLIP/SBERT/FAISS, Matryoshka Representation Learning, quantization
- **Systems & Infrastructure:** Go, OpenSearch/Elasticsearch, ONNX Runtime, Docker, Linux, Git, SQL/PostgreSQL, Redis, FastAPI, Azure, Slurm, W&B/TensorBoard, PyTorch DDP, mixed precision training
- **RL (academic projects):** PPO, DQN, reward shaping, curriculum learning, self-play (Ray/RLlib)

---

## Experience

**Fortinet Technologies Inc. (AIOps R&D)** | *Software Development Engineer I & II (ML/AI Track)* | Bengaluru, India | Feb 2021 – Jul 2025

*   **Led development of an agentic RAG diagnostics system** (SDE II) — a Fortinet Global Hackathon 2023 entry (top-5 finalist, 5th place) taken to production. The LLM plans multi-step tool calls over network telemetry, verifies hypotheses via structured function calling, and produces autonomous root-cause analysis, reducing mean resolution time by ~70%.
*   **Filed patent PCT/IN2022/058026** — unsupervised distributional thresholding for wireless connectivity anomaly detection (US publication US20240121629A1; filed/pending), cutting manual troubleshooting by ~75%.
*   **Re-architected OpenSearch ingestion pipelines** (complete redesign, async I/O + Golang), scaling throughput from 50 to 2,000 events/sec (40x).
*   **Deployed edge-optimized ML models** via ONNX Runtime on network appliances, achieving ~40% latency reduction.
*   **Built SLA forecasting pipelines** supporting 60+ classifiers across 4 categories (performance, capacity, availability, connectivity) with automated evaluation and retraining, enabling 7-day AI-based performance prediction.
*   **Implemented DBSCAN-based anomaly detection** for SD-WAN telemetry, reportedly preventing over 50% of potential network outages.
*   **Integrated ML models into backend services** with Python and Go across distributed systems and telemetry infrastructure, including CPU-only pickle model serving.

---

## Graduate Research — Georgia Tech

**CORE Robotics Lab @ Siemens — Graduate Research Assistant** (Advisor: Dr. Matthew Gombolay) | Jan 2026 – Present

*   **CAD Code Generation Pipeline:** Building a closed-loop agentic pipeline for parametric CAD code generation — an LLM generates CadQuery programs, an execution environment returns compile and geometry feedback, and a LangGraph-orchestrated repair loop iteratively corrects errors. Model is frozen: no training, no fine-tuning — synthesis is framed as inference-time search with verifiable geometric reward. Verification includes Chamfer/Hausdorff distances on STL meshes, AST/code-graph structural analysis, VLM-based visual judgment, and RAG-augmented prompts across a modular multi-package system with a pytest evaluation harness.

**CS 8903 Special Problems — Graduate Researcher** | Jan 2025 – Present

*   **ARTEMIS — Cost-Aware VLM Routing** (Aug 2025 – Present): Trained a neural multi-task router with SLA-aware load balancing (simulation-validated), serving 5 VLMs across 10 vLLM endpoints (Gemma 3 27B, Qwen3-VL, DeepSeek OCR, and others) over VQA, OCR, captioning, and reasoning tasks with 5 dynamic routing modes; ~340K routing profiles across ~68K queries, with 90.3% oracle-utility recovery in balanced mode.
*   **CERBERUS — Vision-Language Alignment for Edge** (Aug 2025 – Present): Cross-modal alignment with frozen encoders (CLIP/SBERT) using Matryoshka Representation Learning (4096→128 dims, ~78% R@5 at 4096-d on PixMo retrieval); LoRA fine-tuning of Qwen2.5 decoders (PEFT r=32 α=64); PyTorch DDP distributed training with mixed precision on H100/A100 clusters. Reported a negative Perceiver Resampler ablation result.
*   **ATHENA — Multi-Agent Screenplay Generation** (Jan – May 2025, co-authored with Prof. Vijay Madisetti): Built a 5-agent supervisor-worker orchestration system via LangGraph Command routing with reflection-based critique and dynamic plan modification; preliminary corpus evaluation (BLEU-4 ~0.15, CLIPScore ~0.47) across 100 reference videos.
*   **SHASTRA — Agent Workflow Orchestration** (Jan 2026 – Present): Built an orchestration framework with constraint-aware plan search and a component registry, plus a runnable GAIA trace-to-graph pipeline (94 sessions, 4,037 events, 442 graph nodes); executor integration and workflow retrieval not yet implemented.
*   **AI Security (coursework-style):** Adversarial attacks (PGD), embedding poisoning, blind backdoors, model extraction, membership inference on LLMs, watermarking.

---

## Education

**Georgia Institute of Technology** | *M.S. in Computer Science (Machine Learning Specialization)* | Atlanta, GA | Aug 2024 – Dec 2026 (expected)

*   GPA: 4.0/4.0 · Advisor: Dr. Matthew Gombolay, CORE Robotics Lab @ Siemens
*   Coursework: Deep Learning, Deep Reinforcement Learning, ML Security, Large & Vision Language Models, Agentic AI, Systems for AI
*   Research focus: agentic AI systems, LLM evaluation, VLM routing, execution-grounded verification

**Maharaja Surajmal Institute of Technology (GGSIPU)** | *B.Tech. in Information Technology* | New Delhi, India | Aug 2016 – Aug 2020

*   CGPA: 8.8/10.0

---

## Selected Projects

**[Malware Analysis: Memory Forensics with ML](https://github.com/Vedaang-Chopra/Malware_Analysis)** — Published at **IEEE ICAIA 2026** ("Integrating Machine Learning and Memory Forensics for Enhancing Cybersecurity in IoT-Enabled Energy Systems", with Sonika Malik): analyzed 18 memory dumps across 12 ransomware families, developed 100+ YARA rules, automated Volatility3 orchestration, and built an ML classification pipeline. Dataset released on Hugging Face (33 GB analysis + ~470 GB raw memory dumps).

**[RL Soccer](https://github.com/Vedaang-Chopra/RL_Soccer_project)** *(academic project)* — PPO (baseline, reward shaping, curriculum, self-play) and DQN agents for 2v2 soccer using Ray/RLlib. Team project.

**[AI-Security](https://github.com/Vedaang-Chopra/AI-Security)** — Adversarial robustness coursework material: backdoors, poisoning, jailbreaks, and adversarial attacks.

---

## Publications & Patents

*   **Patent (Filed):** PCT/IN2022/058026 — Unsupervised distributional thresholding for wireless connectivity anomaly detection ([Justia](https://patents.justia.com/inventor/vedaang-chopra))
*   **IEEE ICAIA 2026:** "Integrating Machine Learning and Memory Forensics for Enhancing Cybersecurity in IoT-Enabled Energy Systems" (with Sonika Malik)
*   **Manuscript in preparation:** Screenplay generation via multi-agent orchestration (ATHENA, co-authored with Prof. Vijay Madisetti)

---

## Research Interests

1. **Agentic AI Systems** — multi-agent orchestration, tool-use planning, workflow reuse, long-horizon task decomposition
2. **Execution-Grounded Evaluation** — functional correctness beyond token-level metrics (code execution, geometric verification, visual judgment)
3. **Efficient Multimodal Inference** — cost-aware routing, model compression (Matryoshka), edge deployment
4. **Production ML Systems** — reliable deployment, monitoring, and evaluation pipelines at scale
