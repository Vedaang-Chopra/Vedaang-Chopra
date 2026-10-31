Title: Resume
Slug: resume
Save_as: resume.html
Date: 2026-10-31

<div class="resume-header">
    <a href="/resume/Vedaang_Chopra_Prof_CV.pdf" class="btn-pill" target="_blank">Download PDF ⬇</a>
</div>

# Vedaang Chopra
**Applied AI / ML / Research Engineer — Production ML & Agentic AI Systems**  
Atlanta, GA | +1 404-740-9905 | [vedaangchopra@gatech.edu](mailto:vedaangchopra@gatech.edu) | [LinkedIn](https://linkedin.com/in/vedaang-chopra) | [GitHub](https://github.com/Vedaang-Chopra)

---

## Summary

Production ML engineer turned researcher. M.S. in Computer Science (Machine Learning) at Georgia Tech (GPA 3.8/4.0, graduating Dec 2026), after 4.5 years at Fortinet where I was promoted from SDE I to SDE II and filed a patent. Current research covers inference-time search with verifiable geometric signals on a frozen LLM, evaluation-harness design, vision-language retrieval, VLM routing benchmarks, and multi-agent orchestration. Paper accepted at IEEE ICAIA 2026.

---

## Experience

**Georgia Institute of Technology — CORE Robotics Lab @ Siemens** | *Graduate Research Assistant* (Advisor: Dr. Matthew Gombolay) | Atlanta, GA | Jan 2026 – Present

*   **CAD code generation with a frozen LLM.** The LLM writes CadQuery, the program is executed and compared against the ground-truth part, and a bounded LangGraph repair loop (up to 3 turns) corrects errors. The model is frozen: no training or fine-tuning, only inference-time search with a verifiable geometric signal.
*   **Results (n=200 per group):** mean voxel IoU rises from 0.82 to 0.92 for expert prompts and from 0.57 to 0.70 for beginner prompts versus single-shot generation, and by about +0.05 over a self-debug baseline (paired bootstrap).
*   **Scope and negatives, stated plainly:** repair is ground-truth-assisted and no ground-truth-free result exists yet; a 36-run feedback-matrix comparison tied across arms; a pre-registered structural-distance gate failed. Most code was written by AI coding agents from my specifications, and I supervised it.

**Fortinet Technologies Inc. (AIOps R&D)** | *Software Development Engineer I & II (ML/AI Track)* | Bengaluru, India | Feb 2021 – Jul 2025

*   **Agentic RAG diagnostics system** (SDE II): Fortinet Global Hackathon 2023 top-5 finalist, taken to production. An LLM plans multi-step tool calls over network telemetry and verifies hypotheses with structured function calling to produce root-cause analysis; ~70% faster mean resolution (self-reported).
*   **Patent filed, PCT/IN2022/058026** (US publication US20240121629A1, pending): unsupervised distributional thresholding for wireless connectivity anomaly detection; ~75% less manual troubleshooting (self-reported).
*   **OpenSearch ingestion re-architecture** (async I/O + Go): throughput from under 50 to 2,000+ events/sec (40x).
*   **Edge inference:** ONNX Runtime with quantization on CPU-only access points and switches; ~40% latency reduction (self-reported).
*   **SLA forecasting pipelines:** 60+ classifiers across 4 categories with automated evaluation and retraining, giving a 7-day performance forecast.
*   **DBSCAN anomaly detection on SD-WAN telemetry:** reportedly prevented over 50% of potential outages.
*   **Backend integration:** deployed data-science team models into Go and Python services, CPU-only.

---

## Research Projects (Georgia Tech)

*   **CERBERUS — vision-language alignment** (Aug 2025 – Present): frozen CLIP ViT-L/14 and MPNet encoders with Matryoshka embeddings; image-to-text / text-to-image R@5 of 77.7% / 77.5% at 4096-d and 62.9% / 64.0% at 128-d on PixMo validation. LoRA SFT (r=64) of Qwen-2.5 7B for image QA, training loss 0.742 to 0.103; no held-out QA metric yet. A Perceiver ablation is reported as a negative result.
*   **ARTEMIS — VLM routing benchmark** (Aug 2025 – Present, offline research): leakage-corrected benchmark of 67.7K questions and 338K responses over five models. On the test set an oracle router reaches 65.4% versus 49.1% for the best single model; the learned router (50.9%) did not beat a per-task lookup (56.4%), reported as a negative result.
*   **ATHENA — storyboard generation** (team): authored Production V1, a 14-node deterministic LangGraph pipeline with typed contracts and FastAPI/NiceGUI/CLI front ends (~800 tests collected). Release candidate with known open issues. Manuscript in preparation.
*   **SHASTRA — agent trace modeling** (team, CS 8903, advisor Dr. Alexey Tumanov): built the analysis stack that turns agent traces into labelled blocks and task-dependency graphs. Early stage, with no reuse findings yet.

---

## Education

**Georgia Institute of Technology** | *M.S. in Computer Science (Machine Learning Specialization)* | Atlanta, GA | Aug 2024 – Dec 2026 (expected)

*   GPA: 3.8/4.0
*   Coursework: Deep Learning, Deep Reinforcement Learning, ML Security, Large & Vision Language Models, Agentic AI, Systems for AI

**Maharaja Surajmal Institute of Technology (GGSIPU)** | *B.Tech. in Information Technology* | New Delhi, India | Aug 2016 – Aug 2020

*   CGPA: 8.8/10.0

---

## Other Projects

**[Malware Analysis: Memory Forensics with ML](https://github.com/Vedaang-Chopra/Malware_Analysis)** — paper accepted at **IEEE ICAIA 2026** (with Sonika Malik). Volatility 2 features with vendored YARA rules over 976 memory windows; best of nine classifiers was Gradient Boosting at 88.74% accuracy (exploratory, with a leakage caveat). Dataset on Hugging Face.

**Neural-Malware-Shield** *(team of 4, coursework)* — GraphSAGE on MalNet-tiny, 0.781 accuracy / 0.777 macro-F1; I owned the GNN work.

**[RL Soccer](https://github.com/Vedaang-Chopra/RL_Soccer_project)** *(academic team project)* — PPO curriculum over 30M steps for 2v2 soccer with Ray/RLlib; 10-episode evaluations, small-N.

**[AI-Security](https://github.com/Vedaang-Chopra/AI-Security)** *(CS 8803 coursework)* — PGD attacks, embedding poisoning, and membership inference on an LLM.

---

## Publications & Patents

*   **IEEE ICAIA 2026 (accepted, presenting Nov 16, 2026):** "Integrating Machine Learning and Memory Forensics for Enhancing Cybersecurity in IoT-Enabled Energy Systems" (with Sonika Malik)
*   **Patent (filed, pending):** PCT/IN2022/058026 — Unsupervised distributional thresholding for wireless connectivity anomaly detection ([Justia](https://patents.justia.com/inventor/vedaang-chopra))
*   **Manuscript in preparation:** ATHENA, multi-agent storyboard generation

---

## Research Interests

1. **Agentic AI Systems** — multi-agent orchestration, tool-use planning, workflow reuse, long-horizon task decomposition
2. **Execution-Grounded Evaluation** — functional correctness (code execution, geometry, visual judgment) over token-level metrics
3. **Efficient Multimodal Inference** — cost-aware routing, Matryoshka embeddings, edge deployment
4. **Verifiable-Reward Structured Generation** — inference-time search on frozen models
5. **Production ML Systems** — reliable deployment, monitoring, and evaluation pipelines
