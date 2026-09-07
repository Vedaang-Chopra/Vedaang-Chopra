# REMOVED.md — Claims deleted from the portfolio site (2026-09-07)

Every removal was unsourced, stale, or on a banned list in `resume_fact_bank.yaml` / `CLAIM_AUDITS_2026-08-22.md`.

## Banned / contradicted claims (Tier-1)

| Removed claim | Where | Why |
|---|---|---|
| "BLEU-4 +18%" (ATHENA) | resume.md | Comparative figure does not exist anywhere (CLAIM_AUDITS #3 UNVERIFIABLE). Banned in fact_bank grep patterns. |
| "CLIPScore +22%" (ATHENA) | resume.md | Banned; no computed relative value exists. |
| "~30% lower inference cost" / "Expected to achieve 30% lower cost" (ARTEMIS) | resume.md | FALSE — recomputation showed +591%/+385% MORE expensive than CascadeFlow. |
| "6+ VLM backends" (ARTEMIS) | resume.md | FALSE as worded — exactly 5 distinct models × 2 endpoints. |
| "100K+ query profiles" (never present here, kept out) | — | Banned; actual ~340K profiles across ~68K queries (used instead). |
| "RouteLLM comparison / outperforming RouteLLM" | — | Banned — citation only, zero comparisons. |
| "16-module system" / "202 automated tests" (CAD) | — (kept out of new content) | CLAIM_AUDITS: stale — 12 top-level packages; pytest collects 188 tests with 97 collection errors. Replaced with "modular multi-package system" / "pytest evaluation harness". |
| "RLHF / RLAIF / DPO / LoRA-QLoRA (unqualified)" | resume.md skills | Zero code evidence; LoRA allowed ONLY scoped to CERBERUS Qwen2.5 (PEFT r=32 α=64), which is how it now appears. |
| "Keyframe synthesis" (ATHENA adjacent) | resume.md | ATHENA generates screenplays, not keyframes. |
| "Patent Granted"-style phrasing | — | Never used; always "Patent (Filed)". |

## Unverified skills (no code evidence)

| Removed skill | Where | Why |
|---|---|---|
| Kubernetes | resume.md skills, uses.md | GitHub-README-only; fact_bank UNVERIFIED. |
| MLflow | resume.md | No evidence (W&B used instead). |
| Airflow / Prefect / RabbitMQ | resume.md | No orchestration-framework code. |
| OpenVINO, Knowledge Distillation, Pruning (as experience) | resume.md | Not in profile_info verified skills. |
| TensorFlow, PyTorch Geometric, Diffusers, RNNs/LSTMs, GNNs, Generative AI (unqualified) | resume.md | Not in verified skills tables. |
| JAX / DeepSpeed / FSDP / Megatron / TensorRT / Triton | — | Banned/UNVERIFIED. |

## Stale content

| Removed item | Where | Why |
|---|---|---|
| "Looking for summer 2025 internships" | home.html | Stale by 1+ year; user graduates Dec 2026. |
| Personal Gmail (`vedaangchopra1009@gmail.com`) buttons/links | home.html, base.html | Standing rule: GT email only, never personal Gmail. |
| Which-VLM Router "2× faster inference" aim (Edge Glass) | resume.md | Unsourced aspirational claim; no artifact. |
| "Aims to deliver…" language generally | resume.md | Aspirational, not verified. |
| Disease_Ontology_Project (CEUR 2020) project card + resume entry | projects.yml, resume.md | profile.md marks OUTDATED (6 years old, different domain); fact_bank include_on_1page: false. CEUR paper itself not claimed anywhere now. |
| Developer_Reference_Repository card | projects.yml | Not in profile_info; not a technical project. |
| Malformed patent number "PCT/172022/958026" | resume.md | Replaced with verified PCT/IN2022/058026. |
| "which-vlm-router" advisor "Dr. Anand Iyer" / Edge-Glass "Dr. Zsolt Kira, Dr. Alan Ritter" attributions | resume.md | Not present in profile_info sources; removed rather than risk unsourced attribution. |
