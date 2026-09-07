# CHANGES.md — Portfolio Site Factual Sweep (2026-09-07)

Sources: `jobhunt-data/profile_info/profile.md`, `resume_fact_bank.yaml`, `accomplishments/accomplishments.md`, `projects/CLAIM_AUDITS_2026-08-22.md` (overrides stale SAFE rows), `publications_patents/publications_patents.md`, `skills/skills.md`, `resume_custom/evidence/EVIDENCE_LIBRARY.md`.

## File: content/resume/resume.md (Resume page — full rewrite)

| Old claim | New claim | Source / rationale |
|---|---|---|
| Tagline "Software Development Engineer (ML/AI)" | "Applied AI Researcher — Production ML & Agentic AI Systems" | profile.md Executive Summary / Target Positioning |
| Skills lines incl. TensorFlow, OpenCV, PyTorch Geometric, Diffusers, RNNs/GNNs, Knowledge Distillation, Pruning, OpenVINO, MLflow, Kubernetes, Airflow, RabbitMQ, C/C++, Distributed Training | Skills rebuilt from profile.md "Technical Skills (Verified)" only | profile.md skills tables; Kubernetes/Airflow/MLflow/JAX explicitly UNVERIFIED |
| "Agentic RAG chatbot … reducing issue-resolution time by over 70%" | "agentic RAG diagnostics system … reducing mean resolution time by ~70%" | accomplishments.md: ~70% is CAUTION/self-reported → must use "~"; "diagnostics system" is canonical wording |
| Patent "(PCT/172022/958026)" (wrong number) | "PCT/IN2022/058026 — (US publication US20240121629A1; filed/pending)" | profile.md, publications_patents.md; old number was malformed; never "Granted" |
| "cutting manual troubleshooting efforts by more than 75%" | "~75%" (in patent bullet) | accomplishments.md: CAUTION, always "~" |
| DBSCAN "proactively preventing over 50%" | "reportedly preventing over 50%" | fact_bank fort_dbscan_sdwan: PARTIAL, qualify with "reportedly" |
| "Which-VLM Router … Expected to achieve 30% lower inference cost" | ARTEMIS bullet: "simulation-validated", "5 VLMs across 10 vLLM endpoints", "~340K routing profiles across ~68K queries", "90.3% oracle-utility recovery (balanced mode)" | CLAIM_AUDITS: "~30% lower cost" FALSE; "6+ backends" FALSE; 100K+ understated; oracle-utility is strongest defensible metric |
| "Edge Glass Assistant … Aims to deliver 2× faster inference" | CERBERUS bullet: MRL 4096→128, ~78% R@5 @4096-d PixMo, LoRA Qwen2.5 (PEFT r=32 α=64), DDP H100/A100, negative Perceiver ablation | CLAIM_AUDITS: MRL ~77.7% R@5 @4096-d; "2× faster" unsourced; LoRA ban lifted only for CERBERUS Qwen |
| ATHENA "BLEU-4 +18%, CLIPScore +22%" | ATHENA bullet: 5-agent supervisor-worker via LangGraph Command; preliminary corpus metrics "BLEU-4 ~0.15, CLIPScore ~0.47" | CLAIM_AUDITS: +18%/+22% banned; absolute 0.154/0.471 persisted → citable as "preliminary" |
| ATHENA "screenplay-to-video generation, memory orchestration" | "multi-agent screenplay generation … reflection-based critique and dynamic plan modification" | "Keyframe synthesis"-adjacent claims banned; audit wording |
| Ontology project 2019–2020 "Improved accuracy by 10%" | Removed from resume page (stale CEUR 2020 paper — 6 years old, different domain) | profile.md marks it OUTDATED; fact_bank include_on_1page: false |
| No CAD/SHASTRA/AI-Security content | Added CAD pipeline (frozen model, inference-time search, Chamfer/Hausdorff on STL meshes, "modular multi-package system", pytest harness — no "16-module"/"202-test"), SHASTRA (honest scoping: executor/retrieval not implemented), AI Security coursework line | CLAIM_AUDITS (16-module/202-test are stale/FALSE; SHASTRA LangGraph/Pydantic FALSE → generic wording) |
| Missing "Team project" note on RL Soccer | RL Soccer added with "(academic project)" + team note | fact_bank rl_soccer critical_note |
| No Summary/Publications sections | Added Summary (profile.md exec summary) and Publications & Patents section (patent Filed, IEEE ICAIA 2026 published, ATHENA manuscript "in preparation") | profile.md, publications_patents.md |
| LinkedIn link trailing variant | linkedin.com/in/vedaang-chopra (canonical) | fact_bank candidate |

## File: content/pages/uses.md (Skills page — full rewrite)

| Old claim | New claim | Source |
|---|---|---|
| "Computer Vision — OpenCV, PyTorch Geometric, Diffusers, CNNs" | VLM & Retrieval section: PyTorch, HF Transformers, vLLM, CLIP/SBERT/FAISS, MRL | profile.md verified skills only |
| "MLOps & Systems — Docker, Kubernetes, FastAPI…" | Kubernetes removed; Docker/FastAPI/ONNX/Redis kept | Kubernetes UNVERIFIED (fact_bank) |
| "Agentic RAG Chatbot — Reduced issue-resolution time by 70%" | "~70%" with "~" qualifier, canonical "diagnostics" name | accomplishments.md CAUTION rule |
| "Which-VLM Router — Semantic router for VLM endpoints" | ARTEMIS with simulation-validated + 5 VLMs/10 endpoints | CLAIM_AUDITS |
| "Ontology-based Text Classification — Improved accuracy by 10%" | Removed; replaced with Mutual Fund Data Ingestion Platform (verified in EVIDENCE_LIBRARY) | EVIDENCE_LIBRARY block |
| ATHENA "Multi-agent screenplay-to-video generation" | Replaced by CAD/ARTEMIS/CERBERUS evidence entries | verified blocks |
| Added "Currently Studying (coursework/self-study)" section | RLHF/DPO-family literature with explicit qualification | fact_bank drl_pt_notes rule: Skills-line only, always qualified |

## File: content/projects/projects.yml

| Old | New | Source |
|---|---|---|
| Edge-Glass: "quantized projectors for on-device inference" | CERBERUS canonical description (frozen encoders, MRL 4096→128, LoRA Qwen2.5, DDP H100/A100, negative ablation) | EVIDENCE_LIBRARY CERBERUS block |
| (missing) | Added ARTEMIS (Which-VLM-Router) card | repo exists on GitHub; verified metrics |
| (missing) | Added RL_Soccer_project card, qualified "(academic project, team)" | fact_bank |
| AI-Security description generic | Expanded with verified coursework topics (PGD, poisoning, model extraction, membership inference, watermarking) | profile.md AI Security row |
| Disease_Ontology_Project | Removed (stale CEUR 2020 work; not current positioning) | profile.md OUTDATED marker |
| Developer_Reference_Repository | Removed (not a technical project; not in profile_info) | sweep rule: not in profile_info |

## File: themes/minimalist/templates/home.html

| Old | New | Source |
|---|---|---|
| `mailto:vedaangchopra1009@gmail.com` button | Removed — only gatech.edu remains | User rule: GT email everywhere, never personal Gmail |
| "AI Research Engineer, crafting intelligent systems…" | "Applied AI Researcher — building production ML systems and agentic AI." | profile.md Target Positioning |
| "summer 2025 internships" | "full-time roles in AI/ML engineering and applied AI research starting 2027" | Graduating Dec 2026 → stale internship line |
| Generic intro text | Verified positioning: MS CS ML GT 4.0 GPA Dec 2026, 4+ yrs Fortinet, IEEE ICAIA 2026, filed patent | profile.md Executive Summary |

## File: themes/minimalist/templates/base.html

| Old | New | Source |
|---|---|---|
| Footer `mailto:vedaangchopra1009@gmail.com` | `mailto:vedaangchopra@gatech.edu` | User rule: GT email everywhere |

## File: pelicanconf.py

| Old | New | Source |
|---|---|---|
| SOCIAL LinkedIn = '#' placeholder | https://linkedin.com/in/vedaang-chopra | fact_bank candidate.linkedin |

## Round 2 (2026-09-07, full-site pass)

| File | Old | New | Source |
|---|---|---|---|
| home.html | "seeking full-time roles… starting 2027" | "actively looking for open AI / ML / Research Engineer roles — graduating December 2026" | User request + profile.md graduation date |
| resume.md | SLA bullet (no horizon) | + "enabling 7-day AI-based performance prediction" | fortinet.md metrics registry (SAFE) |
| resume.md | GT education single line | + Advisor line + verified coursework list | education.md verified wording |
| resume.md | (absent) | Added Research Interests section (4 areas) | profile.md Research Interests (dropped "RL for Structured Generation" — RL framing rule) |
| resume.md | "Applied AI Researcher —" tagline | "Applied AI / ML / Research Engineer —" | User request (role naming) |
| projects.yml | (absent) | Added CAD Code Generation Pipeline card (private-repo note) | EVIDENCE_LIBRARY CAD block; CAD repo is private (fact_bank) |
| projects.html template | Bare "Projects" heading | Added honest subtitle | User request ("other things need updating") |
| index.html template | title "Blog - …"; subtitle "…+ life." | title "Writing - …"; subtitle "Paper walkthroughs and technical deep dives — VLMs, LLM systems, and AI security." | Site nav says Writing; actual content is paper reviews |
| resume.md | "Applied AI Researcher" tagline only | aligned with home page role naming | User request |

## Not changed (verified MATCH)
- Identity block: name, phone, Atlanta GA, GitHub URL, GT email (resume page) — fact_bank.
- Fortinet dates/roles: SDE I & II, Feb 2021 – Jul 2025, Bengaluru — profile.md.
- OpenSearch 50→2,000 events/sec (40x), 60+ classifiers, ONNX ~40% — accomplishments.md SAFE/CAUTION.
- Education rows and GPAs (4.0/4.0, 8.8/10.0) — fact_bank VERIFIED.
- Blog/writing content (Molmo, paper reviews, malware dataset post) — not touched; the malware dataset post's numbers (18 dumps, 12 families, 100+ YARA, ~470 GB, 33 GB) all match accomplishments.md SAFE rows.
- Theme, layout, build config otherwise preserved.
