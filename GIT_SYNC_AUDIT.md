# GIT-SYNC AUDIT — Portfolio projects vs GitHub + local repos (2026-09-07)

## 1. Portfolio project cards — every URL verified live against the GitHub API

| Project card | GitHub repo | API check | Languages (actual) | Stack on site | Status |
|---|---|---|---|---|---|
| CAD Code Generation Pipeline | (private — links to profile) | n/a | private | LangGraph, CadQuery, Pydantic, VLMs, RAG, pytest | OK, marked private |
| Malware Analysis | Malware_Analysis | OK | Python, Jupyter, YARA, Shell, PowerShell, Batchfile, Tcl, JS/HTML/CSS | Python, Volatility3, YARA, scikit-learn, pandas, Jupyter, Docker, HF Hub | OK |
| ARTEMIS | ARTEMIS | OK | Python, Jupyter, Dockerfile, Shell, HTML | PyTorch, vLLM, FastAPI, PostgreSQL, W&B | OK (langs consistent — SQL/parquet layers are infra) |
| CERBERUS | CEREBRUS | OK | Python, Jupyter, Shell, TeX | PyTorch (DDP, mixed precision), PEFT/LoRA, CLIP, SBERT, Matryoshka | OK — **link FIXED (was dead Edge-Glass URL)** |
| RL Soccer | RL_Soccer_project | OK | Python, Jupyter, TeX, BibTeX, Shell | Ray/RLlib, PPO, DQN, Unity ML-Agents | OK |
| CAD-Physics | CAD-Physics | OK | Python, Jupyter | Python, Jupyter, FEA tooling | ADDED this round (was missing) |
| AI-Security | AI-Security | OK | Python, Jupyter, Shell | Python, Jupyter | OK |
| Audit Script | Audit_Script_Development | OK | Shell | Shell, Python | OK (Python cell count small — "Shell" leads, kept) |
| Machine_Learning | Machine_Learning | OK | (no languages reported — notebooks only) | Python, Jupyter | OK, kept as learning-repo card |
| ~~Edge-Glass~~ | 404 Not Found | DEAD | — | — | REMOVED — repo does not exist publicly; CEREBRUS is the real public repo for that project |
| ~~Disease_Ontology_Project~~ | 2019, stale | OK | Python | — | Removed in round 1 (stale CEUR 2020 work, profile marks OUTDATED) |
| ~~Developer_Reference_Repository~~ | — | n/a | — | — | Removed (not a technical project) |

## 2. Local repos vs remotes (unpushed-work check)

| Local repo | Branch | Unpushed commits | Notes |
|---|---|---|---|
| ATHENA | v1-production | 0 | Clean |
| Malware_Analysis | master | 0 | Only untracked .DS_Store/audit artifacts |
| Shastra | analysis-v2 | **44 unpushed** | Research repo is private (gatech-sysml fork); pushing is the user's call — NOT touched (repo rules: no commits/pushes without explicit ask) |
| Edge Assistant (CERBERUS local) | main | **1 unpushed** | Same — left untouched |
| Which_VLM_Router (ARTEMIS local) | main | **1 unpushed** | Same |

The portfolio site itself: 9 modified/untracked files staged locally, **not committed or pushed** (awaiting approval, per original instruction).

## 3. Research vs general projects — separation

**Research (from profile_info + CORE Lab / CS 8903):** CAD Code Generation Pipeline, CAD-Physics, ARTEMIS, CERBERUS/CEREBRUS, SHASTRA (private, no public card — described on Resume page only), ATHENA (private, Resume page only), Malware Analysis (published IEEE ICAIA 2026 — research + security).

**General / engineering / learning:** RL Soccer (academic coursework), AI-Security (coursework material), Audit Script (early tooling), Machine_Learning (learning repo).

Cards lead with research; general projects follow. SHASTRA and ATHENA have no public repos so they are covered on the Resume page research section instead of linking to dead URLs.
