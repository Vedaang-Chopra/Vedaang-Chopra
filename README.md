# Vedaang Chopra
### Applied AI / ML Research Engineer | M.S. CS (Machine Learning) @ Georgia Tech

[![Website](https://img.shields.io/badge/Website-vedaangchopra.live-2ea44f?style=for-the-badge&logo=google-chrome&logoColor=white)](https://vedaangchopra.live)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vedaang-chopra/)
[![Email](https://img.shields.io/badge/Email-gatech.edu-c14438?style=for-the-badge)](mailto:vedaangchopra@gatech.edu)

This repo is the source of my personal site and blog, built with Pelican. Resume, projects and skills live at [vedaangchopra.live](https://vedaangchopra.live).

## About
4.5 years of production ML at Fortinet (SDE I to II, patent filed), now doing graduate research at Georgia Tech on agentic AI systems, execution-grounded evaluation and vision-language models. Graduating Dec 2026. Paper accepted at IEEE ICAIA 2026.

## Selected work
| Project | Description |
|:---|:---|
| **CAD code generation** | Frozen-LLM CadQuery generation with a bounded LangGraph repair loop and geometric verification (CORE Robotics Lab). |
| [**CERBERUS**](https://github.com/Vedaang-Chopra/CEREBRUS) | Vision-text alignment with Matryoshka embeddings and LoRA QA fine-tuning. |
| [**ARTEMIS**](https://github.com/Vedaang-Chopra/ARTEMIS) | Leakage-corrected VLM routing benchmark, including the negative results. |
| [**Malware Analysis**](https://github.com/Vedaang-Chopra/Malware_Analysis) | Memory forensics with ML for ransomware detection. |

## Run locally
```bash
pip install -r requirements.txt
pelican --autoreload --listen content
```

## Updating the interactive resume

The page at `/resume.html` is generated from `content/data/resume.yml`, which is generated from the fact bank. Do not edit the data file by hand.

1. Edit the fact bank (`profile_info/resume_fact_bank.yaml`). Display wording lives in `scripts/resume_curation.yml`; every number listed under a card's `proof` must exist in the fact bank or the sync fails.
2. Sync: `JOBHUNT_DATA=/path/to/jobhunt-data python scripts/sync_resume.py`. This rewrites `content/data/resume.yml` and `scripts/banned_patterns.yml`, and copies the three PDFs into `content/resume/`.
3. Build and check: `pelican content -s pelicanconf.py -o /tmp/site && python scripts/check_claims.py /tmp/site`. The check fails on any banned pattern from the fact bank plus extra rules (personal Gmail, "published" ICAIA, GPA other than 3.8, and so on).
4. Commit `content/data/`, `content/resume/` and `scripts/`, then push to `main`.

Each project's picture is a self-contained module in `themes/minimalist/static/js/vis/<id>.js` that reads `content/data/visuals/<id>.json` and draws inline SVG or HTML. A module registers with `RSVis.mods['<id>'] = function (el, data, ctx)`. Add a project by adding it to `scripts/resume_curation.yml`, adding its visual module and data, and listing the id in `themes/minimalist/templates/resume.html`.
