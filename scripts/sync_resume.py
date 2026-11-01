"""Build content/data/resume.yml from the job-hunt fact bank.

Usage:
    JOBHUNT_DATA=/path/to/jobhunt-data python scripts/sync_resume.py
    python scripts/sync_resume.py --jobhunt-data /path/to/jobhunt-data

Reads profile_info/resume_fact_bank.yaml and profile_info/profile.md, takes the display
content in scripts/resume_curation.yml, and fails if any number or phrase it lists
is missing from the sources. Writes the committed data file, the banned-pattern list
used by check_claims.py, and copies the three resume PDFs into content/resume/.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
CURATION = REPO / "scripts" / "resume_curation.yml"
DATA_OUT = REPO / "content" / "data" / "resume.yml"
BANNED_OUT = REPO / "scripts" / "banned_patterns.yml"
PDF_OUT = REPO / "content" / "resume"
ASSETS = REPO / "scripts" / "resume_assets.yml"
IMAGES = REPO / "content" / "images" / "resume"
MAX_IMAGE_BYTES = 1_500_000


def load_yaml(path: Path) -> dict:
    """Load a YAML file."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def find_fact(bank: dict, fact_id: str) -> dict:
    """Return the overrides_2026_10_29 metric with this id."""
    for item in bank["overrides_2026_10_29"]["new_verified_metrics"]:
        if item.get("id") == fact_id:
            return item
    raise KeyError(f"fact id not in fact bank: {fact_id}")


def normalise(text: str) -> str:
    """Collapse whitespace so multi-line YAML strings compare cleanly."""
    return re.sub(r"\s+", " ", text)


def collect_banned(bank: dict) -> list[str]:
    """Return banned grep patterns from both ban lists in the fact bank."""
    patterns = list(bank.get("banned_tier_1_grep_patterns", []))
    logger.info("fact bank has %d banned_tier_1_grep_patterns", len(patterns))
    return patterns


def verify_proofs(card: dict, haystack: str, fact_text: str) -> list[str]:
    """Return proof strings of this card that appear in neither the fact entry nor the sources."""
    missing = []
    for proof in card.get("proof", []):
        if proof not in fact_text and proof not in haystack:
            missing.append(proof)
    return missing


def build(data_dir: Path) -> tuple[dict, list[str]]:
    """Return the page data and a list of verification errors."""
    info = data_dir / "profile_info"
    bank = load_yaml(info / "resume_fact_bank.yaml")
    profile_text = (info / "profile.md").read_text(encoding="utf-8")
    haystack = normalise((info / "resume_fact_bank.yaml").read_text(encoding="utf-8") + profile_text)
    cur = load_yaml(CURATION)
    errors: list[str] = []

    for phrase in cur.get("facts_check", []):
        if phrase not in haystack:
            errors.append(f"facts_check not found in sources: {phrase!r}")

    for card in cur["cards"]:
        fact_text = ""
        card["tier"] = card.get("tier", "VERIFIED")
        card["source"] = ""
        card["source_file"] = ""
        card["fact_caveats"] = []
        if card.get("fact"):
            try:
                fact = find_fact(bank, card["fact"])
            except KeyError as exc:
                errors.append(f"card {card['id']}: {exc}")
                continue
            fact_text = normalise(str(fact))
            card["tier"] = str(fact.get("tier", "VERIFIED")).split(" (")[0]
            card["source"] = str(fact.get("artifact", ""))
            found = re.search(r"([\w.\-]+\.(?:csv|json|ipynb|md))", card["source"])
            card["source_file"] = found.group(1) if found else ""
            card["fact_caveats"] = list(fact.get("caveats", []))
        for miss in verify_proofs(card, haystack, fact_text):
            errors.append(f"card {card['id']}: not in sources: {miss!r}")

    pdf_names = {p["id"]: p["file"] for p in cur["pdfs"]}
    for lens in cur["lenses"]:
        lens["pdf_file"] = pdf_names[lens["pdf"]]

    profile_stamp = re.search(r"\*\*Last Updated:\*\*\s*([A-Za-z]+ \d+, \d{4})", profile_text)
    cur["meta"]["profile_updated"] = profile_stamp.group(1) if profile_stamp else "unknown"
    cur["meta"]["email"] = bank["candidate"]["email"]
    cur["meta"]["linkedin"] = bank["candidate"]["linkedin"]
    cur["meta"]["github"] = bank["candidate"]["github"]
    cur.pop("facts_check", None)
    attach_figures(cur, errors)
    return cur, errors


def attach_figures(data: dict, errors: list[str]) -> None:
    """Attach curated figures from resume_assets.yml to their projects."""
    if not ASSETS.exists():
        return
    assets = {a["id"]: a for a in load_yaml(ASSETS)["assets"]}
    projects = {p["id"]: p for p in data["projects"]}
    total = 0
    for asset in assets.values():
        path = IMAGES / asset["file"]
        if not path.exists():
            errors.append(f"asset file missing: {asset['file']}")
            continue
        total += path.stat().st_size
        if asset["project"] not in projects:
            errors.append(f"asset {asset['id']}: unknown project {asset['project']}")
    if total > MAX_IMAGE_BYTES:
        errors.append(f"images weigh {total} bytes, limit {MAX_IMAGE_BYTES}")
    for asset in assets.values():
        partner = assets.get(asset.get("compare_with") or "")
        if partner is not None and asset["id"] > partner["id"]:
            continue
        fig = {"id": asset["id"], "file": asset["file"], "alt": asset["alt"], "caption": asset["caption"],
               "tier": asset["tier"], "claim": asset["claim"], "source_file": Path(asset["source"]).name}
        if partner is not None:
            fig.update({"compare_file": partner["file"], "compare_alt": partner["alt"]})
        projects[asset["project"]].setdefault("figures", []).append(fig)
    logger.info("attached %d figures, %d bytes", len(assets), total)


def copy_pdfs(data_dir: Path, pdfs: list[dict]) -> None:
    """Copy the resume PDFs to content/resume/ under stable names."""
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        src = data_dir / pdf["src"]
        if not src.exists():
            raise FileNotFoundError(f"missing PDF: {src}")
        shutil.copyfile(src, PDF_OUT / pdf["file"])
        logger.info("copied %s", pdf["file"])


def write_outputs(data: dict, banned: list[str]) -> None:
    """Write resume.yml and banned_patterns.yml."""
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    header = "# Generated by scripts/sync_resume.py. Do not edit by hand.\n"
    DATA_OUT.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8")
    BANNED_OUT.write_text(
        "# Generated by scripts/sync_resume.py from banned_tier_1_grep_patterns.\n"
        + yaml.safe_dump({"patterns": banned}, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobhunt-data", default=os.environ.get("JOBHUNT_DATA"))
    args = parser.parse_args()
    if not args.jobhunt_data:
        logger.error("set JOBHUNT_DATA or pass --jobhunt-data")
        return 2
    data_dir = Path(args.jobhunt_data).expanduser()
    data, errors = build(data_dir)
    if errors:
        for err in errors:
            logger.error(err)
        logger.error("%d verification errors, nothing written", len(errors))
        return 1
    bank = load_yaml(data_dir / "profile_info" / "resume_fact_bank.yaml")
    write_outputs(data, collect_banned(bank))
    copy_pdfs(data_dir, data["pdfs"])
    logger.info("wrote %s", DATA_OUT.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
