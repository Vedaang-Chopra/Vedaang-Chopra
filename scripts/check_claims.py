"""Fail when the resume data or built page contains a banned claim.

Usage:
    python scripts/check_claims.py [BUILD_DIR]

Scans content/data/resume.yml and BUILD_DIR/resume.html (default: output/) for every
pattern in scripts/banned_patterns.yml (written by sync_resume.py from the fact bank)
plus the regex rules below. Exits 1 on any hit.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
DATA_FILE = REPO / "content" / "data" / "resume.yml"
BANNED_FILE = REPO / "scripts" / "banned_patterns.yml"

# (name, regex). Applied case-insensitively to visible text and data.
RULES = [
    ("personal gmail", r"@gmail\.com"),
    ("ICAIA described as published", r"ICAIA[^.]{0,120}\bpublished\b|\bpublished\b[^.]{0,60}ICAIA"),
    ("patent described as granted", r"patent[^.]{0,80}\bgranted\b"),
    ("GPA other than 3.8", r"(?<!C)GPA[^0-9]{0,12}(?!3\.8)\d\.\d"),
    ("ARTEMIS invalid or banned numbers", r"64\.31|90\.3\s*%"),
    ("CAD as RL or fine-tuning", r"CAD[^.]{0,80}\b(reinforcement|RLVR|fine-tuned)\b"),
    ("Volatility 3", r"Volatility\s?3"),
    ("CERBERUS multi-GPU hardware claim", r"\b(H100|A100)\b"),
    ("CERBERUS tri-modal", r"tri-?modal"),
    ("CERBERUS r=32", r"r\s?=\s?32\b"),
    ("test pass count", r"\b\d[\d,]*\s+(tests\s+)?passing\b"),
]


def strip_tags(html: str) -> str:
    """Return the text content of the page and the values of its data attributes."""
    html = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html))


def scan(label: str, text: str, patterns: list[str]) -> list[str]:
    """Return one message per banned pattern or rule found in text."""
    hits = []
    for pat in patterns:
        if pat.lower() in text.lower():
            hits.append(f"{label}: banned pattern {pat!r}")
    for name, rx in RULES:
        match = next((m for m in re.finditer(rx, text, flags=re.I) if "never" not in m.group(0).lower()), None)
        if match:
            hits.append(f"{label}: rule '{name}' matched {match.group(0)[:60]!r}")
    return hits


def main(argv: list[str]) -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_dir = Path(argv[1]) if len(argv) > 1 else REPO / "output"
    patterns = yaml.safe_load(BANNED_FILE.read_text(encoding="utf-8"))["patterns"]
    targets = {"resume.yml": DATA_FILE.read_text(encoding="utf-8")}
    page = build_dir / "resume.html"
    if not page.exists():
        logger.error("build output not found: %s", page)
        return 2
    raw = page.read_text(encoding="utf-8")
    targets["resume.html text"] = strip_tags(raw)
    targets["resume.html attributes"] = " ".join(re.findall(r'data-[a-z-]+="([^"]*)"', raw))
    hits: list[str] = []
    for label, text in targets.items():
        hits += scan(label, text, patterns)
    for hit in hits:
        logger.error(hit)
    if hits:
        logger.error("%d banned claim hit(s)", len(hits))
        return 1
    logger.info("check_claims: clean (%d patterns, %d rules, %d targets)", len(patterns), len(RULES), len(targets))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
