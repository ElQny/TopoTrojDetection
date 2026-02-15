#!/usr/bin/env python3
"""
validate_topotroj_db.py

Quick validator for TopoTrojDetection-style generated datasets.

Usage:
    python validate_topotroj_db.py --data_root ./data/data2
    python validate_topotroj_db.py --data_root ./data --strict
"""

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ID_DIR_RE = re.compile(r"^id-\d{8}$")


@dataclass
class ValidationIssue:
    level: str   # "ERROR" | "WARN" | "INFO"
    model_dir: str
    message: str


@dataclass
class ModelReport:
    model_dir: Path
    issues: List[ValidationIssue] = field(default_factory=list)

    def add(self, level: str, msg: str):
        self.issues.append(ValidationIssue(level, str(self.model_dir), msg))

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "ERROR")

    @property
    def warn_count(self) -> int:
        return sum(1 for i in self.issues if i.level == "WARN")


def find_model_dirs(data_root: Path) -> List[Path]:
    # Primary convention: id-00000000 folders directly under data_root
    direct = [p for p in data_root.iterdir() if p.is_dir() and ID_DIR_RE.match(p.name)]
    if direct:
        return sorted(direct)

    # Fallback: recursively find id-* if user points to higher level folder
    recursive = [p for p in data_root.rglob("*") if p.is_dir() and ID_DIR_RE.match(p.name)]
    return sorted(recursive)


def is_probable_model_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".bin", ".h5"}


def read_text_file(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def parse_possible_gt_from_text(text: str) -> Optional[str]:
    """
    Heuristic:
    - If file content is single token in {0,1,clean,trojaned,trojan,true,false}, treat as GT value.
    - If content includes key-value form, attempt extracting filename/path after ':' or '='.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None

    low = s.lower()
    if low in {"0", "1", "clean", "trojaned", "trojan", "true", "false"}:
        return s

    # key=value / key: value
    for sep in ("=", ":"):
        if sep in s:
            rhs = s.split(sep, 1)[1].strip()
            if rhs:
                return rhs

    # maybe first token is a path
    first = s.splitlines()[0].strip()
    return first if first else None


def scan_for_metadata_files(model_dir: Path) -> Dict[str, Path]:
    candidates = {}
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n in {"gt.txt", "ground_truth.txt", "metadata.json", "config.json", "model_info.json"}:
            candidates[n] = p
        if n.endswith(".csv") and ("experiment" in n or "train" in n or "test" in n):
            candidates[n] = p
    return candidates


def validate_csv_basic(csv_path: Path) -> Tuple[bool, str]:
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = 0
            max_cols = 0
            for row in reader:
                rows += 1
                max_cols = max(max_cols, len(row))
            if rows == 0:
                return False, "empty CSV"
            if max_cols == 0:
                return False, "CSV has no columns"
        return True, f"rows={rows}"
    except Exception as e:
        return False, f"CSV read error: {e}"


def validate_model_dir(model_dir: Path, strict: bool = False) -> ModelReport:
    r = ModelReport(model_dir=model_dir)

    # 1) Model artifacts
    model_files = [p for p in model_dir.rglob("*") if is_probable_model_file(p)]
    if not model_files:
        r.add("WARN", "No obvious model file found (.pt/.pth/.ckpt/.pkl/.bin/.h5).")
    else:
        r.add("INFO", f"Found {len(model_files)} model artifact(s).")

    # 2) Metadata and GT hints
    meta_files = scan_for_metadata_files(model_dir)
    gt_candidates = [p for p in model_dir.rglob("*") if p.is_file() and "gt" in p.name.lower()]
    if not gt_candidates and "metadata.json" not in meta_files:
        r.add("WARN", "No obvious ground-truth file (e.g., gt.txt) and no metadata.json found.")

    # 3) Validate metadata.json / config.json paths if present
    for key in ("metadata.json", "config.json", "model_info.json"):
        p = meta_files.get(key)
        if not p:
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if not isinstance(data, dict):
                r.add("WARN", f"{p.name} is JSON but not an object.")
                continue

            # Look for path-like fields that often break with None
            suspicious_keys = ["gt_file", "ground_truth", "model", "model_path", "train_csv", "test_csv", "examples"]
            for k in suspicious_keys:
                if k in data and data[k] is None:
                    r.add("ERROR", f"{p.name}: key '{k}' is null (None).")
                if k in data and isinstance(data[k], str):
                    v = data[k].strip()
                    if v == "":
                        r.add("ERROR", f"{p.name}: key '{k}' is empty string.")
                    else:
                        # If it looks like a relative/absolute file path, verify it exists
                        if "/" in v or "\\" in v or v.endswith((".csv", ".txt", ".pt", ".pth", ".json", ".pkl", ".bin", ".h5")):
                            vp = Path(v)
                            resolved = vp if vp.is_absolute() else (model_dir / vp)
                            if not resolved.exists():
                                r.add("WARN", f"{p.name}: referenced path for '{k}' does not exist: {v}")
        except Exception as e:
            r.add("ERROR", f"Failed to parse {p.name}: {e}")

    # 4) Basic CSV checks
    csv_files = [p for p in model_dir.rglob("*.csv")]
    if not csv_files:
        r.add("WARN", "No CSV files found.")
    for c in csv_files:
        ok, msg = validate_csv_basic(c)
        if not ok:
            r.add("ERROR", f"{c.name}: {msg}")
        else:
            # detect suspicious tiny triggered CSVs that may only contain header
            if "trigger" in c.name.lower():
                try:
                    lines = sum(1 for _ in c.open("r", encoding="utf-8", errors="replace"))
                    if lines <= 1:
                        r.add("WARN", f"{c.name}: appears to contain only header/empty data.")
                except Exception:
                    pass

    # 5) Check for absolute-path join bug patterns in text/json
    # (e.g. values like '/data/clean/train.csv' causing os.path.join(base, abs) override)
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".json", ".txt", ".csv", ".yaml", ".yml", ".ini"}:
            continue
        text = read_text_file(p)
        if text and re.search(r'["\']\/data\/clean\/', text):
            r.add("WARN", f"{p.name}: contains absolute '/data/clean/...' path; may break os.path.join semantics.")

    # 6) Strict mode: require common detector-facing files
    if strict:
        must_have_any = [
            ("gt-like file", lambda d: any(f.is_file() for f in d.rglob("gt.txt"))),
            ("train experiment CSV", lambda d: any("experiment_train" in f.name.lower() for f in d.rglob("*.csv"))),
            ("clean test CSV", lambda d: any("test_clean" in f.name.lower() for f in d.rglob("*.csv"))),
        ]
        for label, fn in must_have_any:
            if not fn(model_dir):
                r.add("ERROR", f"Missing required ({label}) in strict mode.")

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to generated dataset root")
    parser.add_argument("--strict", action="store_true", help="Enforce stricter expected detector-facing structure")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[FATAL] data_root not found: {data_root}")
        raise SystemExit(2)

    model_dirs = find_model_dirs(data_root)
    if not model_dirs:
        print(f"[FATAL] No model dirs matching id-######## under: {data_root}")
        raise SystemExit(2)

    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] found {len(model_dirs)} model directories\n")

    all_reports: List[ModelReport] = []
    for md in model_dirs:
        rep = validate_model_dir(md, strict=args.strict)
        all_reports.append(rep)

    total_errors = sum(r.error_count for r in all_reports)
    total_warns = sum(r.warn_count for r in all_reports)

    for r in all_reports:
        print(f"=== {r.model_dir} ===")
        if not r.issues:
            print("  OK: no issues")
            continue
        for issue in r.issues:
            print(f"  [{issue.level}] {issue.message}")
        print()

    print("----- SUMMARY -----")
    print(f"Model dirs checked: {len(all_reports)}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warns}")

    # Exit code useful for CI/scripts
    raise SystemExit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    main()
