#!/usr/bin/env python3
"""
find_none_gt.py

Pinpoints which model folder(s) would produce gt_file=None using a robust
approximation of run_troj_detector.py parsing logic.

It inspects common metadata files and reports:
- where GT was found (and resolved path),
- where GT is missing/empty/null,
- where GT points to non-existing files.

Usage:
    python find_none_gt.py --data_root ./data/data2
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ID_DIR_RE = re.compile(r"^id-\d{8}$")


# Common key names people use for GT or label file pointers
GT_KEYS = [
    "gt_file",
    "gt",
    "ground_truth_file",
    "ground_truth",
    "label_file",
    "truth_file",
    "trojan_label_file",
]

# Fallback filenames to look for directly
GT_FILENAMES = [
    "gt.txt",
    "ground_truth.txt",
    "label.txt",
    "truth.txt",
]


def find_model_dirs(data_root: Path) -> List[Path]:
    direct = [p for p in data_root.iterdir() if p.is_dir() and ID_DIR_RE.match(p.name)]
    if direct:
        return sorted(direct)
    return sorted([p for p in data_root.rglob("*") if p.is_dir() and ID_DIR_RE.match(p.name)])


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def read_csv_header_and_first_row(path: Path) -> Tuple[List[str], Optional[List[str]]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            r = csv.reader(f)
            header = next(r, [])
            first_row = next(r, None)
            return header, first_row
    except Exception:
        return [], None


def parse_gt_from_json_dict(d: Dict[str, Any]) -> Tuple[Optional[Any], Optional[str]]:
    # direct keys
    for k in GT_KEYS:
        if k in d:
            return d[k], f"json key '{k}'"

    # nested scan (one level)
    for k, v in d.items():
        if isinstance(v, dict):
            for kk in GT_KEYS:
                if kk in v:
                    return v[kk], f"json key '{k}.{kk}'"

    return None, None


def parse_gt_from_csv(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Heuristic:
    - if header has gt-like column, read first row value.
    """
    header, first_row = read_csv_header_and_first_row(path)
    if not header:
        return None, None
    lowered = [h.strip().lower() for h in header]
    gt_cols = {"gt", "gt_file", "ground_truth", "ground_truth_file", "label", "label_file", "truth_file"}

    for i, h in enumerate(lowered):
        if h in gt_cols:
            val = None
            if first_row is not None and i < len(first_row):
                val = first_row[i].strip()
            return (val if val != "" else None), f"csv column '{header[i]}'"
    return None, None


def resolve_if_path(model_dir: Path, raw_val: str) -> Path:
    p = Path(raw_val)
    return p if p.is_absolute() else (model_dir / p)


def check_one_model(model_dir: Path) -> Dict[str, Any]:
    result = {
        "model_dir": str(model_dir),
        "status": "UNKNOWN",
        "source": None,
        "raw_gt": None,
        "resolved_gt_path": None,
        "reason": None,
    }

    # 1) direct GT file existence
    for name in GT_FILENAMES:
        p = model_dir / name
        if p.exists():
            result.update({
                "status": "OK_DIRECT_FILE",
                "source": f"direct file '{name}'",
                "raw_gt": name,
                "resolved_gt_path": str(p),
                "reason": "Found GT file directly in model directory."
            })
            return result

    # 2) metadata scan (json/csv/txt)
    candidate_files = []
    for p in model_dir.rglob("*"):
        if not p.is_file():
            continue
        lname = p.name.lower()
        if lname.endswith(".json") or lname.endswith(".csv") or "meta" in lname or "config" in lname or "experiment" in lname:
            candidate_files.append(p)

    # deterministic order
    candidate_files = sorted(candidate_files)

    for f in candidate_files:
        if f.suffix.lower() == ".json":
            data = read_json(f)
            if isinstance(data, dict):
                gt_val, src = parse_gt_from_json_dict(data)
                if src is not None:
                    return classify_gt_value(model_dir, f, gt_val, src)

        elif f.suffix.lower() == ".csv":
            gt_val, src = parse_gt_from_csv(f)
            if src is not None:
                return classify_gt_value(model_dir, f, gt_val, src)

    # 3) text-based fallback for obvious gt*.txt anywhere under model dir
    for p in sorted(model_dir.rglob("*gt*.txt")):
        txt = p.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            # could be label value or path
            if txt.lower() in {"0", "1", "clean", "trojaned", "trojan", "true", "false"}:
                result.update({
                    "status": "OK_GT_VALUE_ONLY",
                    "source": str(p.relative_to(model_dir)),
                    "raw_gt": txt,
                    "resolved_gt_path": None,
                    "reason": "GT value found (not a file path)."
                })
                return result
            rp = resolve_if_path(model_dir, txt)
            if rp.exists():
                result.update({
                    "status": "OK_FROM_TEXT_PTR",
                    "source": str(p.relative_to(model_dir)),
                    "raw_gt": txt,
                    "resolved_gt_path": str(rp),
                    "reason": "GT pointer from text file resolves to existing path."
                })
                return result
            else:
                result.update({
                    "status": "BROKEN_TEXT_PTR",
                    "source": str(p.relative_to(model_dir)),
                    "raw_gt": txt,
                    "resolved_gt_path": str(rp),
                    "reason": "GT pointer found but target does not exist."
                })
                return result

    result.update({
        "status": "NONE_GT",
        "reason": "No GT key/file detected; this folder is a likely source of gt_file=None."
    })
    return result


def classify_gt_value(model_dir: Path, source_file: Path, gt_val: Any, src: str) -> Dict[str, Any]:
    res = {
        "model_dir": str(model_dir),
        "status": None,
        "source": f"{source_file.relative_to(model_dir)} ({src})",
        "raw_gt": gt_val,
        "resolved_gt_path": None,
        "reason": None,
    }

    if gt_val is None:
        res["status"] = "NONE_GT"
        res["reason"] = "GT field exists but value is null/None."
        return res

    if isinstance(gt_val, str):
        v = gt_val.strip()
        if v == "":
            res["status"] = "EMPTY_GT"
            res["reason"] = "GT field exists but value is empty string."
            return res

        # Value labels (not path)
        if v.lower() in {"0", "1", "clean", "trojaned", "trojan", "true", "false"}:
            res["status"] = "OK_GT_VALUE_ONLY"
            res["reason"] = "GT is a label value, not a file path."
            return res

        # Treat as path
        rp = resolve_if_path(model_dir, v)
        res["resolved_gt_path"] = str(rp)
        if rp.exists():
            res["status"] = "OK_PTR_EXISTS"
            res["reason"] = "GT path pointer resolves to existing file."
        else:
            res["status"] = "BROKEN_PTR"
            res["reason"] = "GT path pointer does not resolve to existing file."
        return res

    # Unexpected data type
    res["status"] = "BAD_GT_TYPE"
    res["reason"] = f"GT field has unsupported type: {type(gt_val).__name__}"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Dataset root containing id-######## folders")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[FATAL] data_root not found: {data_root}")
        raise SystemExit(2)

    model_dirs = find_model_dirs(data_root)
    if not model_dirs:
        print(f"[FATAL] No id-######## model directories found under: {data_root}")
        raise SystemExit(2)

    print(f"[INFO] data_root: {data_root}")
    print(f"[INFO] model_dirs: {len(model_dirs)}\n")

    problematic = 0
    for md in model_dirs:
        r = check_one_model(md)
        st = r["status"]
        print(f"=== {md.name} ===")
        print(f"status : {st}")
        if r["source"] is not None:
            print(f"source : {r['source']}")
        if r["raw_gt"] is not None:
            print(f"raw_gt : {r['raw_gt']}")
        if r["resolved_gt_path"] is not None:
            print(f"path   : {r['resolved_gt_path']}")
        print(f"reason : {r['reason']}\n")

        if st in {"NONE_GT", "EMPTY_GT", "BROKEN_PTR", "BROKEN_TEXT_PTR", "BAD_GT_TYPE"}:
            problematic += 1

    print("----- SUMMARY -----")
    print(f"Total model dirs: {len(model_dirs)}")
    print(f"Likely gt_file=None / broken GT dirs: {problematic}")

    # nonzero exit if any likely problematic folders
    raise SystemExit(1 if problematic > 0 else 0)


if __name__ == "__main__":
    main()
