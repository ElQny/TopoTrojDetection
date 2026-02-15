#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

def read_trojan_flag_from_csv(csv_path: Path):
    # try common columns in experiment/test triggered CSVs
    candidates = ["poisoned", "is_poisoned", "triggered", "trojaned", "poison"]
    try:
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            cols = [c.strip().lower() for c in (reader.fieldnames or [])]
            for c in candidates:
                if c in cols:
                    # any True/1 entry => trojaned model
                    for r in rows:
                        v = str(r.get(c, "")).strip().lower()
                        if v in {"1", "true", "yes"}:
                            return 1
                    return 0
    except Exception:
        return None
    return None

def pick_model_file(model_dir: Path):
    cands = []
    for ext in ("*.pt", "*.pth", "*.ckpt", "*.bin", "*.pkl", "*.h5"):
        cands.extend(model_dir.rglob(ext))
    if not cands:
        return None
    # prefer names containing "model"
    cands = sorted(cands, key=lambda p: (0 if "model" in p.name.lower() else 1, len(str(p))))
    return cands[0]

def pick_experiment_train_csv(model_dir: Path):
    # prefer exact canonical name if already present
    p = model_dir / "experiment_train.csv"
    if p.exists():
        return p
    # fallback: mnist generator outputs include this pattern
    cands = sorted(model_dir.glob("*experiment_train*.csv"))
    if cands:
        return cands[0]
    # fallback: any train-like csv
    cands = sorted([x for x in model_dir.rglob("*.csv") if "train" in x.name.lower()])
    return cands[0] if cands else None

def infer_gt(model_dir: Path):
    # already present?
    gt = model_dir / "gt.txt"
    if gt.exists():
        t = gt.read_text(encoding="utf-8", errors="replace").strip()
        if t in {"0", "1"}:
            return int(t)

    # prefer triggered test csv signal
    trig = sorted(model_dir.glob("*experiment_test_triggered*.csv"))
    for t in trig:
        g = read_trojan_flag_from_csv(t)
        if g is not None:
            return g

    # fallback heuristic: if trigger-named directory exists and non-empty
    trigger_dirs = [d for d in model_dir.iterdir() if d.is_dir() and "trigger" in d.name.lower()]
    if trigger_dirs:
        return 1

    # safest default if unknown (change if your experiment encoding differs)
    return 0

def ensure_examples_dir(model_dir: Path):
    ex = model_dir / "examples"
    if ex.exists():
        return
    # optional; detector can run without examples when USE_EXAMPLE=False
    ex.mkdir(exist_ok=True)

def normalize_one(model_dir: Path, dry_run=False):
    msgs = []

    # 1) model artifact -> model.pt.1
    m = pick_model_file(model_dir)
    if m is None:
        msgs.append(f"[WARN] no model file found in {model_dir.name}")
    else:
        dst = model_dir / "model.pt.1"
        if m.resolve() != dst.resolve():
            msgs.append(f"[INFO] model: {m.name} -> {dst.name}")
            if not dry_run:
                shutil.copy2(m, dst)

    # 2) experiment_train.csv canonical copy
    tr = pick_experiment_train_csv(model_dir)
    if tr is None:
        msgs.append(f"[WARN] no train experiment csv found in {model_dir.name}")
    else:
        dst = model_dir / "experiment_train.csv"
        if tr.resolve() != dst.resolve():
            msgs.append(f"[INFO] train csv: {tr.name} -> {dst.name}")
            if not dry_run:
                shutil.copy2(tr, dst)

    # 3) gt.txt
    gt_val = infer_gt(model_dir)
    msgs.append(f"[INFO] gt inferred: {gt_val}")
    if not dry_run:
        (model_dir / "gt.txt").write_text(f"{gt_val}\n", encoding="utf-8")

    # 4) optional examples dir
    ensure_examples_dir(model_dir)

    return msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    model_dirs = sorted([p for p in root.glob("id-*") if p.is_dir()])

    if not model_dirs:
        print(f"[FATAL] no id-* dirs under {root}")
        raise SystemExit(2)

    for md in model_dirs:
        print(f"\n=== {md.name} ===")
        for msg in normalize_one(md, dry_run=args.dry_run):
            print(msg)

    print("\nDone.")
    raise SystemExit(0)

if __name__ == "__main__":
    main()
