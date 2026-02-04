import os
import cv2
import numpy as np
import pandas as pd

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MASK_DIR = os.path.join(ROOT, "data", "annotations", "ct_masks")
OUT_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd.csv")

os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def get_case_id_from_filename(filename: str) -> str:
    """
    Extracts case_id from a mask filename.
    Assumes names like:
        case_00001_000.png
        case_00001_slice_000.png
    We will take the first two parts: case_00001
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        return parts[0] + "_" + parts[1]   # e.g. case_00001
    else:
        # fallback: use full name without extension
        return os.path.splitext(filename)[0]

def main():
    if not os.path.isdir(MASK_DIR):
        raise RuntimeError(f"Mask directory not found: {MASK_DIR}")

    case_coverages = {}  # case_id -> list of coverage values

    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask PNG files found in {MASK_DIR}")

    print(f"Found {len(mask_files)} mask slices in {MASK_DIR}")

    for fname in mask_files:
        case_id = get_case_id_from_filename(fname)
        path = os.path.join(MASK_DIR, fname)

        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] Could not read mask: {path}")
            continue

        # coverage: fraction of pixels > 0
        total_pixels = mask.size
        positive_pixels = (mask > 0).sum()
        coverage = positive_pixels / float(total_pixels)

        if case_id not in case_coverages:
            case_coverages[case_id] = []
        case_coverages[case_id].append(coverage)

    # compute mean coverage for each case
    data = []
    for case_id, cover_list in case_coverages.items():
        if len(cover_list) == 0:
            continue
        mean_cov = float(np.mean(cover_list))
        data.append((case_id, mean_cov))

    if len(data) == 0:
        raise RuntimeError("No valid coverage data computed from masks.")

    df = pd.DataFrame(data, columns=["case_id", "mean_coverage"])
    df = df.sort_values("case_id").reset_index(drop=True)

    print("\nSample coverage values:")
    print(df.head())

    # Compute quartiles for mean_coverage
    q1 = df["mean_coverage"].quantile(0.25)
    q2 = df["mean_coverage"].quantile(0.50)
    q3 = df["mean_coverage"].quantile(0.75)

    print("\nCoverage quartiles:")
    print(f"Q1: {q1:.6f}, Q2: {q2:.6f}, Q3: {q3:.6f}")

    # Map coverage -> synthetic CKD stage (1..4)
    def coverage_to_stage(cov: float) -> int:
        if cov <= q1:
            return 1
        elif cov <= q2:
            return 2
        elif cov <= q3:
            return 3
        else:
            return 4

    df["stage"] = df["mean_coverage"].apply(coverage_to_stage)

    # Keep only case_id and stage in final CSV
    labels_df = df[["case_id", "stage"]].copy()

    print("\nSample synthetic CKD labels:")
    print(labels_df.head())

    labels_df.to_csv(OUT_CSV, index=False)
    print(f"\nSynthetic labels saved to: {OUT_CSV}")
    print("Each case has a stage in {1,2,3,4} based on mask coverage severity.")

if __name__ == "__main__":
    main()
