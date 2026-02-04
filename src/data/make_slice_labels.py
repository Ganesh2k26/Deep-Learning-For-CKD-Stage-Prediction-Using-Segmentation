import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

ROI_DIR = os.path.join(ROOT, "data", "classification", "kidney_rois")
CASE_CSV = os.path.join(ROOT, "data", "raw", "labels_ckd_cases.csv")  # backup file
OUT_CSV  = os.path.join(ROOT, "data", "raw", "labels_ckd.csv")        # new slice labels

def main():
    # 1) load case-level labels: filename like "case_00001.png", stage 1..4
    df_cases = pd.read_csv(CASE_CSV)   # columns: filename, stage
    case_to_stage = {}

    for _, row in df_cases.iterrows():
        # "case_00001.png" -> "case_00001"
        case_id = str(row["filename"]).split(".")[0]
        case_to_stage[case_id] = int(row["stage"])

    print(f"Loaded {len(case_to_stage)} case labels")

    # 2) walk through all ROI images and assign stage by case id
    rows = []
    for fname in os.listdir(ROI_DIR):
        if not fname.endswith(".png"):
            continue

        # "case_00001_z145.png" -> "case_00001"
        stem = fname.split("_z")[0]

        if stem in case_to_stage:
            stage = case_to_stage[stem]
            rows.append({"filename": fname, "stage": stage})
        else:
            print(f"[WARN] No stage found for ROI: {fname}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out_df)} slice labels to {OUT_CSV}")

if __name__ == "__main__":
    main()
