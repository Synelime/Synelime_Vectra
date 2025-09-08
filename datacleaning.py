import argparse
from pathlib import Path
import pandas as pd, pdfplumber, re, chardet, dateparser
from tqdm import tqdm


def detect_encoding(file: Path, sample_size: int = 200_000) -> str:
    with open(file, "rb") as f:
        raw = f.read(sample_size)
    enc = chardet.detect(raw).get("encoding") or "utf-8"
    return enc


def read_csvs(folder: Path):
    frames = []
    csv_files = list(folder.rglob("*.csv")) if folder.exists() else []
    for f in tqdm(csv_files, desc="CSVs"):
        try:
            enc = detect_encoding(f)
            df = pd.read_csv(f, encoding=enc, on_bad_lines="skip", engine="python")
            df["_src"] = str(f)
            frames.append(df)
        except Exception:
            pass
    return frames


def read_pdfs(folder: Path):
    frames = []
    pdf_files = list(folder.rglob("*.pdf")) if folder.exists() else []
    for f in tqdm(pdf_files, desc="PDFs"):
        try:
            with pdfplumber.open(f) as pdf:
                for p in pdf.pages:
                    for t in (p.extract_tables() or []):
                        df = pd.DataFrame(t)
                        if not df.empty and df.iloc[0].notna().all():
                            df.columns, df = df.iloc[0], df[1:]
                        df["_src"] = str(f)
                        frames.append(df)
        except Exception:
            pass
    return frames


def clean(df):
    if df.empty:
        return df
    df.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_").lower() for c in df.columns]
    for c in df.select_dtypes("object"):
        s = df[c].astype(str).str.strip()
        # Try parsing dates where possible
        try:
            parsed = s.map(lambda x: dateparser.parse(x) if isinstance(x, str) else x)
            if pd.notna(parsed).any():
                try:
                    df[c] = pd.to_datetime(parsed)
                except Exception:
                    df[c] = s
            else:
                df[c] = s
        except Exception:
            df[c] = s
    return df


def pipeline(csv_dir, pdf_dir, out_dir, outfile):
    frames = []
    frames += [clean(f) for f in read_csvs(csv_dir)]
    frames += [clean(f) for f in read_pdfs(pdf_dir)]
    if not frames:
        print("No input tables found in CSV or PDF folders.")
        return None
    # Align columns across all frames
    all_cols = sorted({c for d in frames for c in d.columns})
    frames = [f.reindex(columns=all_cols) for f in frames]
    df = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / outfile
    df.to_csv(out, index=False)
    print(f"Saved {out} ({len(df)} rows)")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", type=Path, default=Path("data/input/csv"))
    p.add_argument("--pdf_dir", type=Path, default=Path("data/input/pdf"))
    p.add_argument("--out_dir", type=Path, default=Path("data/output"))
    p.add_argument("--outfile", type=str, default="unified.csv")
    a = p.parse_args()
    pipeline(a.csv_dir, a.pdf_dir, a.out_dir, a.outfile)


if __name__ == "__main__":
    main()
