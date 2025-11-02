from ais_to_parguet import fn
from pathlib import Path
import sys

if __name__ == "__main__":
    csv = Path(r"C:\Users\veron\Desktop\DTU\2nd Semester\Deep Learning\Final Project\aisdk-2025-02-27.csv")
    out = Path(r"C:\Users\veron\Desktop\DTU\2nd Semester\Deep Learning\Final Project\aisdk-2025-02-27.parquet")

    if not csv.exists():
        print(f"ERROR: input file not found: {csv}", file=sys.stderr)
        raise SystemExit(1)

    fn(str(csv), str(out))
