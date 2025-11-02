from ais_to_parguet import fn
from pathlib import Path
import sys
from typing import Optional


DEFAULT_CSV = r"C:\Users\veron\Desktop\DTU\2nd Semester\Deep Learning\Final Project\aisdk-2025-02-27.csv"
DEFAULT_PARQUET = r"C:\Users\veron\Desktop\DTU\2nd Semester\Deep Learning\Final Project\aisdk-2025-02-27.parquet"


def main(csv_path: Optional[str] = None, out_path: Optional[str] = None) -> None:
    """Process CSV -> parquet using ais_to_parguet.fn.

    If csv_path/out_path are None, defaults defined above are used.
    """
    csv = Path(csv_path or DEFAULT_CSV)
    out = Path(out_path or DEFAULT_PARQUET)

    if not csv.exists():
        print(f"ERROR: input file not found: {csv}", file=sys.stderr)
        raise SystemExit(1)

    fn(str(csv), str(out))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        main(args[0], args[1])
    elif len(args) == 1:
        main(args[0], None)
    else:
        main()
