import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

files = ["fixed_results.csv", "rl_results.csv"]

for fname in files:
    src = ROOT / fname
    if not src.exists():
        print(f"❌ Not found: {src}")
        continue
    try:
        df = pd.read_csv(src)
        out = src.with_suffix('.xlsx')
        df.to_excel(out, index=False)
        print(f"✅ Converted: {src.name} -> {out.name}")
    except Exception as e:
        print(f"❌ Failed converting {src.name}: {e}")
