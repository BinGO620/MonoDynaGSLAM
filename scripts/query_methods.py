#!/usr/bin/env python3
"""MonoDynaGSLAM 方法目录查询脚本。

按分类/传感器/关键词筛选方法：

    python scripts/query_methods.py --category anti-dynamic --sensor monocular
    python scripts/query_methods.py --keyword wild
    python scripts/query_methods.py --category face-dynamic --json
"""
import argparse
import json
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "methods.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="查询 dynamic 3DGS SLAM 方法目录")
    ap.add_argument("--category", help="按分类筛选 (anti-dynamic / face-dynamic / static-base)")
    ap.add_argument("--sensor", help="按传感器筛选 (monocular / rgb-d / stereo)")
    ap.add_argument("--keyword", help="按 id/name/全名关键词筛选")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args()

    methods = json.loads(DATA.read_text(encoding="utf-8"))["methods"]

    def match(m: dict) -> bool:
        if args.category and m.get("category") != args.category:
            return False
        if args.sensor and args.sensor not in [s.lower() for s in m.get("sensor", [])]:
            return False
        if args.keyword:
            kw = args.keyword.lower()
            blob = " ".join(
                str(m.get(k, "")) for k in ("id", "name", "full_name", "venue", "verdict")
            ).lower()
            if kw not in blob:
                return False
        return True

    hits = [m for m in methods if match(m)]

    if args.json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
    else:
        if not hits:
            print("无匹配。")
            return 1
        for m in hits:
            print(f"- {m['id']} [{m['category']}] {m.get('full_name', m['name'])}")
            print(f"    venue={m.get('venue')}  sensor={','.join(m.get('sensor', []))}  "
                  f"arxiv={m.get('arxiv') or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
