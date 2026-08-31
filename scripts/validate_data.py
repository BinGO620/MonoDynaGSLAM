#!/usr/bin/env python3
"""MonoDynaGSLAM 数据源校验脚本。

校验 data/methods.json 与 data/categories.json、data/datasets.json 的一致性：
- JSON 语法合法
- 每个方法的 category 必须存在于 categories.json
- 每个方法必须有 arxiv 或 code 或 venue（可追溯性）
- 每个方法的 sensor 非空
- categories.json 的 examples 都能在 methods.json 找到
- datasets.json 的 used_by 都能在 methods.json 找到

用法: python scripts/validate_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

errors = []  # type: list[str]


def load(name: str) -> dict:
    path = DATA / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        errors.append(f"[FATAL] {name} 无法解析: {e}")
        return {}


def main() -> int:
    methods = load("methods.json").get("methods", [])
    categories = load("categories.json").get("categories", [])
    datasets = load("datasets.json").get("datasets", [])

    cat_ids = {c["id"] for c in categories}
    method_ids = {m["id"] for m in methods}
    method_ids_lower = {m["id"].lower() for m in methods}

    # 1. 方法必须属于已定义分类
    for m in methods:
        if m.get("category") not in cat_ids:
            errors.append(
                f"[ERR] {m.get('id')}: category '{m.get('category')}' 不在 categories.json "
                f"(可用: {sorted(cat_ids)})"
            )

    # 2. 可追溯性：arxiv 或 code 或 venue 至少一个
    for m in methods:
        if not (m.get("arxiv") or m.get("code") or m.get("venue")):
            errors.append(f"[ERR] {m.get('id')}: 缺 arXiv/code/venue，无法追溯")

    # 3. sensor 非空
    for m in methods:
        if not m.get("sensor"):
            errors.append(f"[ERR] {m.get('id')}: sensor 为空")

    # 4. categories.examples 都能找到对应方法
    for c in categories:
        for ex in c.get("examples", []):
            if ex.lower() not in method_ids_lower:
                errors.append(f"[WARN] categories/{c['id']} 的 example '{ex}' 不在 methods.json")

    # 5. datasets.used_by 都能找到对应方法
    for d in datasets:
        for m in d.get("used_by", []):
            if m.lower() not in method_ids_lower:
                errors.append(f"[WARN] datasets/{d['id']} 的 used_by '{m}' 不在 methods.json")

    # 6. papers/ 笔记与 methods 一致性（非强制，仅提示）
    notes = sorted(p for p in (ROOT / "papers").glob("*.md"))
    note_ids = [n.stem.split("-")[0] for n in notes if n.stem.split("-")[0].isdigit()]
    arxiv_ids = {str(m.get("arxiv")) for m in methods if m.get("arxiv")}
    for nid in note_ids:
        if nid not in arxiv_ids:
            errors.append(f"[INFO] papers/{nid} 对应 arXiv ID 不在 methods.json（检查是否漏录方法）")

    if errors:
        print(f"发现 {len(errors)} 个问题：")
        for e in errors:
            print("  " + e)
        return 1 if any(e.startswith("[ERR]") or e.startswith("[FATAL]") for e in errors) else 0

    print(f"OK: {len(methods)} 方法 / {len(categories)} 分类 / {len(datasets)} 数据集，全部一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
