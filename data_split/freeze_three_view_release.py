#!/usr/bin/env python3
"""冻结已通过独立验证的三视图 manifest 发布版本。"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# 让 data_split 子目录脚本能导入仓库根目录的 config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import SPLIT_OUTPUT

ROOT = SPLIT_OUTPUT
RELEASE_ID = "three_view_v1_seed42"


def parse_args():
    parser = argparse.ArgumentParser(
        description="冻结已验证的E0/E1/E2 manifest发布版本"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--release-id", default=RELEASE_ID)
    return parser.parse_args()


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    args = parse_args()
    validation_path = args.root / "validation_report.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(
        validation_path.read_text(encoding="utf-8")
    )
    if validation.get("status") != "all_checks_passed":
        raise ValueError("独立验证未全部通过，拒绝冻结")

    release_path = args.root / "frozen_release.json"
    if release_path.exists():
        raise FileExistsError(f"拒绝覆盖：{release_path}")

    tracked = [
        args.root / "patient_split.json",
        args.root / "patient_split.csv",
        args.root / "view_summary.json",
        validation_path,
    ]
    tracked.extend(sorted((args.root / "catalogs").glob("*.csv")))
    tracked.extend(sorted((args.root / "experiments").glob("*/*.csv")))
    if any(not path.is_file() for path in tracked):
        missing = [str(path) for path in tracked if not path.is_file()]
        raise FileNotFoundError(f"冻结文件缺失：{missing}")

    release = {
        "release_id": args.release_id,
        "frozen_at": datetime.now().astimezone().isoformat(),
        "status": "frozen",
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "root": str(args.root),
        "validation_status": validation["status"],
        "validation_report_sha256": file_sha256(validation_path),
        "files": {
            str(path.relative_to(args.root)): file_sha256(path)
            for path in tracked
        },
        "use_policy": (
            "训练和评价只能使用experiments/下对应视图清单；"
            "不得把catalog中held_out_pip_not_in_common_eval行加入训练或公共评价"
        ),
    }
    with release_path.open("w", encoding="utf-8") as handle:
        json.dump(release, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(release, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
