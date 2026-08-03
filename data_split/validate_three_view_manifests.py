#!/usr/bin/env python3
"""独立验证患者级划分和 E0/E1/E2 三视图 manifest。

检查患者零交集、公共 val/test 完全一致、所有路径与 SHA-256、训练清单
精确去重、跨患者完全重复和 E1/E2 画中画补充集合一致性。
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# 让 data_split 子目录脚本能导入仓库根目录的 config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import SPLIT_OUTPUT

ROOT = SPLIT_OUTPUT
VIEW_IDS = [
    "E0_clean_base",
    "E1_clean_plus_right",
    "E2_clean_plus_notch",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="验证统一患者划分与三视图manifest"
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation_report.json"
    )
    return parser.parse_args()


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_unique(rows, field, label):
    values = [row[field] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} 的 {field} 不唯一")


def check_split(root):
    split = json.loads(
        (root / "patient_split.json").read_text(encoding="utf-8")
    )
    sets = {
        name: set(split[name]) for name in ["train", "val", "test"]
    }
    if sets["train"] & sets["val"]:
        raise ValueError("train/val 患者泄漏")
    if sets["train"] & sets["test"]:
        raise ValueError("train/test 患者泄漏")
    if sets["val"] & sets["test"]:
        raise ValueError("val/test 患者泄漏")
    counts = {name: len(values) for name, values in sets.items()}
    if any(value < 1 for value in counts.values()):
        raise ValueError(f"患者划分存在空集合：{counts}")
    return sets, counts


def check_catalog(view_id, rows, split_sets):
    if not rows:
        raise ValueError(f"{view_id} catalog 为空")
    keys = [
        (row["source_kind"], row["relative_path"]) for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{view_id} catalog身份键重复")

    hash_patients = defaultdict(set)
    for row in rows:
        path = Path(row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(f"manifest路径不存在：{path}")
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"SHA-256不一致：{path}")
        if row["patient_id"] not in split_sets[row["split"]]:
            raise ValueError(
                f"患者划分字段不一致：{row['relative_path']}"
            )
        hash_patients[row["sha256"]].add(row["patient_id"])
    cross_patient = {
        digest: patients
        for digest, patients in hash_patients.items()
        if len(patients) > 1
    }
    if cross_patient:
        raise ValueError(
            f"{view_id} 存在跨患者完全重复："
            f"{list(cross_patient.items())[:3]}"
        )


def check_experiment(root, view_id, split_sets):
    subsets = {}
    for split_name in ["train", "val", "test"]:
        path = root / "experiments" / view_id / f"{split_name}.csv"
        rows = load_csv(path)
        subsets[split_name] = rows
        for row in rows:
            if row["patient_id"] not in split_sets[split_name]:
                raise ValueError(
                    f"{view_id}/{split_name} 包含错误患者"
                )
            if not Path(row["image_path"]).is_file():
                raise FileNotFoundError(row["image_path"])

    train = subsets["train"]
    if not train or not subsets["val"] or not subsets["test"]:
        raise ValueError(f"{view_id} 的 train/val/test 存在空清单")
    assert_unique(train, "sha256", f"{view_id}/train")
    if any(row["train_keep_after_exact_dedup"] != "yes" for row in train):
        raise ValueError(f"{view_id} 训练清单含未通过精确去重的行")
    if any(row["source_kind"] != "clean" for row in subsets["val"]):
        raise ValueError(f"{view_id} val 含非干净图")
    if any(row["source_kind"] != "clean" for row in subsets["test"]):
        raise ValueError(f"{view_id} test 含非干净图")
    return subsets


def identity(rows):
    return [
        (
            row["relative_path"],
            row["patient_id"],
            row["image_path"],
            row["sha256"],
        )
        for row in rows
    ]


def main():
    args = parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖：{args.output}")

    split_sets, split_counts = check_split(args.root)
    catalogs = {}
    experiments = {}
    for view_id in VIEW_IDS:
        catalog = load_csv(
            args.root / "catalogs" / f"{view_id}.csv"
        )
        check_catalog(view_id, catalog, split_sets)
        catalogs[view_id] = catalog
        experiments[view_id] = check_experiment(
            args.root, view_id, split_sets
        )

    base_val = identity(experiments["E0_clean_base"]["val"])
    base_test = identity(experiments["E0_clean_base"]["test"])
    for view_id in VIEW_IDS[1:]:
        if identity(experiments[view_id]["val"]) != base_val:
            raise ValueError(f"{view_id} val 与E0不完全一致")
        if identity(experiments[view_id]["test"]) != base_test:
            raise ValueError(f"{view_id} test 与E0不完全一致")

    supplemental_sets = {}
    for view_id in VIEW_IDS[1:]:
        supplemental = {
            row["relative_path"]
            for row in catalogs[view_id]
            if row["source_kind"] == "processed_pip"
        }
        supplemental_sets[view_id] = supplemental
    if (
        supplemental_sets["E1_clean_plus_right"]
        != supplemental_sets["E2_clean_plus_notch"]
    ):
        raise ValueError("E1/E2画中画身份集合不一致")

    if any(
        row["source_kind"] != "clean"
        for row in catalogs["E0_clean_base"]
    ):
        raise ValueError("E0 catalog 含非干净图")
    clean_identities = {
        view_id: {
            (row["relative_path"], row["patient_id"], row["sha256"])
            for row in catalogs[view_id]
            if row["source_kind"] == "clean"
        }
        for view_id in VIEW_IDS
    }
    if len({frozenset(values) for values in clean_identities.values()}) != 1:
        raise ValueError("E0/E1/E2 干净图身份集合不一致")

    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "all_checks_passed",
        "script": str(Path(__file__)),
        "script_sha256": file_sha256(Path(__file__)),
        "root": str(args.root),
        "patient_split_counts": split_counts,
        "checks": {
            "patient_intersections_empty": True,
            "all_manifest_paths_exist": True,
            "all_sha256_match": True,
            "no_cross_patient_exact_duplicates": True,
            "train_sha256_unique_after_dedup": True,
            "common_val_identical_across_views": True,
            "common_test_identical_across_views": True,
            "val_test_clean_only": True,
            "right_notch_supplement_identity_sets_equal": True,
        },
        "catalog_counts": {
            view_id: len(catalogs[view_id]) for view_id in VIEW_IDS
        },
        "experiment_counts": {
            view_id: {
                split_name: len(experiments[view_id][split_name])
                for split_name in ["train", "val", "test"]
            }
            for view_id in VIEW_IDS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
