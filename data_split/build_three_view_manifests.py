#!/usr/bin/env python3
"""生成统一患者级划分和 E0/E1/E2 三种数据视图 manifest。

默认口径：

1. 对至少有一张干净图的患者按 seed=42 做 80/10/10 患者级划分；
2. 当前只有画中画图的患者全部加入 train；
3. E0 = clean_base；
4. E1 = clean_base + pip_right；
5. E2 = clean_base + pip_notch；
6. val/test 患者的画中画派生图只保留在完整 catalog，不进入公共评价；
7. 字节级完全重复只从训练清单跳过；跨患者完全重复直接报错。

脚本只写 manifest 和汇总，不复制、移动或修改任何图像。
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# 让 data_split 子目录脚本能导入仓库根目录的 config.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import (
    SPLIT_OUTPUT,
    STAGE2_MAPPING,
    STAGE3_PIP_LABELS,
    STAGE4_OUTPUT,
)


STAGE4_MAPPING = STAGE4_OUTPUT / "mapping.csv"
BOX_QC_STATUS = (
    STAGE4_OUTPUT / "box_qc_human_review" / "final_box_qc_status.csv"
)
OUTPUT_ROOT = SPLIT_OUTPUT
# 兼容调用方：PIP_LABELS 别名
PIP_LABELS = STAGE3_PIP_LABELS
DEFAULT_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一患者级划分与E0/E1/E2 manifest"
    )
    parser.add_argument(
        "--stage2-mapping", type=Path, default=STAGE2_MAPPING
    )
    parser.add_argument("--pip-labels", type=Path, default=PIP_LABELS)
    parser.add_argument(
        "--stage4-mapping", type=Path, default=STAGE4_MAPPING
    )
    parser.add_argument(
        "--box-qc-status", type=Path, default=BOX_QC_STATUS
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    return parser.parse_args()


def validate_args(args):
    for path in [
        args.stage2_mapping,
        args.pip_labels,
        args.stage4_mapping,
        args.box_qc_status,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(value <= 0 for value in ratios):
        raise ValueError("train/val/test 比例必须全部大于 0")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/val/test 比例之和必须为 1")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"拒绝写入空 CSV：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patient_id(relative_path):
    parts = Path(relative_path).parts
    if len(parts) < 2:
        raise ValueError(f"relative_path 缺少患者目录：{relative_path}")
    return parts[0]


def validate_unique(rows, label):
    paths = [row["relative_path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} 存在重复 relative_path")


def load_and_validate_inputs(args):
    stage2 = load_csv(args.stage2_mapping)
    pip_labels = load_csv(args.pip_labels)
    stage4 = load_csv(args.stage4_mapping)
    box_qc = load_csv(args.box_qc_status)
    for rows, label in [
        (stage2, "stage2 mapping"),
        (pip_labels, "pip labels"),
        (stage4, "stage4 mapping"),
        (box_qc, "box qc"),
    ]:
        validate_unique(rows, label)

    if any(row["manual_label"] != "pip" for row in pip_labels):
        raise ValueError("画中画处理池包含非 pip 标签")
    if any(row["final_box_qc_status"] != "pass" for row in box_qc):
        raise ValueError("存在未通过最终框质控的画中画图像")

    stage2_paths = {row["relative_path"] for row in stage2}
    pip_paths = {row["relative_path"] for row in pip_labels}
    stage4_paths = {row["relative_path"] for row in stage4}
    qc_paths = {row["relative_path"] for row in box_qc}
    if pip_paths != stage4_paths or pip_paths != qc_paths:
        raise ValueError("画中画标签、双分支映射和最终框质控集合不一致")
    if not pip_paths <= stage2_paths:
        raise ValueError("画中画集合包含第二阶段映射之外的图像")

    for row in stage2:
        path = Path(row["masked"])
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise FileNotFoundError(f"第二阶段图像不存在：{path}")
    for row in stage4:
        for field in ["right_output", "notch_output"]:
            path = Path(row[field])
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                raise FileNotFoundError(f"双分支图像不存在：{path}")
    return stage2, pip_labels, stage4


def build_patient_split(
    stage2,
    pip_paths,
    seed,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
):
    paths_by_patient = defaultdict(list)
    for row in stage2:
        paths_by_patient[patient_id(row["relative_path"])].append(
            row["relative_path"]
        )

    clean_paths = {
        row["relative_path"] for row in stage2
        if row["relative_path"] not in pip_paths
    }
    all_patients = set(paths_by_patient)
    clean_patients = sorted({
        patient_id(path) for path in clean_paths
    })
    pip_patients = {
        patient_id(path) for path in pip_paths
    }
    pip_only_patients = sorted(pip_patients - set(clean_patients))
    mixed_patients = pip_patients & set(clean_patients)

    if len(clean_patients) < 3:
        raise ValueError("至少需要 3 名有干净图患者才能划分 train/val/test")

    shuffled = list(clean_patients)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val = int(len(shuffled) * val_ratio)
    n_test = len(shuffled) - n_train - n_val
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "当前有干净图患者数量和划分比例无法保证 train/val/test 均非空"
        )
    split = {
        "train": sorted(shuffled[:n_train] + pip_only_patients),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }
    split_sets = {name: set(values) for name, values in split.items()}
    if split_sets["train"] & split_sets["val"]:
        raise ValueError("train 与 val 患者有交集")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("train 与 test 患者有交集")
    if split_sets["val"] & split_sets["test"]:
        raise ValueError("val 与 test 患者有交集")
    if set().union(*split_sets.values()) != all_patients:
        raise ValueError("患者划分未覆盖全部患者")

    split_by_patient = {
        patient: name
        for name, patients in split.items()
        for patient in patients
    }
    clean_count = Counter(
        patient_id(path) for path in clean_paths
    )
    pip_count = Counter(
        patient_id(path) for path in pip_paths
    )
    patient_rows = []
    for patient in sorted(all_patients):
        if patient in pip_only_patients:
            cohort = "pip_only_current"
        elif patient in mixed_patients:
            cohort = "mixed_current_images"
        else:
            cohort = "clean_only"
        patient_rows.append({
            "patient_id": patient,
            "split": split_by_patient[patient],
            "cohort": cohort,
            "clean_image_count": clean_count[patient],
            "pip_image_count": pip_count[patient],
            "total_image_count": (
                clean_count[patient] + pip_count[patient]
            ),
        })
    return split, split_by_patient, patient_rows, clean_paths


def base_manifest_row(
    view_id,
    relative_path,
    image_path,
    split_name,
    cohort,
    source_kind,
    pip_processing,
):
    is_clean = source_kind == "clean"
    if split_name == "train":
        experiment_role = "train"
        include = "yes"
        exclusion_reason = ""
    elif is_clean:
        experiment_role = split_name
        include = "yes"
        exclusion_reason = ""
    else:
        experiment_role = "held_out_pip_not_in_common_eval"
        include = "no"
        exclusion_reason = (
            "val/test患者的画中画派生图不进入公共干净评价"
        )
    return {
        "view_id": view_id,
        "relative_path": relative_path,
        "patient_id": patient_id(relative_path),
        "split": split_name,
        "patient_cohort": cohort,
        "source_kind": source_kind,
        "pip_processing": pip_processing,
        "image_path": str(image_path),
        "experiment_role": experiment_role,
        "include_in_common_experiment": include,
        "exclusion_reason": exclusion_reason,
        "sha256": "",
        "exact_duplicate_of": "",
        "train_keep_after_exact_dedup": "",
    }


def build_catalogs(
    stage2,
    stage4,
    clean_paths,
    split_by_patient,
    cohort_by_patient,
):
    clean_rows = []
    for row in stage2:
        relative_path = row["relative_path"]
        if relative_path not in clean_paths:
            continue
        patient = patient_id(relative_path)
        clean_rows.append(base_manifest_row(
            "E0_clean_base",
            relative_path,
            Path(row["masked"]),
            split_by_patient[patient],
            cohort_by_patient[patient],
            "clean",
            "none",
        ))

    catalogs = {
        "E0_clean_base": [dict(row) for row in clean_rows],
        "E1_clean_plus_right": [],
        "E2_clean_plus_notch": [],
    }
    for view_id in ["E1_clean_plus_right", "E2_clean_plus_notch"]:
        catalogs[view_id] = [
            {**row, "view_id": view_id} for row in clean_rows
        ]

    for row in stage4:
        relative_path = row["relative_path"]
        patient = patient_id(relative_path)
        split_name = split_by_patient[patient]
        cohort = cohort_by_patient[patient]
        catalogs["E1_clean_plus_right"].append(base_manifest_row(
            "E1_clean_plus_right",
            relative_path,
            Path(row["right_output"]),
            split_name,
            cohort,
            "processed_pip",
            "right",
        ))
        catalogs["E2_clean_plus_notch"].append(base_manifest_row(
            "E2_clean_plus_notch",
            relative_path,
            Path(row["notch_output"]),
            split_name,
            cohort,
            "processed_pip",
            "notch",
        ))

    for rows in catalogs.values():
        rows.sort(key=lambda row: (
            row["patient_id"],
            row["relative_path"],
            0 if row["source_kind"] == "clean" else 1,
        ))
    return catalogs


def add_hashes_and_exact_dedup(catalogs):
    hash_cache = {}
    for rows in catalogs.values():
        for row in rows:
            path = Path(row["image_path"])
            key = str(path)
            if key not in hash_cache:
                hash_cache[key] = file_sha256(path)
            row["sha256"] = hash_cache[key]

    duplicate_summary = {}
    for view_id, rows in catalogs.items():
        by_hash = defaultdict(list)
        for row in rows:
            by_hash[row["sha256"]].append(row)

        within_patient_duplicate_count = 0
        train_skipped_count = 0
        for digest, group in by_hash.items():
            patients = {row["patient_id"] for row in group}
            if len(patients) > 1:
                examples = [row["image_path"] for row in group[:5]]
                raise RuntimeError(
                    f"{view_id} 发现跨患者完全重复图像 "
                    f"sha256={digest}：{examples}"
                )
            if len(group) > 1:
                within_patient_duplicate_count += len(group) - 1
                primary = group[0]
                for duplicate in group[1:]:
                    duplicate["exact_duplicate_of"] = primary["image_path"]

        train_seen = {}
        for row in rows:
            if row["experiment_role"] != "train":
                continue
            previous = train_seen.get(row["sha256"])
            if previous is None:
                train_seen[row["sha256"]] = row["image_path"]
                row["train_keep_after_exact_dedup"] = "yes"
            else:
                row["train_keep_after_exact_dedup"] = "no"
                row["exact_duplicate_of"] = previous
                row["include_in_common_experiment"] = "no"
                row["exclusion_reason"] = (
                    "同患者训练视图字节级完全重复"
                )
                train_skipped_count += 1

        duplicate_summary[view_id] = {
            "within_patient_duplicate_count_all_splits": (
                within_patient_duplicate_count
            ),
            "train_exact_duplicate_skipped_count": train_skipped_count,
        }
    return duplicate_summary


def write_view_outputs(output, view_id, rows):
    catalog_path = output / "catalogs" / f"{view_id}.csv"
    write_csv(catalog_path, rows)

    train_rows = [
        row for row in rows
        if row["experiment_role"] == "train"
        and row["train_keep_after_exact_dedup"] == "yes"
    ]
    val_rows = [
        row for row in rows
        if row["experiment_role"] == "val"
    ]
    test_rows = [
        row for row in rows
        if row["experiment_role"] == "test"
    ]
    write_csv(output / "experiments" / view_id / "train.csv", train_rows)
    write_csv(output / "experiments" / view_id / "val.csv", val_rows)
    write_csv(output / "experiments" / view_id / "test.csv", test_rows)
    return {
        "catalog": str(catalog_path),
        "catalog_row_count": len(rows),
        "train_row_count_after_exact_dedup": len(train_rows),
        "val_row_count": len(val_rows),
        "test_row_count": len(test_rows),
        "held_out_pip_not_in_common_eval_count": sum(
            row["experiment_role"] == "held_out_pip_not_in_common_eval"
            for row in rows
        ),
    }


def main():
    args = parse_args()
    validate_args(args)
    stage2, pip_labels, stage4 = load_and_validate_inputs(args)
    pip_paths = {
        row["relative_path"] for row in pip_labels
    }
    split, split_by_patient, patient_rows, clean_paths = (
        build_patient_split(
            stage2,
            pip_paths,
            args.seed,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
    )
    cohort_by_patient = {
        row["patient_id"]: row["cohort"] for row in patient_rows
    }
    catalogs = build_catalogs(
        stage2,
        stage4,
        clean_paths,
        split_by_patient,
        cohort_by_patient,
    )
    duplicate_summary = add_hashes_and_exact_dedup(catalogs)

    args.output.mkdir(parents=True, exist_ok=False)
    split_json_path = args.output / "patient_split.json"
    with split_json_path.open("w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    write_csv(args.output / "patient_split.csv", patient_rows)

    view_outputs = {}
    for view_id, rows in catalogs.items():
        view_outputs[view_id] = write_view_outputs(
            args.output, view_id, rows
        )

    split_summary = {}
    for split_name in ["train", "val", "test"]:
        patients = [
            row for row in patient_rows if row["split"] == split_name
        ]
        split_summary[split_name] = {
            "patient_count": len(patients),
            "cohort_counts": dict(Counter(
                row["cohort"] for row in patients
            )),
            "clean_image_count": sum(
                int(row["clean_image_count"]) for row in patients
            ),
            "pip_image_count": sum(
                int(row["pip_image_count"]) for row in patients
            ),
        }

    created_at = datetime.now().astimezone().isoformat()
    summary = {
        "created_at": created_at,
        "status": "candidate_manifests_validated",
        "split_policy": {
            "seed": args.seed,
            "clean_capable_patient_ratios": {
                "train": args.train_ratio,
                "val": args.val_ratio,
                "test": args.test_ratio,
            },
            "pip_only_current_patients": "train_only",
            "common_val_test": "clean_images_only",
        },
        "split_summary": split_summary,
        "views": view_outputs,
        "exact_duplicate_summary": duplicate_summary,
        "patient_split_json": str(split_json_path),
        "important_interpretation": (
            f"E1/E2 各补充 {len(pip_labels)} 张画中画派生图；属于 val/test "
            "患者的画中画派生图仅保留追溯，不进入公共干净评价或训练"
        ),
    }
    with (args.output / "view_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    script_path = Path(__file__)
    manifest = {
        "created_at": created_at,
        "script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "inputs": {
            "stage2_mapping": str(args.stage2_mapping),
            "stage2_mapping_sha256": file_sha256(args.stage2_mapping),
            "pip_labels": str(args.pip_labels),
            "pip_labels_sha256": file_sha256(args.pip_labels),
            "stage4_mapping": str(args.stage4_mapping),
            "stage4_mapping_sha256": file_sha256(args.stage4_mapping),
            "box_qc_status": str(args.box_qc_status),
            "box_qc_status_sha256": file_sha256(args.box_qc_status),
        },
        "output": str(args.output),
        "view_summary": str(args.output / "view_summary.json"),
    }
    with (args.output / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
