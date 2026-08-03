#!/usr/bin/env python3
"""记录画中画框初筛候选全部人工通过的复核结论。

本脚本不修改初筛输出。它读取全量 mapping 和初筛 candidate_list，
为初筛候选记录 human_pass，并将未触发初筛的图像记录为
initial_screen_pass，输出独立的逐图最终状态和汇总。
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


from config import STAGE4_OUTPUT

STAGE_ROOT = STAGE4_OUTPUT
MAPPING = STAGE_ROOT / "mapping.csv"
CANDIDATES = STAGE_ROOT / "box_qc_initial_screen/candidate_list.csv"
OUTPUT_ROOT = STAGE_ROOT / "box_qc_human_review"


def parse_args():
    parser = argparse.ArgumentParser(
        description="记录画中画框初筛候选全部人工通过"
    )
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def validate_args(args):
    for path in [args.mapping, args.candidates]:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
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


def main():
    args = parse_args()
    validate_args(args)
    mapping_rows = load_csv(args.mapping)
    candidate_rows = load_csv(args.candidates)

    mapping_paths = {
        row["relative_path"] for row in mapping_rows
    }
    candidate_paths = {
        row["relative_path"] for row in candidate_rows
    }
    if len(mapping_paths) != len(mapping_rows):
        raise ValueError("全量 mapping 存在重复 relative_path")
    if len(candidate_paths) != len(candidate_rows):
        raise ValueError("初筛候选存在重复 relative_path")
    if not candidate_paths <= mapping_paths:
        raise ValueError("初筛候选包含全量 mapping 之外的图像")

    reviewed_at = datetime.now().astimezone().isoformat()
    human_rows = []
    for row in candidate_rows:
        human_rows.append({
            "relative_path": row["relative_path"],
            "patient_id": row["patient_id"],
            "screen_reasons": row["screen_reasons"],
            "effective_rect_xywh": row["effective_rect_xywh"],
            "review_status": "human_pass",
            "review_note": "用户完成4×4联图人眼核查，框完整",
            "reviewed_at": reviewed_at,
        })

    final_rows = []
    for row in mapping_rows:
        is_candidate = row["relative_path"] in candidate_paths
        final_rows.append({
            "relative_path": row["relative_path"],
            "patient_id": row["patient_id"],
            "effective_rect_xywh": row["effective_rect_xywh"],
            "rect_source": row["rect_source"],
            "initial_screen_candidate": "yes" if is_candidate else "no",
            "final_qc_basis": (
                "human_pass"
                if is_candidate
                else "initial_screen_pass"
            ),
            "final_box_qc_status": "pass",
            "reviewed_at": reviewed_at if is_candidate else "",
        })

    args.output.mkdir(parents=True, exist_ok=False)
    human_path = args.output / "human_reviewed_candidates.csv"
    final_path = args.output / "final_box_qc_status.csv"
    write_csv(human_path, human_rows)
    write_csv(final_path, final_rows)

    script_path = Path(__file__)
    summary = {
        "created_at": reviewed_at,
        "status": "box_qc_passed",
        "script": str(script_path),
        "script_sha256": file_sha256(script_path),
        "inputs": {
            "mapping": str(args.mapping),
            "mapping_sha256": file_sha256(args.mapping),
            "initial_candidates": str(args.candidates),
            "initial_candidates_sha256": file_sha256(args.candidates),
        },
        "total_image_count": len(mapping_rows),
        "human_reviewed_candidate_count": len(candidate_rows),
        "human_pass_count": len(candidate_rows),
        "human_needs_override_count": 0,
        "human_uncertain_count": 0,
        "initial_screen_non_candidate_count": (
            len(mapping_rows) - len(candidate_rows)
        ),
        "final_pass_count": len(final_rows),
        "final_fail_count": 0,
        "new_override_count": 0,
        "human_reviewed_candidates": str(human_path),
        "final_box_qc_status": str(final_path),
        "important_scope": (
            "22张初筛候选经人工逐图通过；其余278张为初筛未触发，"
            "并非逐图人工复核"
        ),
    }
    with (args.output / "box_qc_final_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
