#!/usr/bin/env python3
"""画中画无生成裁剪候选：向下裁剪与向右裁剪。

本脚本只处理已人工确认的画中画图像，不修改任何输入文件。每张图生成：

1. down：去掉画中画下边界以上区域；
2. right：去掉画中画右边界左侧区域。

两个方向随后都在剩余区域内选择“有效 FOV 像素最多”的最大正方形，
用于模拟 SDXL 方形训练输入。脚本输出候选图、三联预览、指标表和运行清单。
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import (
    STAGE2_MAPPING,
    STAGE2_MASKED,
    STAGE2_MASKS,
    STAGE3_QUALITY_FLAGS,
    STAGE3_PIP_LABELS,
    STAGE3_VALIDATION_LIST,
)

MASKED_ROOT = STAGE2_MASKED
MASK_ROOT = STAGE2_MASKS
QUALITY_FLAGS = STAGE3_QUALITY_FLAGS
PIP_LABELS = STAGE3_PIP_LABELS
OUTPUT_ROOT = Path(
    str(STAGE3_VALIDATION_LIST.parent)
    + "/08_画中画无生成裁剪_跨患者30例"
)

DEFAULT_SAMPLE_COUNT = 30
DEFAULT_MARGIN_RATIO = 0.008
PANEL_SIZE = 480


def parse_args():
    parser = argparse.ArgumentParser(description="画中画两方向无生成裁剪验证")
    parser.add_argument("--masked-root", type=Path, default=MASKED_ROOT)
    parser.add_argument("--mask-root", type=Path, default=MASK_ROOT)
    parser.add_argument("--stage2-mapping", type=Path, default=STAGE2_MAPPING)
    parser.add_argument("--quality-flags", type=Path, default=QUALITY_FLAGS)
    parser.add_argument("--pip-labels", type=Path, default=PIP_LABELS)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT
    )
    parser.add_argument(
        "--margin-ratio", type=float, default=DEFAULT_MARGIN_RATIO
    )
    return parser.parse_args()


def validate_args(args):
    for path in [
        args.masked_root,
        args.mask_root,
        args.stage2_mapping,
        args.quality_flags,
        args.pip_labels,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{args.output}")
    if args.sample_count < 2 or args.sample_count % 2:
        raise ValueError("--sample-count 必须是大于等于 2 的偶数")
    if not 0 <= args.margin_ratio <= 0.05:
        raise ValueError("--margin-ratio 必须位于 [0, 0.05]")


def load_csv(path):
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_rect(text):
    values = [int(value) for value in text.split(",")]
    if len(values) != 4:
        raise ValueError(f"非法 rect_xywh：{text}")
    return tuple(values)


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f"图像读取失败：{path}")
    return image


def image_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integral_rect_sum(integral, x, y, width, height):
    x2 = x + width
    y2 = y + height
    return (
        integral[y2, x2]
        - integral[y, x2]
        - integral[y2, x]
        + integral[y, x]
    )


def best_square(mask, x0, y0, width, height):
    """在给定矩形内选择有效 FOV 像素最多的最大正方形。"""
    side = min(width, height)
    if side <= 0:
        raise ValueError("候选区域为空")
    binary = (mask > 0).astype(np.uint8)
    integral = cv2.integral(binary, sdepth=cv2.CV_64F)

    best = None
    if width > height:
        for x in range(x0, x0 + width - side + 1):
            score = integral_rect_sum(integral, x, y0, side, side)
            item = (score, -abs((x + side / 2) - mask.shape[1] / 2), x, y0)
            if best is None or item > best:
                best = item
    elif height > width:
        for y in range(y0, y0 + height - side + 1):
            score = integral_rect_sum(integral, x0, y, side, side)
            item = (score, -abs((y + side / 2) - mask.shape[0] / 2), x0, y)
            if best is None or item > best:
                best = item
    else:
        score = integral_rect_sum(integral, x0, y0, side, side)
        best = (score, 0, x0, y0)

    return int(best[2]), int(best[3]), int(side), float(best[0])


def black_ratio(image, threshold=12):
    if image.size == 0:
        return 1.0
    black = np.all(image <= threshold, axis=2)
    return float(black.mean())


def make_main_fov_mask(mask, pip_rect_with_margin):
    """仅用于指标计算：将被画中画覆盖的左上矩形从分母中移除。"""
    main = (mask > 0).astype(np.uint8)
    _, _, rect_w, rect_h = pip_rect_with_margin
    main[:rect_h, :rect_w] = 0
    return main


def crop_candidate(image, main_mask, direction, cut_x, cut_y):
    height, width = image.shape[:2]
    if direction == "down":
        region = (0, cut_y, width, height - cut_y)
    elif direction == "right":
        region = (cut_x, 0, width - cut_x, height)
    else:
        raise ValueError(direction)

    x0, y0, region_w, region_h = region
    square_x, square_y, side, fov_pixels = best_square(
        main_mask, x0, y0, region_w, region_h
    )
    square_image = image[
        square_y:square_y + side,
        square_x:square_x + side,
    ]
    square_mask = main_mask[
        square_y:square_y + side,
        square_x:square_x + side,
    ]
    return {
        "image": square_image,
        "mask": square_mask,
        "crop_xywh": (square_x, square_y, side, side),
        "side": side,
        "fov_pixels": fov_pixels,
        "black_ratio": black_ratio(square_image),
    }


def normalise_features(items):
    matrix = np.array([item["features"] for item in items], dtype=np.float64)
    mins = matrix.min(axis=0)
    spans = matrix.max(axis=0) - mins
    spans[spans == 0] = 1
    return (matrix - mins) / spans


def diverse_select(items, count):
    """按几何特征执行确定性的最远点采样。"""
    if len(items) < count:
        raise ValueError(f"候选患者不足：需要 {count}，实际 {len(items)}")
    features = normalise_features(items)
    centroid = features.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(features - centroid, axis=1)))
    selected = [first]
    min_distances = np.linalg.norm(features - features[first], axis=1)

    while len(selected) < count:
        min_distances[selected] = -1
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = np.linalg.norm(
            features - features[next_index], axis=1
        )
        min_distances = np.minimum(min_distances, distances)
    return [items[index] for index in selected]


def build_selection(args):
    mapping_rows = load_csv(args.stage2_mapping)
    all_paths_by_patient = defaultdict(list)
    for row in mapping_rows:
        all_paths_by_patient[
            row["relative_path"].split("/")[0]
        ].append(row["relative_path"])

    quality_by_path = {
        row["relative_path"]: row for row in load_csv(args.quality_flags)
    }
    pip_rows = load_csv(args.pip_labels)
    pip_paths = {row["relative_path"] for row in pip_rows}
    pip_paths_by_patient = defaultdict(list)
    for row in pip_rows:
        pip_paths_by_patient[row["patient_id"]].append(row["relative_path"])

    all_pip_patients = {
        patient for patient, paths in all_paths_by_patient.items()
        if paths and all(path in pip_paths for path in paths)
    }

    patient_items = []
    for patient, paths in sorted(pip_paths_by_patient.items()):
        candidates = []
        for relative_path in paths:
            quality = quality_by_path.get(relative_path)
            if quality is None or not quality["rect_xywh"]:
                continue
            image_path = args.masked_root / relative_path
            image = read_image(image_path)
            height, width = image.shape[:2]
            _, _, rect_w, rect_h = parse_rect(quality["rect_xywh"])
            rect_area_ratio = (rect_w * rect_h) / (width * height)
            features = [
                rect_w / width,
                rect_h / height,
                rect_area_ratio,
                float(quality["pip_score"]),
                width / height,
                1.0 if quality["proposal_type"].endswith("fallback") else 0.0,
            ]
            candidates.append({
                "patient_id": patient,
                "relative_path": relative_path,
                "quality": quality,
                "width": width,
                "height": height,
                "features": features,
            })
        if not candidates:
            raise ValueError(f"患者没有可用矩形：{patient}")
        # 每名患者只选一张：优先选择最接近本患者画中画面积中位数的图。
        ratios = sorted(item["features"][2] for item in candidates)
        median = ratios[len(ratios) // 2]
        representative = min(
            candidates,
            key=lambda item: (
                abs(item["features"][2] - median),
                item["relative_path"],
            ),
        )
        representative["selection_group"] = (
            "all_current_images_pip"
            if patient in all_pip_patients
            else "mixed_current_images"
        )
        patient_items.append(representative)

    half = args.sample_count // 2
    all_pip_items = [
        item for item in patient_items
        if item["selection_group"] == "all_current_images_pip"
    ]
    mixed_items = [
        item for item in patient_items
        if item["selection_group"] == "mixed_current_images"
    ]
    selected = (
        diverse_select(all_pip_items, half)
        + diverse_select(mixed_items, half)
    )
    return sorted(selected, key=lambda item: (
        item["selection_group"], item["patient_id"]
    ))


def fit_panel(image, panel_size=PANEL_SIZE):
    height, width = image.shape[:2]
    scale = min(panel_size / width, panel_size / height)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
    x = (panel_size - resized.shape[1]) // 2
    y = (panel_size - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def labelled_panel(image, title, detail):
    panel = fit_panel(image)
    header = np.full((52, PANEL_SIZE, 3), 35, dtype=np.uint8)
    cv2.putText(
        header, title[:60], (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        header, detail[:88], (7, 43),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
        (0, 255, 255), 1, cv2.LINE_AA,
    )
    return np.vstack([header, panel])


def original_panel(image, rect):
    marked = image.copy()
    _, _, rect_w, rect_h = rect
    cv2.rectangle(marked, (0, 0), (rect_w, rect_h), (0, 255, 255), 3)
    return labelled_panel(
        marked, "original masked + PiP box",
        f"size={image.shape[1]}x{image.shape[0]} pip={rect_w}x{rect_h}",
    )


def write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(
        str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not ok:
        raise OSError(f"图像写入失败：{path}")


def process_item(args, item):
    relative_path = item["relative_path"]
    image = read_image(args.masked_root / relative_path)
    mask = read_image(
        args.mask_root / Path(relative_path).with_suffix(".png"),
        cv2.IMREAD_GRAYSCALE,
    )
    if image.shape[:2] != mask.shape:
        raise ValueError(f"图像与掩膜尺寸不一致：{relative_path}")

    rect = parse_rect(item["quality"]["rect_xywh"])
    _, _, rect_w, rect_h = rect
    margin = max(
        4, int(round(min(image.shape[:2]) * args.margin_ratio))
    )
    cut_x = min(image.shape[1] - 1, rect_w + margin)
    cut_y = min(image.shape[0] - 1, rect_h + margin)
    rect_margin = (0, 0, cut_x, cut_y)
    main_mask = make_main_fov_mask(mask, rect_margin)
    main_fov_pixels = int(main_mask.sum())
    if main_fov_pixels <= 0:
        raise ValueError(f"主FOV为空：{relative_path}")

    down = crop_candidate(image, main_mask, "down", cut_x, cut_y)
    right = crop_candidate(image, main_mask, "right", cut_x, cut_y)
    for candidate in [down, right]:
        candidate["fov_retention"] = (
            candidate["fov_pixels"] / main_fov_pixels
        )
        candidate["fov_coverage"] = float(candidate["mask"].mean())

    down_path = args.output / "candidates" / "down" / relative_path
    right_path = args.output / "candidates" / "right" / relative_path
    write_image(down_path, down["image"])
    write_image(right_path, right["image"])

    preview = np.hstack([
        original_panel(image, rect_margin),
        labelled_panel(
            down["image"], "down: remove top band",
            (
                f"retain={down['fov_retention']:.1%} "
                f"FOV={down['fov_coverage']:.1%} "
                f"black={down['black_ratio']:.1%} side={down['side']}"
            ),
        ),
        labelled_panel(
            right["image"], "right: remove left band",
            (
                f"retain={right['fov_retention']:.1%} "
                f"FOV={right['fov_coverage']:.1%} "
                f"black={right['black_ratio']:.1%} side={right['side']}"
            ),
        ),
    ])
    preview_path = args.output / "previews" / relative_path
    write_image(preview_path, preview)

    recommended = (
        "down"
        if (
            down["fov_retention"],
            -down["black_ratio"],
            down["side"],
        ) >= (
            right["fov_retention"],
            -right["black_ratio"],
            right["side"],
        )
        else "right"
    )
    return {
        "relative_path": relative_path,
        "patient_id": item["patient_id"],
        "selection_group": item["selection_group"],
        "proposal_type": item["quality"]["proposal_type"],
        "pip_score": item["quality"]["pip_score"],
        "source_width": image.shape[1],
        "source_height": image.shape[0],
        "pip_rect_xywh": ",".join(map(str, rect)),
        "margin_pixels": margin,
        "cut_x": cut_x,
        "cut_y": cut_y,
        "main_fov_pixels": main_fov_pixels,
        "down_crop_xywh": ",".join(map(str, down["crop_xywh"])),
        "down_side": down["side"],
        "down_fov_retention": round(down["fov_retention"], 6),
        "down_fov_coverage": round(down["fov_coverage"], 6),
        "down_black_ratio": round(down["black_ratio"], 6),
        "right_crop_xywh": ",".join(map(str, right["crop_xywh"])),
        "right_side": right["side"],
        "right_fov_retention": round(right["fov_retention"], 6),
        "right_fov_coverage": round(right["fov_coverage"], 6),
        "right_black_ratio": round(right["black_ratio"], 6),
        "metric_recommendation": recommended,
        "manual_preference": "",
        "manual_note": "",
        "down_output": str(down_path),
        "right_output": str(right_path),
        "preview": str(preview_path),
    }


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main():
    args = parse_args()
    validate_args(args)
    selected = build_selection(args)
    args.output.mkdir(parents=True, exist_ok=False)

    rows = []
    for index, item in enumerate(selected, start=1):
        row = process_item(args, item)
        rows.append(row)
        print(
            f"[{index}/{len(selected)}] "
            f"{row['selection_group']} {row['relative_path']} "
            f"recommend={row['metric_recommendation']}"
        )

    mapping_path = args.output / "mapping.csv"
    with mapping_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__)),
        "script_sha256": script_sha256(),
        "sample_count": len(rows),
        "selection_groups": {
            group: sum(row["selection_group"] == group for row in rows)
            for group in sorted({row["selection_group"] for row in rows})
        },
        "margin_ratio": args.margin_ratio,
        "metric_recommendations": {
            direction: sum(
                row["metric_recommendation"] == direction for row in rows
            )
            for direction in ["down", "right"]
        },
        "mean_down_fov_retention": float(np.mean([
            float(row["down_fov_retention"]) for row in rows
        ])),
        "mean_right_fov_retention": float(np.mean([
            float(row["right_fov_retention"]) for row in rows
        ])),
        "important_limitation": (
            "指标只衡量FOV与黑区，不知道病灶位置；最终选择必须人工确认"
        ),
        "mapping": str(mapping_path),
    }
    with (args.output / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
