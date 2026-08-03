#!/usr/bin/env python3
"""Image-cropping 可复用流水线的集中路径配置。

所有路径都有默认值（与结肠镜项目一致），均可通过环境变量覆盖。

新项目复用本流水线时：
    1. 设置 IMGCROP_OUTPUT_ROOT 指向本项目的结果根目录，各阶段路径会自动跟随；
    2. 如需更细控制，可单独设置某个阶段的路径环境变量；
    3. 设置 ENDOSCOPY_FOV_MODEL_PATH 指向本项目的 FOV 分割 ONNX 模型。

本文件不修改任何输入图像；所有脚本只读取这些路径配置。
"""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def _p(name: str, default: str | Path) -> Path:
    """读取环境变量作为路径；未设置时返回默认值。"""
    return Path(os.environ.get(name, default))


# ── FOV 分割模型（shiye_V1 ONNX）──
# 默认指向仓库内 model.onnx（通过 Git LFS 管理）；新项目可覆盖。
MODEL_PATH = _p("ENDOSCOPY_FOV_MODEL_PATH", REPO_ROOT / "model.onnx")

# ── 项目根目录约定 ──
# 原始输入图像根目录（第一阶段输入）。
INPUT_ROOT = _p("IMGCROP_INPUT_ROOT", "./数据/扁平数据")

# 项目结果根目录：新项目整体重定向的入口。
OUTPUT_ROOT = _p("IMGCROP_OUTPUT_ROOT", "./结果/正式预处理")

# ── 第一阶段：亮度四边裁剪 ──
STAGE1_OUTPUT = _p(
    "IMGCROP_STAGE1_OUTPUT",
    str(OUTPUT_ROOT / "01_矩形裁剪_brightness_v2"),
)
STAGE1_CROPS = _p(
    "IMGCROP_STAGE1_CROPS",
    str(STAGE1_OUTPUT / "crops"),
)

# ── 第二阶段：FOV 遮罩（shiye_V1 ONNX）──
STAGE2_OUTPUT = _p(
    "IMGCROP_STAGE2_OUTPUT",
    str(OUTPUT_ROOT / "02_FOV遮罩_shiyeV1"),
)
STAGE2_MASKED = _p(
    "IMGCROP_STAGE2_MASKED",
    str(STAGE2_OUTPUT / "masked"),
)
STAGE2_MASKS = _p(
    "IMGCROP_STAGE2_MASKS",
    str(STAGE2_OUTPUT / "masks"),
)
STAGE2_MAPPING = _p(
    "IMGCROP_STAGE2_MAPPING",
    str(STAGE2_OUTPUT / "mapping.csv"),
)

# ── 第三阶段：画中画筛查 ──
STAGE3_OUTPUT = _p(
    "IMGCROP_STAGE3_OUTPUT",
    str(OUTPUT_ROOT / "03_画中画筛查_v2_全量"),
)
STAGE3_QUALITY_FLAGS = _p(
    "IMGCROP_STAGE3_QUALITY_FLAGS",
    str(STAGE3_OUTPUT / "quality_flags.csv"),
)
STAGE3_PIP_LABELS = _p(
    "IMGCROP_STAGE3_PIP_LABELS",
    str(STAGE3_OUTPUT / "confirmed_pip_labels.csv"),
)
STAGE3_VALIDATION_OUTPUT = _p(
    "IMGCROP_STAGE3_VALIDATION_OUTPUT",
    str(OUTPUT_ROOT / "90_历史验证归档" / "03_画中画筛查_v2_30例"),
)
STAGE3_VALIDATION_LIST = _p(
    "IMGCROP_STAGE3_VALIDATION_LIST",
    str(
        OUTPUT_ROOT / "90_历史验证归档"
        / "07_最终候选验证链" / "06_亮度四边裁剪v2_跨患者30例" / "mapping.csv"
    ),
)

# ── 第四阶段：right/notch 双分支 ──
STAGE4_OUTPUT = _p(
    "IMGCROP_STAGE4_OUTPUT",
    str(OUTPUT_ROOT / "04_画中画right_notch双分支_全量候选"),
)

# ── 第五阶段：患者级划分与三视图 manifest ──
SPLIT_OUTPUT = _p(
    "IMGCROP_SPLIT_OUTPUT",
    str(OUTPUT_ROOT / "05_患者级划分与三视图manifest"),
)

# ── 参考文件（随仓库维护）──
PIP_VALIDATION_LABELS = _p(
    "PIP_VALIDATION_LABELS",
    REPO_ROOT / "reference" / "validation_labels.csv",
)
PIP_OVERRIDES = _p(
    "PIP_OVERRIDES",
    REPO_ROOT / "reference" / "pip_rect_overrides.csv",
)
