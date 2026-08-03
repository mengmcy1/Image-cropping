# Image-cropping — 内镜图像裁剪与画中画处理流水线

可复用的内镜图像预处理流水线，用于清除黑边、设备界面叠加层和画中画，
为下游生成式模型（如 SDXL LoRA）训练提供干净数据。

## 适用场景

本流水线针对内镜图像（白光结肠镜、胃镜等）的常见伪影：

| 伪影 | 处理方法 |
|------|---------|
| 纯黑边框（圆形 FOV 不填满矩形画面） | 亮度轮廓 + 四边扫描 |
| FOV 外的设备文字、刻度、工具栏 | ONNX 视野分割（shiye_V1） |
| 左上角画中画（历史画面叠加） | 自动检测 + 人工复核 + right/notch 双分支 |

## 环境依赖

```bash
# 推荐独立 conda 环境，避免污染训练环境
conda create -n image-tools python=3.13 pip
conda activate image-tools
pip install -r requirements.txt
```

依赖：`opencv-python-headless`、`onnxruntime`、`numpy`。

`model.onnx` 是视野分割模型（shiye_V1），通过 Git LFS 管理，
`git clone` 后需执行 `git lfs pull` 才能获得完整权重。

## 流水线总览

```
原始图像（患者一级目录/图像）
   │  brightness_crop_v2.py
   ▼
01_矩形裁剪_brightness_v2/crops     ← 去黑边
   │  fov_mask_v1.py
   ▼
02_FOV遮罩_shiyeV1/masked          ← 去 FOV 外 UI 叠加
   │  pip_detector.py
   ▼
03_画中画筛查_v2_全量/quality_flags.csv ← 画中画自动检测
   │  人工复核 → confirmed_pip_labels.csv
   ▼
04_画中画right_notch双分支_全量候选    ← right/notch 处理
   │  pip_dual_branch_full.py
   ▼
05_患者级划分与三视图manifest          ← 患者级 train/val/test 划分
   │  data_split/
   ▼
E0/E1/E2 训练清单
```

## 各阶段用法

输入和输出路径可通过参数或环境变量整体重定向。仓库内模型和参考文件默认
相对于 `config.py` 定位，因此可以从其他项目目录调用这些脚本。

### 阶段 1：亮度四边裁剪（去黑边）

```bash
python brightness_crop_v2.py \
  --input <原始图像目录> \
  --output <输出目录>
```

### 阶段 2：FOV 遮罩（shiye_V1 ONNX）

```bash
python fov_mask_v1.py \
  --onnx <model.onnx> \
  --input <阶段1输出/crops> \
  --output <阶段2输出>
```

### 阶段 3：画中画检测

```bash
# 先准备不提交 Git 的项目标签，再在验证集上校准
cp reference/validation_labels.example.csv reference/validation_labels.csv
# 填写项目自己的匿名化标签后：
python pip_detector.py --validate-only \
  --validation-list <验证图像清单.csv> \
  --validation-labels reference/validation_labels.csv

# 全量筛查
python pip_detector.py
```

输出 `quality_flags.csv` 和候选预览，**不删除任何图像**。
画中画结论必须人工复核后写入 `confirmed_pip_labels.csv`。

### 阶段 4：right/notch 双分支

```bash
python pip_dual_branch_full.py
```

阶段 4 必须提供人工确认的 `confirmed_pip_labels.csv`。人工修正框文件是可选的；
不存在时直接使用自动框。需要修正时可从模板开始：

```bash
cp reference/pip_rect_overrides.example.csv reference/pip_rect_overrides.csv
```

### 阶段 5：患者级划分 + 三视图 manifest

```bash
python data_split/build_three_view_manifests.py
python data_split/validate_three_view_manifests.py
python data_split/freeze_three_view_release.py
```

默认按有干净图患者 80/10/10 划分，当前只有画中画图的患者仅进入 train。
比例可通过 `--train-ratio`、`--val-ratio`、`--test-ratio` 修改。脚本不再要求
固定患者数或固定图像数，但 E0/E1/E2 仍表示 clean / +right / +notch 三视图。

## 路径参数化（config.py）

所有脚本通过 `config.py` 读取路径默认值，均可通过环境变量覆盖：

| 环境变量 | 默认值 | 用途 |
|----------|--------|------|
| `ENDOSCOPY_FOV_MODEL_PATH` | `{仓库根}/model.onnx` | FOV 分割模型 |
| `IMGCROP_INPUT_ROOT` | `./数据/扁平数据` | 原始图像输入 |
| `IMGCROP_OUTPUT_ROOT` | `./结果/正式预处理` | 项目结果根目录 |
| `IMGCROP_STAGE1_OUTPUT` | `{OUTPUT_ROOT}/01_矩形裁剪_brightness_v2` | 阶段 1 输出 |
| `IMGCROP_STAGE2_OUTPUT` | `{OUTPUT_ROOT}/02_FOV遮罩_shiyeV1` | 阶段 2 输出 |
| `IMGCROP_STAGE3_OUTPUT` | `{OUTPUT_ROOT}/03_画中画筛查_v2_全量` | 阶段 3 输出 |
| `IMGCROP_STAGE4_OUTPUT` | `{OUTPUT_ROOT}/04_画中画right_notch双分支_全量候选` | 阶段 4 输出 |
| `IMGCROP_SPLIT_OUTPUT` | `{OUTPUT_ROOT}/05_患者级划分与三视图manifest` | 阶段 5 输出 |
| `PIP_VALIDATION_LABELS` | `{仓库根}/reference/validation_labels.csv` | 画中画验证标签（项目自定义，不入仓库） |
| `PIP_OVERRIDES` | `{仓库根}/reference/pip_rect_overrides.csv` | 可选人工修正框（项目自定义，不入仓库） |

**新项目最小配置**：设置 `IMGCROP_INPUT_ROOT` 和 `IMGCROP_OUTPUT_ROOT`。
只有替换仓库内模型时才需要设置 `ENDOSCOPY_FOV_MODEL_PATH`。

## 安全规则

- 原始图像始终只读，不覆盖、不重命名、不原地预处理。
- 所有脚本输出到独立派生目录，非空输出目录拒绝覆盖。
- 画中画自动检测只生成候选和预览，去留必须人工确认。
- 患者级数据划分是铁律：同一患者的所有图像必须属于同一集合。
- 生成式训练只允许使用冻结的 `experiments/<view>/train.csv` 清单，禁止全目录扫描。
- `reference/validation_labels.csv` 和 `reference/pip_rect_overrides.csv` 含项目级
  标识，已由 `.gitignore` 排除；只能提交不含真实记录的 `*.example.csv`。

## 目录结构

```
Image-cropping/
├── config.py                     # 路径配置（env 可覆盖）
├── brightness_crop_v2.py         # 阶段 1：亮度四边裁剪
├── onnx_infer.py                 # shiye_V1 ONNX 推理共享库
├── fov_mask_v1.py                # 阶段 2：FOV 遮罩
├── pip_detector.py               # 阶段 3：画中画检测
├── pip_crop_candidates.py        # 画中画候选共享库
├── pip_dual_branch_full.py       # 阶段 4：right/notch 双分支
├── pip_box_qc_initial_screen.py  # 画中画框质控初筛
├── pip_box_qc_record_all_pass.py # 质控结果记录
├── data_split/                   # 阶段 5：患者级划分与三视图
│   ├── build_three_view_manifests.py
│   ├── validate_three_view_manifests.py
│   └── freeze_three_view_release.py
├── reference/                    # 项目自定义参考（真实 CSV 不入仓库）
│   ├── README.md                 # 参考文件格式说明
│   ├── validation_labels.example.csv
│   └── pip_rect_overrides.example.csv
├── model.onnx                    # shiye_V1 视野分割模型（LFS）
└── requirements.txt
```
