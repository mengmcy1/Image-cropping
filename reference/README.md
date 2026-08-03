# reference/ — 项目自定义参考文件（不入仓库）

本目录存放各项目自定义的参考数据，**不随仓库分发**，因为其中包含患者级标识信息。

新项目使用本流水线时，请在此目录自行创建：

| 文件 | 用途 | 格式 |
|------|------|------|
| `validation_labels.csv` | `pip_detector.py --validate-only` 的校准标签 | `relative_path,label,review_note`，label ∈ {pip, non_pip} |
| `pip_rect_overrides.csv` | 画中画框人工修正 | `relative_path,original_rect_xywh,corrected_rect_xywh,correction_reason,review_status,reviewed_at` |

可从不含真实患者信息的模板开始：

```bash
cp reference/validation_labels.example.csv reference/validation_labels.csv
cp reference/pip_rect_overrides.example.csv reference/pip_rect_overrides.csv
```

这两个实际 CSV 已被 `.gitignore` 精确排除。`pip_rect_overrides.csv` 是可选的；
没有人工修正时，阶段 4 会直接使用自动框。

`config.py` 中的默认路径相对于本仓库定位，不受启动命令的当前工作目录影响。
