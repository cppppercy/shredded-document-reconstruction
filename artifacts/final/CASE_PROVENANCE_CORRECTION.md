# 最终案例来源更正说明

本文件保留对一次交付前来源问题的可追溯说明，不修改 `artifacts/v0.1/` 至 `artifacts/v0.4/` 的任何历史实验记录。

## 发现的问题

旧版 `scripts/prepare_final_analysis.py` 默认读取 `artifacts/v0.4/ablation.json`，但把图片写入 `artifacts/final/cases/`。因此旧 `manifest.json` 的来源文字、种子和最终研究页所声称的冻结最终集不一致。

## 更正方式

- 默认输入现为 `artifacts/final/independent_evaluation.json`；历史 V0.4 JSON 会被脚本拒绝，避免再次覆盖最终案例。
- 新 manifest 使用 schema 2，记录输入相对路径、输入 SHA-256、精确源行 ID、种子、条件、打乱种子、保存的原始顺序预测、重放的输入位置顺序和每张案例图 SHA-256。
- 写图前，脚本用与最终独立评测相同的 V0.4 训练配置确定性重放案例，并核对两种方法的预测原始顺序。
- 自动测试会检查 manifest 的输入哈希、源行、预测顺序和图片哈希。

Git 历史保留更正前版本；当前 `artifacts/final/cases/` 仅代表冻结最终独立评测中的四类真实案例。
