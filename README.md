# Automatic Reconstruction of Shredded Documents

《机器学习导论》Final Project：**基于局部特征学习与全局优化的碎纸文档自动重建**。

程序将完整页面模拟切成 8、10 或 12 条竖直纸条并随机打乱；重建器只读取打乱后的像素，正确顺序仅用于训练标签和预测后的评测。

> 当前是规则竖直纸条的研究原型，不等同于物理碎纸扫描恢复；不处理旋转、重叠、任意碎片形状或缺失纸片。

![V0.3 自动选择的合成成功与失败案例](artifacts/v0.3/selected-cases-grid.png)

## V0.4 最终确认与课堂展示材料

V0.4 不再把 V0.3 已公布的最终集用于调参，而是创建了新的、按原始文档身份隔离的训练（24 个文档族）/验证（12 个）/确认（24 个）划分。确认集有 24 × 5 = **120** 个案例；五个对照方案在运行确认集之前固定。

| V0.4 独立确认（按文档平均相邻恢复率） | 相邻恢复率 | 完整恢复率 |
| --- | ---: | ---: |
| 传统 V0.2（含端点辅助项） | **78.54%** | **47.50%** |
| 传统 V0.2（去除端点辅助项） | 72.96% | 1.67% |
| 学习 V0.3（不做干扰增强） | 77.39% | 48.33% |
| 学习 V0.3（无上下对齐特征/搜索） | 76.28% | 40.00% |
| 学习 V0.3（完整配置） | 77.78% | 45.00% |

完整学习配置相对“无增强”提高 **0.39** 个百分点，相对“无上下对齐”提高 **1.50** 个百分点；不过它在这批新的文档族上仍略低于传统 V0.2。这个结果说明 V0.3 的干扰训练和对齐特征确有小幅贡献，但当前轻量特征模型并不稳定地胜过传统规则，项目保留该负结果。

![V0.4 独立确认消融对照](artifacts/v0.4/ablation-confirmation.png)

### 新外部页面测试

V0.4 新增三份从未进入训练的公开来源：USGS Open-File Report、NIST SP 800-63A-4 和 GPO Annual Report；每份渲染两页，共 6 页。它们都是完整公开页面的**模拟竖直裁切**，不是物理碎纸扫描。学习 V0.3 得到 82.76% 的文档平均相邻恢复率、1/6 完整恢复；传统 V0.2 为 77.22%、0/6 完整恢复。具体来源、许可、页码、SHA-256 与逐页结果均保存在 [`artifacts/v0.4/`](artifacts/v0.4/)。

- [USGS Open-File Report 2019-1112](https://pubs.usgs.gov/of/2019/1112/ofr20191112.pdf)（USGS 已发布报告 PDF 可公开使用，报告内说明可能的例外项）
- [NIST SP 800-63A-4](https://csrc.nist.gov/pubs/sp/800/63/a/4/final)（NIST 声明该出版物在美国不受版权保护）
- [GPO Annual Report 2023](https://www.govinfo.gov/app/details/GOVPUB-GP-d7a27c6b9d380a16190fb4651b2e74da)（GovInfo 说明其为 17 U.S.C. §105 下不受版权保护的美国政府作品）

![V0.4 新外部来源成功案例](artifacts/v0.4/success-v04-usgs-landstat-economic-valuation-10-external_page_simulated_cut-usgs-landstat-economic-valuation-9.png)

![V0.4 新外部来源失败案例](artifacts/v0.4/failure-v04-gpo-annual-report-2023-10-external_page_simulated_cut-gpo-annual-report-2023-4.png)

课堂用 5 页内容草案与中文讲稿见 [`artifacts/v0.4/presentation_outline.md`](artifacts/v0.4/presentation_outline.md)。

## V0.3 结果摘要

V0.3 使用新的按原始文档身份隔离的训练/验证/最终测试划分（各 24 个文档族）。最终测试含 24 × 5 = 120 个合成观察案例；模型选择只看验证集。

| 方法 | 文档平均相邻恢复率 | 汇总相邻恢复率 | 完整恢复率 | 平均单页耗时 |
| --- | ---: | ---: | ---: | ---: |
| 传统 V0.2（有限上下对齐） | 80.19% | 80.19% | 47.50% | 0.490 s |
| 学习 V0.2（干净训练、无对齐） | 73.17% | 73.33% | 38.33% | 0.158 s |
| **学习 V0.3（增强 + 对齐特征）** | **84.84%** | **84.72%** | **55.00%** | **0.493 s** |

V0.3 相对传统 V0.2 提高 **4.64 个百分点**，相对 V0.2 学习模型提高 **11.67 个百分点**。它并非在所有条件都更好：`clean` 下为 94.04%，低于 V0.2 学习的 95.94%；`edge_loss` 下为 66.99%，也低于 V0.2 学习的 70.35%；`vertical_shift` 下为 74.31%，仍低于传统 V0.2 的 85.84%。真实逐案例数据和失败案例均在 [`artifacts/v0.3/`](artifacts/v0.3/)。

最大改善来自 `no_assists`：V0.3 的相邻恢复率为 89.98%、完整恢复率为 62.5%，传统 V0.2 的完整恢复率为 0%。验证集在端点权重 0.0 与 0.8 的候选中选择了 **0.0**，减少对合成边框/纹理先验的依赖；详见 [`model_selection.json`](artifacts/v0.3/model_selection.json)。

### 外部页面测试

外部测试使用两份 GovInfo 公开美国政府报告，各渲染 5 个真实 PDF 页面，共 10 页。它们**从未进入训练或验证集**；程序对这些完整页进行模拟竖切和打乱，不能把结果称为物理碎纸扫描实验。

| 方法 | 文档平均相邻恢复率 | 完整恢复率 |
| --- | ---: | ---: |
| 传统 V0.2 | 77.99% | 0% |
| 学习 V0.2 | 83.98% | 0% |
| **学习 V0.3** | **93.54%** | **80%** |

来源、页码、SHA-256 与许可说明见 [`external_sources.json`](artifacts/v0.3/external/external_sources.json)。两份 PDF 均明确标注为美国政府作品、依据 17 U.S.C. §105 属公共领域：

- [Technical Evaluation for a U.S. CBDC System (OSTP, 2022)](https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193426/pdf/GOVPUB-PREX23-PURL-gpo193426.pdf)
- [Blueprint for an AI Bill of Rights (OSTP, 2022)](https://www.govinfo.gov/content/pkg/GOVPUB-PREX23-PURL-gpo193638/pdf/GOVPUB-PREX23-PURL-gpo193638.pdf)

![外部页面的 V0.3 成功案例](artifacts/v0.3/external-success-us-cbdc-technical-evaluation-2.png)

![外部页面的 V0.3 失败案例](artifacts/v0.3/external-failure-us-cbdc-technical-evaluation-3.png)

## 方法

1. **传统 V0.2**：接缝像素差、线性外推、梯度和边缘轮廓等人工特征；对每对纸条尝试有限、非循环的上下对齐；再用 Held–Karp 动态规划找全局排列。
2. **学习 V0.2**：以相同局部特征训练 scikit-learn Logistic Regression，但只见过干净合成页且不搜索上下偏移。
3. **学习 V0.3**：仍是轻量 scikit-learn Logistic Regression；训练文档族轮换覆盖五种观察条件，特征额外包含无标签的对齐收益、偏移大小和有效重叠比例，并以验证集选择端点权重。正确纸条顺序不进入推理过程。

V0.2 学习法落后主要因为训练/测试分布不一致、没有对齐能力，且端点项容易利用合成边框线索。V0.3 解决前两项并在验证集上删除端点先验；永久边缘缺失和弱纹理空白区仍然困难。

## 安装

需要 Python 3.10+。项目不下载预训练模型、不需要 GPU，也不调用付费 API。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行

### 命令行演示

```powershell
.\.venv\Scripts\python.exe -m shredded_reconstruction.demo `
  --variant mixed --strips 10 --condition vertical_shift `
  --method learned-v0.3 --seed 2026 --shuffle-seed 73
```

方法名与实现一一对应：`traditional-v0.1`、`traditional-v0.2`、`learned-v0.1`、`learned-v0.2`、`learned-v0.3`。结果写入 Git 忽略的 `outputs/demo/`。

### Streamlit 交互演示

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

页面提供离线文档图库（10 张文字、表格、图文混排及公开外部页面）和 PNG/JPG 上传。操作流程被拆成四步：**选择文档 → 切成纸条 → 随机打乱 → 开始重建**。规则匹配和学习匹配始终使用同一组当前打乱的输入；真实顺序只在预测完成后用于显示准确率和错误位置，绝不作为预测输入。

演示图库在 [`demo_documents/`](demo_documents/)，其中的外部页面来源和许可记录在 `manifest.json`，因此断网仍可使用。面向课程学生的原理说明见 [`docs/PROJECT_GUIDE_ZH.md`](docs/PROJECT_GUIDE_ZH.md)，1～2 分钟的现场演示步骤见 [`docs/CLASSROOM_DEMO_ZH.md`](docs/CLASSROOM_DEMO_ZH.md)。

### 重跑正式评测

```powershell
.\.venv\Scripts\python.exe scripts\run_v03_evaluation.py --output-dir artifacts\v0.3
```

网络可用时脚本下载两份上述公共领域 PDF 到 Git 忽略的 `.tmp/` 缓存；页面与实验工件保存到 `artifacts/v0.3/`，不会覆盖 V0.1/V0.2。

### 运行 V0.4 独立确认与外部验证

```powershell
.\.venv\Scripts\python.exe scripts\run_v04_ablation.py --output-dir artifacts\v0.4
```

该脚本会以新文档族重新训练固定的消融配置，运行 120 个确认案例，并下载三个公开 PDF 缓存到 Git 忽略的 `.tmp/`；正式 CSV、JSON、图表、成功/失败图与来源说明保存到 `artifacts/v0.4/`。

V0.4 的原始外部来源清单和追溯性更正说明分别保存在 `artifacts/v0.4/external/external_sources.json` 与 [`MANIFEST_CORRECTION.md`](artifacts/v0.4/external/MANIFEST_CORRECTION.md)；更正不改变历史实验数字。

### 重现最终独立确认与研究页案例

```powershell
.\.venv\Scripts\python.exe scripts\prepare_interactive_model.py
.\.venv\Scripts\python.exe scripts\run_final_evaluation.py
.\.venv\Scripts\python.exe scripts\prepare_final_analysis.py
.\.venv\Scripts\python.exe scripts\profile_streamlit_reruns.py
```

最后一条记录的是 Streamlit 服务端重跑耗时，不把它冒充为浏览器绘制耗时；真实浏览器布局截图在 `artifacts/final/screenshots/`。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

测试覆盖裁切/打乱、无干扰兼容性、扰动可复现性、非循环上下移动、预测排列合法性、文档级数据隔离、全局优化器与 Streamlit 示例页。

## 目录

```text
shredded_reconstruction/
  document.py       合成文档
  shred.py          切条、打乱和受控扰动
  features.py       V0.1/V0.2 与 V0.3 局部兼容度特征
  baseline.py       传统兼容度方法
  learned.py        scikit-learn 学习兼容度模型
  optimizer.py      Held–Karp 全局排列优化
  external.py       外部公开页面下载、渲染和来源记录
  ui_state.py       Streamlit 的切条/打乱/预测状态转换
  v03_splits.py     V0.3 文档族训练/验证/最终测试划分
scripts/run_v03_evaluation.py  固定评测、选参与外部页面测试
demo_documents/               离线演示页面与来源清单
docs/                         中文项目原理与课堂演示说明
artifacts/v0.1/               保留的历史记录
artifacts/v0.2/               保留的历史记录
artifacts/v0.3/               V0.3 CSV、JSON、页面、成功和失败案例
```

## 下一步

- 对真实物理碎纸扫描建立带人工标注的小规模数据集；不要把模拟裁切冒充为物理实验。
- 针对边缘缺失设计更鲁棒的局部表示，并研究首尾方向不确定时的全局建模。
- 扩展到旋转、重叠与不规则碎片之前，先保持可复现的外部页面基准。

## 最终独立确认（当前研究结论）

为避免把已用于开发的历史确认集继续当作最终证据，项目新增了一个按原始文档身份隔离的最终集：24 个新文档族、5 种固定观察条件、共 120 个案例。规则匹配的文档平均相邻恢复率为 **80.73%**，学习匹配为 **78.32%**；学习减规则的文档族自助法 95% 区间为 **[−6.63, +1.62] 个百分点**，因此不能声称任一方法总体稳定更优。

更有意义的结论是条件性的：学习匹配在无明显边框/纹理条件下更强，而规则匹配在上下错位和图文混排条件下更可靠。逐案例 JSON/CSV、组合统计和四类可复现案例在 [`artifacts/final/`](artifacts/final/)；中文研究总结见 [`docs/RESEARCH_SUMMARY_ZH.md`](docs/RESEARCH_SUMMARY_ZH.md)，最新五分钟讲稿见 [`docs/FINAL_PRESENTATION_ZH.md`](docs/FINAL_PRESENTATION_ZH.md)。

## License

MIT
