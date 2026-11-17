# 实验记录

本目录只保留精简、可核验的结果。原始数据、完整预测、模型和完整日志保存在实验输出目录，不提交到仓库。基本任务与拓展任务的结果分别记录。

## 基本任务：已评分的抽样基线

记录日期：2026-11-17。对应文件：[basic_results.json](basic_results.json)。

| 项目 | 结果 |
| --- | --- |
| 官方训练记录 | 7,377,418 |
| 训练样本 | 全文件均匀无放回抽样 300,000 条，种子 42 |
| 本地划分 | 分层随机留出；240,000 条训练，60,000 条验证 |
| 模型 | LightGBM，17 个基础特征 |
| 早停选择 | 最多 200 轮，第 66 轮取得最佳验证结果 |
| 最终重训 | 全部 300,000 条抽样记录，66 轮 |
| 本地验证 ROC-AUC | 0.7257947463534034 |
| 本地验证对数损失 | 0.6112660062248723 |
| 完整测试预测 | 2,556,790 条，输出格式检查通过 |
| Kaggle Public ROC-AUC | 0.65287 |
| Kaggle Private ROC-AUC | 0.65598 |
| Kaggle 提交编号与状态 | 58453167，COMPLETE |
| 平台记录的提交时间 | 2026-11-17 02:43:33.607 UTC（北京时间 10:43:33.607） |

平台提交说明为 `CS4304 baseline: uniform 300k seed42, LightGBM 66 rounds`。平台成绩来自该提交的评分结果，本地成绩来自训练阶段的 `metrics.json`，二者分别保存。训练时该文件中的 `platform_score` 尚为空，平台返回值已收录到本目录的独立提交记录中。

这次实验用于验证真实数据上的训练、重训、完整推理、文件检查和平台评分流程。它仅使用了 300,000 条训练记录，不代表全量模型表现。本地随机留出与官方按时间划分的测试分布不同，二者分数不直接等同；也不依据这些得分推算历史比赛名次。

按已完成实验的参数复现：

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/baseline_sample --max-train-rows 300000 --num-boost-round 200 --threads 4
python -m basic.predict --data-dir data/basic --model-dir outputs/basic/baseline_sample --output outputs/basic/baseline_sample/submission.csv --threads 4
python -m basic.submission check --submission outputs/basic/baseline_sample/submission.csv --sample data/basic/sample_submission.csv --test data/basic/test.csv
```

以上命令在已安装基本任务依赖、备齐官方数据的仓库根目录执行。使用新输出目录，避免覆盖已有实验；跨平台或依赖版本变化可能影响结果。环境和完整流程见[基本任务说明](../basic/README.md)。

## 基本任务：增强特征的本地对比

同样的 300,000 条抽样记录、种子 42 和 240,000/60,000 划分下，32 个增强特征取得本地 ROC-AUC **0.7320454603779785**、对数损失 **0.6063985437671019**，最佳轮数为 67。相对基础模式的本地 ROC-AUC 增加 **0.0062507140245751**。该模型尚未提交，不能将本地改进视为已验证的平台改进。

## 全量训练状态

免费私有 Kaggle 作业 `gaoanghe/kkbox-baseline` 的第一版在安装依赖时遇到 `pypi.org` DNS 失败，状态为 ERROR，尚未开始全量训练。正在调整作业以复用预装依赖；当前没有全量验证分数或平台分数，恢复后另行记录实际结果。

## 拓展任务：官方格式验证

真实预处理数据已生成 1,000,000 名用户的 208 维 `float16` 表征，用时 109.418 秒，并通过固定版本原版官方验证器的完整用户检查。格式验证不等于下游性能评估，三项任务尚无成绩。精确来源与摘要见[表征验证记录](../extension/baseline_validation.json)，运行方式见[拓展任务说明](../extension/README.md)。

## 记录口径

- 本地验证、Kaggle Public 与 Kaggle Private 分别列出，不混用。
- 每项结果保留样本量、随机种子、参数、依赖版本与数据摘要；提交文件同时保留 SHA-256。
- 本次抽样训练发生在实现尚未提交的工作树中，JSON 如实记录当时的基准提交及修改状态。该基准提交本身不包含完整训练实现。
- 尚在运行的任务不填入分数，合成数据测试不作为模型性能结果。
