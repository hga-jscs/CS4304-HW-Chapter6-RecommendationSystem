# CS4304-HW-Chapter6-RecommendationSystem

CS4304《大数据算法与分析》Chapter 6 推荐系统大作业。本项目目标是完成 KKBox 音乐推荐基本任务与 Synerise 用户行为表征拓展任务，在可复现评估的基础上尽可能提高性能，并通过消融实验验证自己的改进。

**当前状态（2026-11-17）：KKBox 已完成真实数据抽样训练、完整测试集预测、格式检查与 Kaggle 提交评分，增强特征已取得本地验证增益。免费 Kaggle 全量训练准备中。拓展任务已生成百万用户的 208 维表征并通过官方格式验证，尚无三项下游任务成绩。**

运行说明分别见[基本任务](basic/README.md)与[拓展任务](extension/README.md)，实测结果见[实验记录](reports/README.md)。

## 已完成的基线

从 7,377,418 条训练记录中以种子 42 均匀抽取 300,000 条，固定分层划分为 240,000 条训练、60,000 条验证。17 个基础特征的 LightGBM 在第 66 轮取得最佳验证分数，再使用全部 300,000 条抽样记录重训，生成全部 2,556,790 条测试预测。

| 实验 | 本地验证 ROC-AUC | Kaggle Public | Kaggle Private | 平台状态 |
| --- | --- | --- | --- | --- |
| 300,000 行抽样基线 | 0.725795 | 0.65287 | 0.65598 | COMPLETE，提交编号 58453167 |
| 同样 300,000 行的增强特征 | 0.732045 | 待提交 | 待提交 | 已完成本地训练与验证 |

增强模式使用 32 个特征，在相同抽样、种子与划分下，比基础模式的本地 ROC-AUC 高 0.006251；最佳轮数为 67，平台增益尚未验证。这些均为抽样实验，全量模型尚未产生结果。平台分数与本地随机留出分数分别记录，不据此推算历史名次。精确参数、输出摘要与得分记录保存在 [basic_results.json](reports/basic_results.json)。

## 任务与验收标准

| 部分 | 任务 | 评测与交付 |
| --- | --- | --- |
| 基本任务 | 根据 KKBox 历史交互、用户和歌曲信息，预测用户首次可观察听歌后一个月内是否再次收听同一歌曲 | 生成测试集预测并提交 Kaggle，以平台返回的 ROC-AUC 为准，越高越好 |
| 拓展任务 | 使用 Synerise 行为数据构建通用用户表征，完成流失预测、产品倾向性预测、类别倾向性预测 | 同一套表征通过官方格式检查，再由官方固定下游模型分别训练、评估三项任务 |

要求依据课程《大作业内容与说明.pdf》的 Section 6。课程鼓励在已有方法上提出改进，不局限于复现。基本任务与拓展任务的代码、配置、实验记录和结果分别组织。

### 基本任务：KKBox 音乐推荐

- [竞赛与评测说明](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/overview)
- [数据与字段说明](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)
- [竞赛规则](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/rules)

官方提供六个压缩文件，解压后为：

| 文件 | 用途 |
| --- | --- |
| `train.csv` | 用户、歌曲、播放来源及二分类标签 `target` |
| `test.csv` | 待预测交互与提交用行编号 `id` |
| `members.csv` | 用户资料 |
| `songs.csv` | 歌曲属性 |
| `song_extra_info.csv` | 歌名、ISRC 等补充属性 |
| `sample_submission.csv` | 提交格式与测试行编号参考 |

提交文件使用 `id,target` 两列，`target` 为再次收听的预测概率。生成时按官方样例对齐，检查所有测试编号恰好出现一次、没有额外索引列、缺失值或无穷值，概率位于 `[0,1]`。

官方训练集和测试集按时间划分，但公开交互字段没有明确的事件时间戳。目前使用种子 42 的分层随机留出验证，不将任意行序视为时间顺序。类别与频次只在当前训练部分拟合，并单独记录新用户、新歌曲的效果；后续涉及标签统计或交叉验证时，同样隔离验证标签、按折计算。

课程要求在第 16 周之前提交结果。核对时 Kaggle 页面显示 **Late Submission** 入口；账号实际提交权限、次数限制及得分展示以提交页面为准。课程允许多次尝试不代表平台没有每日限制。记录平台原始分数及其类型，将本地验证成绩与平台成绩分别报告；对照历史榜单的分数不写成已获得的历史正式名次。

安装[基本任务依赖](basic/README.md#环境和数据)并在本机配置 Kaggle 凭据后，可在仓库根目录下载官方数据。训练、推理完成后，再运行校验与上传：

```powershell
python -m basic.download --data-dir data/basic
python -m basic.submission check --submission outputs/basic/baseline_sample/submission.csv --sample data/basic/sample_submission.csv --test data/basic/test.csv
python -m basic.submission upload --submission outputs/basic/baseline_sample/submission.csv --sample data/basic/sample_submission.csv --test data/basic/test.csv --message "CS4304 baseline sample"
python -m basic.submission list
```

上传前会再次检查文件；同一输出目录中的历史记录可防止重复提交相同文件。上传成功只表示文件已交给平台，需继续查询直至评分完成。已完成基线的精确训练参数见[实验复现记录](reports/README.md)。

### 拓展任务：Synerise 用户行为表征

- [ACM RecSys 2025 介绍](https://recsys.acm.org/recsys25/challenge/)
- [任务与评测规则](https://www.recsyschallenge.com/2025/)
- [数据说明与下载](https://recsys.synerise.com/data-set)
- [官方评测代码](https://github.com/Synerise/recsys2025)

使用购买、加入购物车、移出购物车、页面访问、搜索五类行为及商品属性构建同一套用户向量。课程指定的三项任务为：

| 官方任务名 | 预测目标 | 官方加权分数 |
| --- | --- | --- |
| `churn` | 对历史有购买的活跃用户，预测未来 14 天是否不再购买 | AUROC |
| `propensity_sku` | 预测未来 14 天是否购买指定的 100 个商品 | 0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity |
| `propensity_category` | 预测未来 14 天是否购买指定的 100 个类别 | 0.8 × AUROC + 0.1 × Novelty + 0.1 × Diversity |

评分以[官方指标实现](https://github.com/Synerise/recsys2025/blob/main/training_pipeline/metrics_containers.py)为准，保留各项指标和加权分数。固定官方下游模型与训练参数，主要比较用户表征的改进，不以自行更换评测器得到的分数代替官方结果。

优先使用官网提供的 RecSys 2025 **预处理数据集**。当前数据说明将其分为前五个月的 `input`、随后两周的 `train_target.parquet` 和再随后两周的 `validation_target.parquet`。表征使用 `input` 中的历史行为及官方提供的商品属性构建；目标窗口分别用于官方下游训练和验证。官方仓库仍有赛时目标隐藏、自行切分的旧说明，接入时按所下载版本核对目录，避免对预处理数据再次默认切分或把目标期行为混入表征。

官方表征文件要求：

| 文件 | 格式 |
| --- | --- |
| `client_ids.npy` | 一维 `int64` 数组，恰好覆盖 `relevant_clients.npy` 中的用户 |
| `embeddings.npy` | 二维 `float16` 矩阵，行数等于用户数，维度不超过 2048，无 NaN 或 Inf |

两个文件逐行对应。后续流程为：生成表征 → 官方 `validator.run` 格式检查 → 官方 `training_pipeline.train` 运行上述三项任务 → 保存分数与实验配置。固定并记录使用的官方代码版本，正式评估保留完整用户覆盖检查。

真实预处理数据已生成 1,000,000 名用户的 208 维 `float16` 表征，用时 109.418 秒，并通过固定版本原版官方验证器的完整用户检查。该结果只验证表征格式与覆盖，三项下游性能尚未测得，详见[验证记录](extension/baseline_validation.json)。

RecSys 2025 官方赛事已结束，本项目按课程要求进行本地官方脚本评估。历史总榜采用任务排名的 Borda 积分，不能把三项本地分数平均后当作历史总榜成绩。当前官方代码中的其他任务不纳入本项目首轮交付范围。

## 文件组织

两部分代码已分别放在 `basic/` 与 `extension/`。数据、模型与完整输出保存在本地，不进入 Git；下面的 `data/`、`outputs/` 是运行命令所用的示例目录。

```text
README.md
basic/          # 基本任务：数据处理、特征、训练、预测、提交检查
extension/      # 拓展任务：用户表征、格式检查、官方评测接入
reports/        # 已核验的实验结果；后续补充报告与汇报材料
tests/          # 数据划分、输出格式等关键行为的检查
data/           # 本地原始与处理数据，不提交 Git
outputs/        # 本地模型、预测、表征与完整日志，不提交 Git
```

两部分各自提供依赖说明、运行入口和复现步骤。数据与输出也按 `basic/`、`extension/` 分开存放。GitHub 保留可复现代码、必要配置、检查及精简实验结果；原始数据、大型模型、缓存和账号凭据通过忽略规则排除。

准备工作与临时实验在指定工作目录进行，验收后的代码同步到本仓库。运行路径使用参数或配置提供，不写死个人电脑路径。

## 性能优化计划

1. **打通评测。** 基本任务已完成真实抽样训练至平台评分，拓展表征已通过官方格式检查。接下来验证全量 KKBox 训练与提交，并完成拓展三项评估。
2. **基本任务优化。** 先比较简单基线与树模型，再检验用户、歌曲、播放场景的统计与交互特征；按验证结果尝试协同过滤表示及模型融合。标签统计严格隔离验证集。
3. **拓展任务优化。** 从行为计数、近期活跃度等聚合表征出发，逐步检验时间衰减、多行为融合和序列表征，在官方固定评测器下比较三项收益。
4. **消融与复现。** 对保留的改进做增量对比，记录性能、训练时间、内存/显存和推理开销；仅在验证有效时增加模型复杂度。
5. **正式交付。** 固定最终配置，从数据处理重新运行到输出，完成平台提交或官方评估，并整理报告和汇报材料。

计算资源采用**本机与免费 Kaggle 算力**。本机约 16 GB 内存、RTX 4060 Laptop 8 GB 显存，优先采用分块读取、紧凑数据类型和缓存复用。全量 LightGBM 基线准备使用私有 Kaggle CPU 作业，未启用 GPU；后续需要 GPU 的实验再使用免费配额。本计划不依赖付费算力。

每次正式实验记录代码状态、数据版本、划分方式、随机种子、参数、运行环境、耗时和指标。Kaggle 提交另记文件、时间、平台返回分数与对应模型，拓展任务保留官方评测输出。尚未完成的实验只记录状态，不预填成绩。

## 最终交付清单

- [x] 基本任务：可复现的数据处理、训练、预测与提交文件检查，已用真实抽样训练验证。
- [x] 基本任务：有效的 Kaggle 提交及平台得分记录。
- [x] 拓展任务：同一套用户表征与通过的官方格式检查。
- [ ] 拓展任务：三项任务的官方评估结果、配置和运行记录。
- [ ] 性能分析：基线对比、改进方法、消融实验及资源开销。
- [x] 仓库说明：分别写清两部分的环境、数据获取、运行命令和复现步骤。
- [ ] 英文报告与 5–10 分钟汇报材料。

课程《大作业报告要求.pptx》要求沿用综述报告的 LaTeX/Word 模板，报告使用英文，框架为 Abstract、Introduction、Literature Review (Related Works)、Problem Definition、Methodology、Experiment、Conclusion、Lessons Learned。第 16 周汇报，每组 5–10 分钟，一人汇报即可，顺序见课程共享文档。

## 后续工作

- 课程材料只给出“第 16 周”，具体日历截止时间按课程通知执行；目前无额外代码打包或提交平台要求。
- 修复 Kaggle 依赖安装阶段的 DNS 失败，启动全量 KKBox 基线；同时提交已验证的增强抽样模型，检查本地增益能否体现为平台增益。
- 完成拓展官方三项下游评估，再比较时间衰减等改进。
- 根据实测结果整理消融实验、英文报告与汇报材料。
