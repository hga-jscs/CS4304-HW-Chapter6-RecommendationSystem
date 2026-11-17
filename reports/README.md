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

## 基本任务：增强特征对比

同样的 300,000 条抽样记录、种子 42 和 240,000/60,000 划分下，32 个增强特征取得本地 ROC-AUC **0.7320454603779785**、对数损失 **0.6063985437671019**，最佳轮数为 67。相对基础模式的本地 ROC-AUC 增加 **0.0062507140245751**。

完整预测经平台评分：提交 **58453417**，时间 **2026-11-17 02:53:04.663 UTC**，状态 COMPLETE，Public **0.64723**，Private **0.65049**。两项平台分数都低于基础模型，因此继续做频次消融及场景概率实验。

## 基本任务：频次消融与场景概率

以下实验继续使用同样的 300,000 条抽样记录、种子 42 和 240,000/60,000 划分；最大训练轮数均为 300、早停耐心为 30 轮，最终分别按最佳轮数在全部 300,000 条记录上重训。两份完整测试预测均通过格式检查并完成平台评分。

| 实验 | 特征数 | 最佳轮数 | 本地 ROC-AUC | Kaggle Public | Kaggle Private |
| --- | ---: | ---: | ---: | ---: | ---: |
| 元数据消融，移除五项频次 | 27 | 79 | 0.7256308918250057 | 0.65317 | 0.65600 |
| 元数据加用户场景概率 | 30 | 75 | 0.7273300103048881 | 0.65373 | 0.65697 |

- 元数据消融：提交 **58453740**，时间 **2026-11-17 03:04:57.573 UTC**，状态 COMPLETE；说明为 `CS4304 metadata ablation: 300k seed42, no frequency, 79 rounds`。
- 场景概率：提交 **58453892**，时间 **2026-11-17 03:10:45.373 UTC**，状态 COMPLETE；说明为 `CS4304 context: 300k seed42, metadata plus user context, 75 rounds`。

移除频次后，本地分数降低、平台分数恢复，支持将频次特征作为本次退化的重要排查对象。三个用户场景条件概率的方向参考了[冠军作者实现](https://github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution/blob/master/input/training/script/cnt_log_process.py#L59-L86)，本项目的统计范围仍限制在训练部分，不照搬其联合训练与测试的统计方式。

当前场景概率模型在四组样本实验中的平台得分最高，Public/Private 分别比原始基线增加 **0.00086/0.00099**。差值较小，未做显著性检验；不能据此宣称稳定显著改进，也不能外推全量训练收益。精确配置、运行耗时和提交文件摘要均见 [basic_results.json](basic_results.json)。

复现实验的训练命令：

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/metadata_sample --feature-mode enhanced --disable-frequency-features --max-train-rows 300000 --num-boost-round 300 --threads 4
python -m basic.train --data-dir data/basic --output-dir outputs/basic/context_sample --feature-mode enhanced --disable-frequency-features --user-context-features --max-train-rows 300000 --num-boost-round 300 --threads 4
```

分别将模型目录交给 `basic.predict` 生成完整预测，再通过 `basic.submission check` 检查。运行路径及其余默认参数与基础实验一致。

## 验证与测试的分布诊断

只用用户、歌曲编号和播放来源统计覆盖情况，不使用测试标签。随机验证的“已见”参照 240,000 条训练部分；测试的“已见”参照最终重训的 300,000 条样本，因此表中同时标明参照范围。

| 行级指标 | 随机验证：60,000 行，相对 240,000 条训练 | 测试：2,556,790 行，相对 300,000 条训练 |
| --- | ---: | ---: |
| 新用户 | 869（1.45%） | 236,947（9.27%） |
| 新歌曲 | 9,055（15.09%） | 664,355（25.98%） |
| 用户与歌曲均已见 | 50,177（83.63%） | 1,707,727（66.79%） |

完整 300,000 条训练样本中，`source_type=local-library` 占 30.771%，测试中占 22.7765%；`online-playlist` 则从 26.676% 变为 30.2931%。这些结果表明随机留出与官方测试的覆盖和场景分布不同，支持单独观察平台表现，但不证明某一特征是全部误差的原因。

频次消融的动机也来自第 6 名 EdZhang 的[作者论文](https://www.researchgate.net/publication/328838360)第 4 节 Conclusion：作者报告过歌曲听众数量计数提高验证表现、降低测试表现的现象，并讨论了相关分布差异。本项目通过自己的对照实验检验这一方向，不使用未知行序充当可靠时间，也不把测试标签用于特征。

## 基本任务：已评分的全量基线

免费私有 Kaggle CPU 作业 `gaoanghe/kkbox-baseline` 第二版已完成训练、推理与平台评分，未使用 GPU。实际数据使用全部 7,377,418 条训练记录，没有抽样。

| 项目 | 结果 |
| --- | --- |
| 留出划分 | 种子 42，分层随机；5,901,934 条训练，1,475,484 条验证 |
| 模型与轮数 | LightGBM，17 个基础特征，最多 400 轮，最佳轮数 400 |
| 留出验证 ROC-AUC | 0.795647783128468 |
| 留出验证对数损失 | 0.5482478278052512 |
| 最终重训 | 全部 7,377,418 条训练记录，400 轮 |
| 训练流程总耗时 | 1,394.3061074080001 秒，约 23.24 分钟 |
| 其中验证阶段训练、预测与指标计算 | 671.500581574 秒 |
| 其中全量模型重训 | 530.8831438039999 秒 |
| 完整推理与输出检查 | 2,556,790 条；352.8061022940001 秒，约 5.88 分钟 |
| Kaggle Public / Private | **0.67527 / 0.67706** |
| 提交编号与状态 | **58454264，COMPLETE** |
| 平台记录的提交时间 | 2026-11-17 03:24:51.997 UTC（北京时间 11:24:51.997） |

平台提交说明为 `CS4304 full baseline: 7377418 rows seed42, 17 features, 400 rounds`。分数及提交状态来自 Kaggle CLI `competitions submissions`；精确训练指标、预测摘要及数据 SHA-256 来自本次作业的输出文件，保存在 [basic_results.json](basic_results.json) 的 `full_data_experiments` 中。完整预测下载后也已通过本机文件检查；另独立核验 1,475,484 行验证预测的 `row_id` 全部唯一，标签与官方训练数据对应行逐一匹配，重算 ROC-AUC 和对数损失与原记录的差异均小于 4 × 10⁻¹⁶。该模型在当前五组已完成实验中的 Public 与 Private 均最高，相对抽样基线分别提高 0.02240、0.02108；数据量、训练轮数和运行环境均不同，因此这不是单一因素消融。最佳轮数达到上限，不代表已经验证继续增加轮数无益。

源码快照生成时记录的基准提交是 `63749fd82b82f265a1c59522554715a02bb17fc3`，工作树有未提交改动，这一事实原样保留。另对 `basic/__init__.py`、`basic/features.py`、`basic/train.py`、`basic/predict.py`、`basic/submission.py` 逐个校验，快照 SHA-256 与提交 `f74f2c8296338c424875c1efd16c206bfaef21d0` 中的文件全部一致；验证范围仅限这五个执行文件，不包括作业生成器。

作业实际使用 Kaggle 预装的 Python 3.13.18、LightGBM 4.6.0、NumPy 2.1.3、pandas 2.3.3、SciPy 1.16.3、scikit-learn 1.6.1、PyArrow 23.0.1。原清单中的计划依赖没有重新安装，复现时应以 `runtime_versions` 的实测版本为准。第一版在安装依赖时遇到 DNS 失败，第二版改用预装环境后完成；本地抽样实验使用的 Python 3.11.3 与 LightGBM 4.7.0 等版本另行保留。

核心训练和推理命令如下；精确复现还需匹配上述源码和运行环境：

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/baseline_full --max-train-rows 0 --num-boost-round 400 --seed 42 --threads 4
python -m basic.predict --data-dir data/basic --model-dir outputs/basic/baseline_full --output outputs/basic/baseline_full/submission.csv --threads 4
python -m basic.submission check --submission outputs/basic/baseline_full/submission.csv --sample data/basic/sample_submission.csv --test data/basic/test.csv
```

## 待完成的全量场景模型

免费私有 Kaggle CPU 作业 `gaoanghe/kkbox-context` 第一版正在运行，使用全部 7,377,418 条记录、种子 42、最多 400 轮，以及 `enhanced`、`--disable-frequency-features`、`--user-context-features` 的组合。提交时源码为干净的 `1a72fba10d47c91afaa4d35349ab5825aeaffec7`。尚无最终指标或平台成绩，不根据样本实验预填收益。

## 拓展任务：官方格式验证

真实预处理数据已生成 1,000,000 名用户的 208 维 `float16` 表征，用时 109.418 秒，并通过固定版本原版官方验证器的完整用户检查。目标缓存已通过一致性核对，正在运行三项下游评测，尚未报告全部最终分数。格式验证不等于下游性能评估；精确来源与摘要见[表征验证记录](../extension/baseline_validation.json)，运行方式见[拓展任务说明](../extension/README.md)。

## 记录口径

- 本地验证、Kaggle Public 与 Kaggle Private 分别列出，不混用。
- 每项结果保留样本量、随机种子、参数、依赖版本与数据摘要；提交文件同时保留 SHA-256。
- 抽样训练与全量基线快照均如实保留生成时的基准提交及工作树修改状态，不把基准提交冒充完整执行代码。全量基线另有五个执行文件与后续提交的逐文件哈希核验。
- 尚在运行的任务只记录可追溯的阶段性日志及其精度，最终指标与平台分数需等待对应输出；合成数据测试不作为模型性能结果。
