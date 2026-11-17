# 基本任务：KKBox 音乐推荐

预测用户在首次可观察听歌后一个月内是否再次收听同一歌曲，以 Kaggle ROC-AUC 为正式评价指标。本目录提供 LightGBM 基线、可对比的增强特征、分块推理和提交检查。

当前推荐采用[全量日期修正配置](../reports/README.md#全量偏好与行序方案)：增强元数据、关闭五项频次、启用留一法场景概率，400 轮上限；Public/Private 为 **0.67905/0.68125**。本轮偏好及行序加容量调整方案已完成平台评分，均未超过此配置。下面仍保留各实验开关和最小基线命令，便于复现对照。

## 环境和数据

建议使用 Python 3.11，在仓库根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r basic/requirements.txt
```

Linux/Kaggle 环境使用 `python -m pip install -r basic/requirements.txt`，或激活自己的隔离环境。以下命令中的 `python` 指已安装依赖的解释器。

依赖采用兼容版本范围，便于在本机和 Kaggle 安装；每次训练仍在 `configuration.json` 记录实际版本。历史实验中记录的精确版本保留原样，不用新的安装范围改写旧实验环境。

加入[官方比赛](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)并确认规则后，获取六个官方压缩包。解压到同一目录：

```text
data/basic/
  train.csv
  test.csv
  members.csv
  songs.csv
  song_extra_info.csv
  sample_submission.csv
```

目录路径可以通过命令行替换。数据、模型、账号凭据和大规模预测文件不提交 Git。

## 训练、推理与检查

先从整个训练集固定种子均匀抽取 300,000 行，跑通一次完整流程：

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/baseline_sample --max-train-rows 300000 --threads 8
python -m basic.predict --data-dir data/basic --model-dir outputs/basic/baseline_sample --output outputs/basic/baseline_sample/submission.csv --threads 8
```

`basic.predict` 始终预测完整测试集；抽样只影响训练。它使用临时文件输出，检查测试编号、列名、概率范围等要求，通过后才生成最终 `submission.csv`。单独的提交检查及平台上传命令见仓库的提交工具说明。

确认资源足够后，去掉 `--max-train-rows` 使用全部训练数据。可使用新的实验目录，也可加 `--overwrite` 替换该目录中本程序的训练产物；无关文件不会被删除。预测同样支持 `--overwrite`，校验成功后原子替换目标文件，遗留的 `.partial` 会在重试时清理。重训不自动改写已有预测，需再运行预测命令。

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/baseline_full --threads 8
python -m basic.train --data-dir data/basic --output-dir outputs/basic/enhanced_full --feature-mode enhanced --threads 8
```

比较模式时保持数据、抽样参数、验证比例和种子一致，再决定是否保留增强特征。可用参数还包括 `--num-boost-round`、`--early-stopping-rounds`、`--learning-rate`、`--num-leaves`、`--min-child-samples`、`--max-cat-threshold`、`--feature-fraction`、`--validation-fraction`、`--seed` 和 `--chunk-size`。快速试验可加 `--no-refit`，直接保存验证选出的模型和与其对应的特征状态。完整说明见 `python -m basic.train --help`。

## 使用 Kaggle CPU 运行全量实验

本机可用内存不足时，可将同一套代码生成私有 Kaggle 脚本作业。生成器只写配置和代码，不启动任务，也不把账号凭据嵌入脚本。以下命令需要已完成官方 Kaggle CLI 认证并加入比赛。

```powershell
python -m basic.kaggle_job --owner YOUR_KAGGLE_USERNAME --slug kkbox-baseline --output-directory outputs/basic/kaggle_baseline --feature-modes baseline
python -m kaggle kernels push -p outputs/basic/kaggle_baseline
python -m kaggle kernels status YOUR_KAGGLE_USERNAME/kkbox-baseline
python -m kaggle kernels output YOUR_KAGGLE_USERNAME/kkbox-baseline -p outputs/basic/kaggle_results
```

作业固定为私有、CPU，仅挂载 KKBox 官方比赛数据，默认关闭联网并复用预装环境；`enable_gpu=false` 不消耗 GPU 配额，仍受 Kaggle 账户的 CPU 作业与运行时长限制。默认全量训练，基线最多 400 轮、增强模式最多 600 轮，两者使用相同随机种子和留出规则。先用 `--feature-modes baseline` 跑通基线，之后可用 `--feature-modes enhanced` 单独启动增强实验，或用 `--feature-modes baseline enhanced` 在同一作业内顺序执行。轮数和抽样大小也可通过生成器参数调整。

脚本包含生成时的本地源代码快照及摘要。启动时检查预装 NumPy、Pandas、SciPy、scikit-learn、LightGBM、PyArrow，并在 `runtime_versions.json` 记录实际版本；缺少或无法导入依赖时明确报错，不隐式联网安装。只有显式添加 `--install-dependencies` 才启用联网并安装兼容范围内的运行依赖，不安装 pytest 或 Kaggle CLI。

脚本自动发现 `/kaggle/input` 下六个官方 CSV 或 `.csv.7z`。成功准备后在 `prepared_files.json` 记录字节大小及修改时间，相符的非空 CSV 可直接复用；旧版本没有完成记录的文件会重新提取一次。新解压先写入 `.partial`，完成后再改名，避免复用中断产物。压缩包优先使用系统 `7z`、`7zz` 或 `7za`，核对成员后只提取指定文件；系统工具不可用时才尝试 py7zr。临时数据放在 `/kaggle/temp`，代码在 `/kaggle/working/code` 重建，每个实验的模型、指标、日志和通过校验的 `submission.csv` 保存到 `/kaggle/working/baseline` 或 `/kaggle/working/enhanced`。作业本身不自动提交比赛，下载后按提交流程单独上传。

作业配置字段依据 [Kaggle 官方 metadata 文档](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md)。源代码修改后重新生成脚本；`--overwrite` 允许替换已有生成文件，并允许同一作业内的训练/推理重跑。生成器支持下面的数值、计数、偏好、采样、划分与不重训选项，并透传主要树模型参数。

## 特征和验证

| 模式 | 特征 |
| --- | --- |
| `baseline` | 用户、歌曲、播放来源类别；歌曲时长、曲风、艺人、词曲作者、语言；用户城市、年龄、性别、注册渠道及日期字段 |
| `enhanced` | 基线特征，加训练部分的用户/歌曲/艺人/曲风/来源频次、艺人和曲风数量、注册年月、会员时长、ISRC 国家及参考年代理、歌名长度 |

新训练默认 `--numeric-features calendar`：注册/到期日期转换为距 `2000-01-01` 的连续天数，保留相邻日期的区别，避免八位日期整数转 `float32` 时合并部分日期。ISRC 两位参考年使用 2017 年为世纪分界（00–17 对应 2000–2017，18–99 对应 1918–1999）。这是历史数据上的明确假设；[ISRC 参考年表示赋码年份](https://isrc.ifpi.org/isrc-standard/isrc-structure)，不等于歌曲发行年。`--numeric-features legacy` 保留旧数值定义供对照；旧模型文件缺少该配置时自动按旧定义推理。

历史 `context_full` 实验（Public AUC 0.67847、Private AUC 0.68125）使用旧数值定义，源代码为提交 `1a72fba10d47c91afaa4d35349ab5825aeaffec7`。严格复现应使用该提交及 [历史实验记录](../reports/basic_results.json) 中的参数和实际依赖版本；直接使用新默认 `calendar` 会形成新实验，`legacy` 开关用于新代码内的受控对照，不单独保证逐位复现旧结果。

频次消融使用 `--feature-mode enhanced --disable-frequency-features`：只移除用户、歌曲、艺人、曲风、来源的五个频次特征，保留其余增强特征及类别编码。保持样本、种子和训练参数一致，并使用新的输出目录比较；开关会记录到 `configuration.json` 和特征状态中。`basic.kaggle_job` 支持同名开关并传给训练脚本。默认仍包含频次特征，基线模式不受该开关影响。

`--frequency-mode relative` 将计数除以拟合行数，默认 `count` 保留绝对计数。每次拟合内这只是共同的单调缩放；全量重训会重新学习树阈值，不能把未归一化频次直接认定为“复用旧阈值”的错误，也不预设归一化会涨分。

可选的 `--user-context-features` 增加三个用户与播放场景的平滑概率，场景为 `source_type`、`source_screen_name`、`source_system_tab`，计算式为 `(count(u,c) + 10 × P(c)) / (count(u) + 10)`。统计只用当前训练交互，不读取标签或测试统计；训练特征的分子计数和用户计数各减一，扣除当前行，全局 `P(c)` 保持训练范围的频率。验证与推理使用完整训练统计，新用户回退到全局频率，新场景概率为零，训练中出现的空值作为一致的独立场景处理。该开关默认关闭，可与频次消融组合，Kaggle 生成器支持同名参数。

计数方式默认 `--context-counting leave_one_out`，可用 `--context-counting full` 消融训练行的自身扣减。两种方式都不读取目标标签；差异涉及自身计数和长尾样本分布，需要实测比较。`--user-preference-features` 按同样的平滑及计数方式增加用户—艺人、用户—曲风组合概率；艺人和曲风以元数据原字段为类别，多曲风字符串视为一个组合。

`--collaborative-features` 增加用户—歌曲、用户—艺人图的重构内积和余弦，共四列。只用当前训练部分的无标签交互建立稀疏矩阵，依次应用 `log1p` 计数、按用户覆盖率计算的列 IDF 和行 L2 归一化。采用固定种子 42、3 次迭代的随机 SVD，歌曲最多 32 维、艺人最多 16 维；用户与物品因子均乘以奇异值平方根，内积对应截断重构值。未知用户/物品返回零，矩阵小于可用秩时记录实际秩，秩为零时四列为零。矩阵保持稀疏，特征查询分块进行，图不会在每个预测块重新拟合。全量重训会另建对应的图，两个阶段的形状、秩和拟合时间写入 `metrics.json`。该开关默认关闭，收益和运行成本需要按具体数据规模验证。

[本轮 300,000 行消融](../reports/basic_review_results.json)已逐组复算验证预测：同一随机留出上，用户偏好将 AUC 从 0.727455 提高至 0.729332，叠加当前图特征降至 0.724193；绝对频次与相对频次的预测完全一致。图特征保持可选，默认关闭。行序和末段抽样使用不同验证记录，其分数不能直接作为特征优劣对照；这些样本结果也不代表平台得分。

全量平台复核进一步显示：日期修正模型为 Public 0.67905 / Private 0.68125，加入用户偏好后为 0.67569 / 0.67849，尽管随机验证 AUC 从 0.799568 升至 0.809235。当前偏好特征保持可选，不用随机验证的提升替代平台证据；完整参数、提交编号和核验记录保存在同一实验报告中。

上述用户场景特征借鉴[冠军作者原代码](https://github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution/blob/master/input/training/script/cnt_log_process.py#L59-L86)的按用户场景比例。本项目将统计范围限制为训练集，并新增强度为 10 的平滑与训练行留一处理；原代码合并了训练和测试交互，未使用这里的平滑与留一方式。

默认模式的类别词表与频次只从本次训练部分拟合，不使用验证标签或测试集统计；下面的转导模式显式改变无标签统计范围。未知类别编码为 `-1`，由 LightGBM 按缺失处理；未知频次为零。元数据以唯一 ID 查表，拒绝重复键，防止合并导致记录膨胀。日期无效、年龄不在 1–100 或歌曲时长非正数时按缺失处理。ISRC 信息可能有误，其增益需要实测确认。

默认 `--split-mode random` 保留固定种子的分层随机留出，便于与早期实验比较。推荐同时使用 `--split-mode row_order`，按原文件前后顺序留出末段；[第 8 名作者论文](https://web-ainf.aau.at/pub/jannach/files/Workshop_WSDMCUP_2018.pdf)第 2.3.2/3.2 节报告了按顺序切分的做法。代码仍将它标为行序验证，公开交互没有明确时间戳，行号差也不等于秒数或会话间隔。若换用含重复用户/歌曲对的自定义数据，需自行去重或采用分组划分；随机留出不会自动防止重复对泄漏。

程序先训练并早停，用验证结果确定 `best_iteration`，默认随后重新拟合特征状态，在全部所选训练行上训练最终模型；`--no-refit` 则保存早停模型及其原始训练部分的特征状态，两者始终成对。若指定抽样，上述“全部”指抽样后的训练数据。冷启动分组分数只作诊断；当组内没有两种标签时，其 AUC 记为 `null`。

`--sampling-mode uniform` 默认按种子从全文件无放回均匀选择行号；`--sampling-mode tail` 配合 `--max-train-rows N` 保留最后 N 行，并保存原始行号，用于末段快速实验。原始字符串使用 Arrow 紧凑存储，类别特征为 `int32`、数值特征为 `float32`。分块推理降低测试集的峰值内存；全量训练仍需同时容纳训练特征，具体资源占用以实测为准。

## 转导统计与目标编码

`--transductive` 显式使用完整 `train.csv + test.csv` 的无标签交互，读取时严格限定为五个 `INTERACTION_COLUMNS`，不读取其中的 `target` 或测试 `id`。抽样只限制有监督训练行数，无标签范围仍是完整约 993 万行。类别、相对频次、已启用的用户场景/艺人/曲风概率以及可选 SVD 在这一范围拟合一次；全量 refit 复用同一份状态。该模式统一使用相对频次和完整无标签计数，不只对训练行扣减自身；`configuration.json` 的 `effective_statistics` 记录实际生效的范围和模式。`--disable-frequency-features` 仍可单独关闭五个频次列。

转导 SVD 默认歌曲 48 维、艺人 24 维、曲风组合 16 维，输出三组内积和余弦共六列；`--song-components`、`--artist-components`、`--genre-components` 可调整，设为 0 可关闭对应因子。旧训练范围模式仍默认 32/16/0，旧四列模型可继续推理。完整无标签图让验证与测试交互也参与拓扑统计，但不保证涨分；仅训练图中缺少某条边，也不意味着其低秩重构内积一定为零。未知用户或物品与“已知节点之间缺边”是两种情况。

`--row-order-features` 要求 `--transductive`，增加六列：

| 特征 | 定义 |
| --- | --- |
| `global_time_pos` | 原始全局行号除以完整 train+test 行数 |
| `user_prev_row_gap` / `user_next_row_gap` | 同用户相邻记录的行号差取 `log1p`；首/末记录缺少邻居时为缺失值 |
| `user_cum_count` / `user_rev_cum_count` | 当前行之前/之后同用户的出现次数，均不计当前行 |
| `song_recent_pop_ratio` | 后 30% 行中歌曲频率除以全序列频率 |

训练使用原始 `row_id`；测试使用“官方完整训练行数 + 测试原始位置”，与抽样大小、CSV 的 `id` 值、推理块大小无关。行号差是文件顺序代理，不是秒数，也不能保证识别真实会话。未来无标签交互参与这些特征，因此应按转导式离线评测理解，不把它表述为只能看到过去数据的在线预测。

`--target-encoding` 独立增加用户、歌曲、艺人、用户×系统入口、用户×播放来源五个标签统计。每个类别的编码为 `(positive_count + 20 × prior) / (count + 20)`。随机划分使用固定种子 5 折 OOF，各折的计数和先验都只来自其他折；`row_order` 使用按原始行号排序的五个连续块，每块只用严格前序块，首块固定 0.5，不使用未来全局标签均值。验证/测试仅使用当前有监督训练部分的标签统计，未知类别回退其训练先验。refit 时重新生成 OOF 特征和全训练标签统计，`--no-refit` 则保留验证模型与其对应的训练分区状态。

这些选项均保持显式开启。一个待验证的完整候选命令如下；拆分消融时逐项添加开关，不能预设各项都会涨分：

```powershell
python -m basic.train --data-dir data/basic --output-dir outputs/basic/transductive_sample --feature-mode enhanced --max-train-rows 500000 --sampling-mode tail --split-mode row_order --no-refit --disable-frequency-features --user-context-features --transductive --row-order-features --collaborative-features --target-encoding
```

Kaggle 生成器透传同名选项以及树容量参数。Linux 训练记录进程峰值 RSS；500,000 行标签抽样仍需处理全量无标签图，全量训练建议在 Kaggle CPU 实测内存，不能把小数据测试通过解释为 16 GB 本机全量已验证。

## 实验产物

| 文件 | 内容 |
| --- | --- |
| `model.txt`、`features.pkl` | 最终 LightGBM 模型和匹配的特征状态，两者必须成对使用 |
| `validation_model.txt` | 在留出验证中选择的模型 |
| `metrics.json` | 本地 AUC、对数损失、常数基线、冷启动诊断、样本数和耗时 |
| `configuration.json` | 命令参数、模型参数、代码提交、工作树状态、依赖版本和运行平台 |
| `data_manifest.json` | 训练所用文件的字节大小及纳秒修改时间，不在每轮重新扫描大文件计算哈希 |
| `validation_predictions.csv` | 原始训练 CSV 行号、真实标签与验证概率 |
| `validation_history.json`、`feature_importance.csv` | 每轮验证 AUC、验证模型的特征增益 |
| `submission.csv`、`submission.csv.json` | 完整测试预测、校验结果和文件摘要 |

`features.pkl` 只加载本项目可信训练输出。`metrics.json` 中的 `platform_score` 在未实际提交前保留为空，不将本地分数当作 Kaggle 成绩。

数据内容的 SHA-256 保留在首次下载/准备数据生成的 `manifest.json`；大小和修改时间只是轻量实验指纹，不声称等价于内容校验。推理复用提交检查已计算的 CSV 摘要，仍记录对应模型摘要；提交前继续检查完整编号覆盖与有限概率，避免上传无效结果。

## 代码检查

```powershell
python -m pytest tests/test_basic_training.py -q
```

测试使用人工构造的小数据验证抽样、未知类别、训练/验证隔离和训练到推理流程；测试中的分数不代表真实比赛性能。

实现依据：[官方数据定义](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)、[LightGBM 训练接口](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)、[类别特征处理](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html#categorical-feature-support)。
