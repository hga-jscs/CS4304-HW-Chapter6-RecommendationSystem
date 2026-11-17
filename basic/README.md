# 基本任务：KKBox 音乐推荐

预测用户在首次可观察听歌后一个月内是否再次收听同一歌曲，以 Kaggle ROC-AUC 为正式评价指标。本目录提供 LightGBM 基线、可对比的增强特征、分块推理和提交检查。

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

频次消融使用 `--feature-mode enhanced --disable-frequency-features`：只移除用户、歌曲、艺人、曲风、来源的五个频次特征，保留其余增强特征及类别编码。保持样本、种子和训练参数一致，并使用新的输出目录比较；开关会记录到 `configuration.json` 和特征状态中。`basic.kaggle_job` 支持同名开关并传给训练脚本。默认仍包含频次特征，基线模式不受该开关影响。

`--frequency-mode relative` 将计数除以拟合行数，默认 `count` 保留绝对计数。每次拟合内这只是共同的单调缩放；全量重训会重新学习树阈值，不能把未归一化频次直接认定为“复用旧阈值”的错误，也不预设归一化会涨分。

可选的 `--user-context-features` 增加三个用户与播放场景的平滑概率，场景为 `source_type`、`source_screen_name`、`source_system_tab`，计算式为 `(count(u,c) + 10 × P(c)) / (count(u) + 10)`。统计只用当前训练交互，不读取标签或测试统计；训练特征的分子计数和用户计数各减一，扣除当前行，全局 `P(c)` 保持训练范围的频率。验证与推理使用完整训练统计，新用户回退到全局频率，新场景概率为零，训练中出现的空值作为一致的独立场景处理。该开关默认关闭，可与频次消融组合，Kaggle 生成器支持同名参数。

计数方式默认 `--context-counting leave_one_out`，可用 `--context-counting full` 消融训练行的自身扣减。两种方式都不读取目标标签；差异涉及自身计数和长尾样本分布，需要实测比较。`--user-preference-features` 按同样的平滑及计数方式增加用户—艺人、用户—曲风组合概率；艺人和曲风以元数据原字段为类别，多曲风字符串视为一个组合。

`--collaborative-features` 增加用户—歌曲、用户—艺人图的重构内积和余弦，共四列。只用当前训练部分的无标签交互建立稀疏矩阵，依次应用 `log1p` 计数、按用户覆盖率计算的列 IDF 和行 L2 归一化。采用固定种子 42、3 次迭代的随机 SVD，歌曲最多 32 维、艺人最多 16 维；用户与物品因子均乘以奇异值平方根，内积对应截断重构值。未知用户/物品返回零，矩阵小于可用秩时记录实际秩，秩为零时四列为零。矩阵保持稀疏，特征查询分块进行，图不会在每个预测块重新拟合。全量重训会另建对应的图，两个阶段的形状、秩和拟合时间写入 `metrics.json`。这是一项待实测的图协同信号，不预设收益或运行时间。

上述用户场景特征借鉴[冠军作者原代码](https://github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution/blob/master/input/training/script/cnt_log_process.py#L59-L86)的按用户场景比例。本项目将统计范围限制为训练集，并新增强度为 10 的平滑与训练行留一处理；原代码合并了训练和测试交互，未使用这里的平滑与留一方式。

类别词表与频次只从本次训练部分拟合，不使用验证标签或测试集统计。未知类别编码为 `-1`，由 LightGBM 按缺失处理；未知频次为零。元数据以唯一 ID 查表，拒绝重复键，防止合并导致记录膨胀。日期无效、年龄不在 1–100 或歌曲时长非正数时按缺失处理。ISRC 信息可能有误，其增益需要实测确认。

默认 `--split-mode random` 保留固定种子的分层随机留出，便于与早期实验比较。推荐同时使用 `--split-mode row_order`，按原文件前后顺序留出末段；[第 8 名作者论文](https://web-ainf.aau.at/pub/jannach/files/Workshop_WSDMCUP_2018.pdf)第 2.3.2/3.2 节报告了按顺序切分的做法。代码仍将它标为行序验证，公开交互没有明确时间戳，行号差也不等于秒数或会话间隔。若换用含重复用户/歌曲对的自定义数据，需自行去重或采用分组划分；随机留出不会自动防止重复对泄漏。

程序先训练并早停，用验证结果确定 `best_iteration`，默认随后重新拟合特征状态，在全部所选训练行上训练最终模型；`--no-refit` 则保存早停模型及其原始训练部分的特征状态，两者始终成对。若指定抽样，上述“全部”指抽样后的训练数据。冷启动分组分数只作诊断；当组内没有两种标签时，其 AUC 记为 `null`。

`--sampling-mode uniform` 默认按种子从全文件无放回均匀选择行号；`--sampling-mode tail` 配合 `--max-train-rows N` 保留最后 N 行，并保存原始行号，用于末段快速实验。原始字符串使用 Arrow 紧凑存储，类别特征为 `int32`、数值特征为 `float32`。分块推理降低测试集的峰值内存；全量训练仍需同时容纳训练特征，具体资源占用以实测为准。

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
