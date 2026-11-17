# 拓展任务：Synerise 用户行为表征

本目录独立于 KKBox 基本任务，实现历史行为聚合表征及官方评测接入。208 维基线、独立列标准化、673 维历史信息增强表征均已完成课程指定的三项官方公开任务的本地评估。

2026-11-17 在完整官方预处理数据的 100 万用户上完成以下对照。原始基线记录保留在 [baseline_validation.json](baseline_validation.json) 和 [baseline_scores.json](baseline_scores.json)；新增实验的成绩、版本、计时方式和文件记录见 [review_results.json](review_results.json)。三种表征的用户行序一致，均通过固定版本的官方格式验证器。

| 表征 | 维数 | `churn` AUROC | `propensity_category` 官方分数 | `propensity_sku` 官方分数 |
| --- | ---: | ---: | ---: | ---: |
| 原始聚合基线 | 208 | 0.7285 | 0.7889 | 0.7799 |
| 原基线独立标准化 | 208 | 0.7269 | 0.7890 | 0.7821 |
| 历史信息增强并标准化 | 673 | 0.7286 | 0.7961 | 0.7881 |

独立标准化使 churn 下降 0.0016，SKU 分数提高 0.0022，说明标准化并非普遍有利。增强表征相对原基线的三项变化为 +0.0001、+0.0072、+0.0082：主要改善出现在两个 propensity 任务，churn 基本持平。每行使用同一份表征完成三项任务，没有拼接不同表征的单项最好成绩。增强实验同时改动多个特征块并做标准化，不能将涨分单独归因于某一块。

三组均使用 RTX 4060 Laptop GPU、Windows、PyTorch `2.5.0+cu118`、入口种子 42、相同官方模型和三轮训练，以及下文的等价目标缓存；分数定义见下文。原基线完整格式验证与三任务训练耗时 2783.939 秒，独立标准化实验约 2809 秒（按日志与结果文件时间估算），增强实验单调计时为 2747.804 秒。单次计时存在未控制的并发负载，不能据此认定维数增加或布局改动导致训练提速。成绩来自单个种子和公开验证划分，尚未检验统计显著性，也不代表隐藏任务或线上排行榜成绩。

原基线表征构建耗时 109.418 秒；独立标准化是在既有向量上追加 7.002 秒处理；增强表征完整构建耗时 319.357 秒。增强构建的这次实测使用 C-order 临时矩阵；随后改为 F-order 以适应按列累加，只核对了输出逐元素一致，尚未测量全量构建提速。

完整 SKU 验证集的 100 类中有 3 类没有正例；保留了固定版本 TorchMetrics 的官方宏平均 AUROC 处理，未剔除类别或改变指标。小批量 sanity validation 也可能出现无正例提示，类别支持数已记录在成绩文件中。

## 历史预训练与 1001 维实验流程

新增流程由三部分组成：`pretrain_encoder` 使用历史内伪未来学习 457 维表征，`history_representation build` 生成 544 维统计/内容/协同特征，`combine` 拼接并按列标准化为 1001 维。它是新候选，不能用上表 673 维的分数代表其性能。

预训练把完整 `input/` 最后 14 个日历日作为伪目标。当前数据对应 2022-10-27 至 2022-11-09；伪输入止于 10-26 日末。类别/SKU Top100、聚合词表只从伪输入选择，完整历史推理继续使用同一列定义。模型使用残差 MLP 与最近 32 条行为的 GRU，输出 256 维瓶颈和 201 维 logits。内部按用户分层留出，标准化及类别权重仅由训练用户拟合，用验证 BCE 选 checkpoint；这不是官方三项任务得分。官方 `target/` 不参与预训练或特征构建。

在包含 PyTorch 的拓展训练环境执行准备、训练及提取（阶段可分别重跑，无需重复准备缓存）：

```powershell
python -m extension.pretrain_encoder --stage prepare --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --pseudo-days 14 --sequence-length 32
python -m extension.pretrain_encoder --stage train --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --epochs 10 --patience 2 --batch-size 1024 --device cuda --seed 42
python -m extension.pretrain_encoder --stage extract --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --device cuda
```

在安装 `extension/requirements.txt` 的特征环境中构建其余特征，再拼接：

```powershell
python -m extension.history_representation build --data-directory data/extension --output-directory outputs/extension/history --batch-size 100000
python -m extension.history_representation combine --history-directory outputs/extension/history --pretrained-directory outputs/extension/pretrained --output-directory outputs/extension/combined
```

544 维包括：96 维统计（40 个事件统计、5 个是否有历史、6 个跨事件/间隔指标、三类商品事件各 15 个有序价格桶统计），256 维历史 Top128 商品/类别精确衰减偏好，64 维名称/搜索 PQ 共现池化，以及 128 维稀疏图用户因子。PQ 位置和码字被视为离散类别，名称和查询分别拟合，不能称为恢复了原始语义码本。默认图包含全部元数据类别及历史高频的 4096 个 SKU、2048 个 URL，使用历史衰减交互、`log1p`、行 L2 后做随机 SVD，并对输出按行归一化；无图边用户为零。

中间历史矩阵为磁盘上的 `float32`，只有拼接标准化后输出官方要求的 `float16`。裁剪与量化后不保证精确均值 0、方差 1。加 `--long-tail` 可保留额外两组 64 桶，最终是 1129 维：这 128 维没有包含在原清单的 1001 维加总内。完整图的 SVD 仍需额外内存，不建议与预训练数据准备同时运行。

最后将 `outputs/extension/combined` 传给下文 `extension.evaluate`，先 `--validate-only`，再使用相同官方版本、目标缓存和三任务协议取得真实评分。保留旧表征及旧成绩，不用伪任务验证损失替换正式结果。

## 数据与环境

使用[官方预处理数据](https://recsys.synerise.com/data-set)，下载地址为 [challenge_dataset.tar.gz](https://data.recsys.synerise.com/dataset/challenge_dataset.tar.gz)。解压后把包含以下内容的目录传给 `--data-directory`：

```text
product_properties.parquet
input/
    relevant_clients.npy
    product_buy.parquet
    add_to_cart.parquet
    remove_from_cart.parquet
    page_visit.parquet
    search_query.parquet
target/
    train_target.parquet
    validation_target.parquet
    其他官方目标清单文件
```

保留完整官方目录。已预处理的数据无需再次运行官方 `split_data`。表征只读取 `input/` 和根目录商品属性，两个目标窗口仅交给官方下游评测。

Python 3.11 下安装表征依赖；以下命令均在本项目仓库根目录执行：

```text
python -m pip install -r extension/requirements.txt
```

官方模型训练需要独立环境。`official-requirements.txt` 记录固定官方版本的直接依赖；GPU 环境先按 PyTorch 官方安装方式安装对应 CUDA 的 `torch==2.5.0`，再安装该文件。它不是已验证适用于所有操作系统的完整依赖锁文件。官方原始锁文件还包含 Linux/CUDA 平台包，不应直接混入 KKBox 环境。

## 生成表征

```text
python -m extension.build_embeddings --data-directory data/extension --output-directory outputs/extension/aggregates --batch-size 100000 --decay-days 30
```

- 208 维：五类事件的总量、1/7/30 天计数、7/30 天衰减量、最近活动和历史跨度共 40 维；三类商品事件的商品、类别、价格桶散列共 120 维；页面 URL 散列 16 维；搜索量化代码散列 32 维。
- 散列直方图使用 30 天半衰期。`--decay-days 0` 可用于无衰减消融，其余统计保持一致。
- 搜索量化代码按“坐标、类别编号”编码，不把类别编号大小当作连续语义距离；未使用商品名称量化表征。
- 所有特征做 `log1p`，避免重度用户计数在 `float16` 中溢出。没有历史事件的目标用户仍保留对应行。
- 流式读取 Parquet；100 万用户的主要 `float32` 累加矩阵约 832 MB，最终用户向量约 416 MB。累加矩阵使用磁盘 memmap，另有分批数据、用户索引与商品属性的内存开销。`--batch-size` 控制分批读取，输出和标准化最多按 20,000 行分块。

输出 `client_ids.npy`（一维 `int64`）和 `embeddings.npy`（二维 `float16`），行序与官方 `relevant_clients.npy` 一致；`features.json` 记录列定义，`manifest.json` 记录数据文件、时间边界、参数、版本和耗时。重复运行可加 `--overwrite` 覆盖本流程的既定产物，目录中的其他文件保留，遗留临时矩阵会在重试时重建。

### 独立标准化消融

直接使用已生成的同一份表示，只增加按列标准化及 `[-10, 10]` 截断，不必重新扫描全部历史：

```text
python -m extension.standardize --embeddings-directory outputs/extension/aggregates --output-directory outputs/extension/standardized
```

均值和总体标准差使用全部已有用户向量计算，不读取下游标签；常量列的除数设为 1。处理分块进行，源表示保持不变，参数保存在 `standardization.json`。这可检验固定三轮训练下的尺度影响，但官方网络内部已有 LayerNorm，不能预先保证标准化会涨分。独立标准化以原始 `float16` 文件为输入；构建器的 `--standardize` 则作用在最终量化前的 `float32` 特征上，记录时区分这两种入口。

### 历史信息增强模式

```text
python -m extension.build_embeddings --data-directory data/extension --output-directory outputs/extension/enhanced --mode enhanced --standardize
```

默认在实际数据上为 673 维，包含以下改动：

- 从历史购买和加购频次中选出热门 SKU/类目，两者各自排序；购买保留 Top-128、加购 Top-64、移除 Top-32 的独立衰减计数，其余条目仍进长尾哈希桶。具体编号保存在 manifest，选择过程不读 `target/`。单个字段不足指定数量时按实际数量输出。
- 价格字段改为桶编号的均值、最大值、标准差、30 天均值、观察数和五个有序区间的衰减计数。它们描述价格分位桶，不代表实际消费金额。
- 增加各事件的历史存在指示、购买/加购/搜索等平滑计数比、浏览数与历史跨度之比，以及至少两次购买时的 `span / (count - 1)` 平均间隔和相对最近购买时间；单次或无购买用户的间隔特征为 0。

默认增强矩阵临时文件约 2.69 GB（`float32`），输出约 1.35 GB（`float16`）；额外内存用于输入分块、热门项统计和属性映射。可用 `--top-k` 缩小精确偏好块。增加维度和消除部分碰撞不等于保证提升；商品名称 PQ 和图分解尚未加入这一轮改动。基础模式仍保留原来的 208 维计算，便于与历史结果比较。

## 官方验证与评测

固定官方源码到提交 [`e91c1a62f84611151a399ce157e6031a4a73618f`](https://github.com/Synerise/recsys2025/tree/e91c1a62f84611151a399ce157e6031a4a73618f)：

```text
git clone https://github.com/Synerise/recsys2025.git external/recsys2025
git -C external/recsys2025 checkout e91c1a62f84611151a399ce157e6031a4a73618f
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/aggregates --score-directory outputs/extension/validation --validate-only
```

格式验证仅需要 NumPy，可在表征环境运行。独立官方训练环境准备好后，用其 Python 运行：

```text
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/aggregates --score-directory outputs/extension/scores --accelerator gpu --device 0 --num-workers 0 --cache-targets
```

也可通过 `--python` 指定官方环境的 Python 路径。入口默认核对官方提交及受版本控制文件未被修改，先运行原始 `validator.run`，初始化随机种子后再执行原始 `training_pipeline.train` 的 `churn`、`propensity_category`、`propensity_sku`。入口种子默认为 42；官方网络内部的 `torch.manual_seed(1278)` 保持原样，模型和训练参数均未修改。不提供关闭用户覆盖检查或替换官方指标的开关。CPU 支持 `--accelerator cpu`，但完整官方模型训练通常更适合 GPU。

实验重跑可加 `--overwrite`，只清除旧的三份结果 JSON，重新写入日志，其他文件不删除。调试修改过的官方代码时可显式加 `--allow-modified-official`；结果会记录真实提交和受版本控制文件的改动，并将 `official_protocol_verified` 设为 `false`。正式性能对比仍使用固定、无改动的官方代码，不能把调试结果与正式结果混为一谈。

官方验证失败时进程返回非零退出码；只有三项训练成功且分数文件完整时才记录 `training_completed: true`。官方分数写入 `scores.json`，最佳 epoch 写入 `epochs.json`，执行命令与官方版本写入 `evaluation.json`，终端日志保留在 `validation.log` 和 `training.log`。

`churn` 的分数是 AUROC；两个 `propensity` 任务的分数是 `0.8 × AUROC + 0.1 × Diversity + 0.1 × Novelty`。官方在三个 epoch 中按此分数取最大值，再四舍五入到四位小数。`epochs.json` 的 0/1/2 分别对应第 1/2/3 轮；`scores.json` 中的 `placeholder` 不是任务成绩。

### 等价目标缓存

`--cache-targets` 是可选的数据加载优化，省略时使用原版逐用户目标计算。原版每取一个用户都扫描完整目标表；缓存版在下游 Dataset 初始化时建立相同的购买集合和商品/类别成员矩阵，按原顺序返回新的 `float32` 标签数组。适配层仅在运行时替换 `training_pipeline.data_module.BehavioralDataset` 引用，官方源码文件保持不变；类别顺序、网络、优化器、指标、batch size 和三个 epoch 均不变。缓存只服务下游训练和验证，目标数据不会进入用户表征。

验证命令：

```text
python -m extension.verify_target_cache --official-directory external/recsys2025 --data-directory data/extension --output outputs/extension/target_cache_validation.json
```

实测核对见 [target_cache_validation.json](target_cache_validation.json)：六个“任务 × 时间窗口”组合通过独立的全量正例集合及补集比较，覆盖 401,021,942 个标签单元；另对官方原函数抽样调用 17,193 个用户案例，并验证 18 个显式边界案例，均无差异。这里没有逐个调用全部用户的原函数。相同抽样集的目标查询加速约 328–466 倍，这个比例仅针对目标读取，不代表完整训练加速比例。

## 验证边界

`python -m pytest tests/test_extension.py` 检查目标数据隔离、不同分批大小结果一致、用户对齐、无历史用户、重复用户和非法搜索代码。合成数据检查只验证实现行为，不能代替完整官方三任务评估；完整数据的实测状态和结果应单独记录。
