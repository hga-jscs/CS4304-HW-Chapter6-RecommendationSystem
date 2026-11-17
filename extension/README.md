# 拓展任务：Synerise 用户行为表征

本目录独立于 KKBox 基本任务，实现历史行为聚合表征及官方评测接入。当前方法是本项目的轻量基线，不是已经复现的官方基线成绩。

2026-11-17 已在完整官方预处理数据上生成 100 万用户的 208 维表征，耗时 109.4 秒，并通过固定版本的官方格式验证器。文件校验值与环境记录见 [baseline_validation.json](baseline_validation.json)。三项下游模型训练尚未完成，当前没有拓展任务性能分数。

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
- 流式读取 Parquet；100 万用户的主要 `float32` 累加矩阵约 832 MB，最终用户向量约 416 MB，另有分批数据、用户索引与商品属性的内存开销。`--batch-size` 控制分批内存。

输出 `client_ids.npy`（一维 `int64`）和 `embeddings.npy`（二维 `float16`），行序与官方 `relevant_clients.npy` 一致；`features.json` 记录列定义，`manifest.json` 记录数据文件、时间边界、参数、版本和耗时。输出目录已有同名结果时停止，避免覆盖实验。

## 官方验证与评测

固定官方源码到提交 [`e91c1a62f84611151a399ce157e6031a4a73618f`](https://github.com/Synerise/recsys2025/tree/e91c1a62f84611151a399ce157e6031a4a73618f)：

```text
git clone https://github.com/Synerise/recsys2025.git external/recsys2025
git -C external/recsys2025 checkout e91c1a62f84611151a399ce157e6031a4a73618f
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/aggregates --score-directory outputs/extension/validation --validate-only
```

格式验证仅需要 NumPy，可在表征环境运行。独立官方训练环境准备好后，用其 Python 运行：

```text
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/aggregates --score-directory outputs/extension/scores --accelerator gpu --device 0 --num-workers 0
```

也可通过 `--python` 指定官方环境的 Python 路径。入口核对官方提交及受版本控制文件未被修改，先运行原始 `validator.run`，初始化随机种子后再执行原始 `training_pipeline.train` 的 `churn`、`propensity_category`、`propensity_sku`。种子默认为 42，可用 `--seed` 记录重复实验；未修改官方模型和训练参数。不提供关闭用户覆盖检查或替换官方指标的开关。CPU 支持 `--accelerator cpu`，但完整官方模型训练通常更适合 GPU。

官方验证失败时进程返回非零退出码；只有三项训练成功且分数文件完整时才记录 `training_completed: true`。官方分数写入 `scores.json`，最佳 epoch 写入 `epochs.json`，执行命令与官方版本写入 `evaluation.json`，终端日志保留在 `validation.log` 和 `training.log`。原始官方分数中的 `placeholder` 不是任务成绩。

## 验证边界

`python -m pytest tests/test_extension.py` 检查目标数据隔离、不同分批大小结果一致、用户对齐、无历史用户、重复用户和非法搜索代码。合成数据检查只验证实现行为，不能代替完整官方三任务评估；完整数据的实测状态和结果应单独记录。
