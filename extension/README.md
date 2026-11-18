# 拓展任务：Synerise 用户行为表征

本目录独立于 KKBox 基本任务，实现历史行为聚合表征及官方评测接入。以下十组表征均已完成课程指定的三项官方公开任务的本地评估，包含标准化、分组缩放、历史预训练和购买节律/商品共现对照。

截至 2026-11-18，在完整官方预处理数据的 100 万用户上完成以下对照。原始基线记录保留在 [baseline_validation.json](baseline_validation.json) 和 [baseline_scores.json](baseline_scores.json)；标准化与 673 维实验见 [review_results.json](review_results.json)，旧 544/1001 维实验见 [advanced_results.json](advanced_results.json)，第三轮分组缩放与 1203/1797 维实验见 [behavior_results.json](behavior_results.json)，1874/2002 维实验见 [purchase_results.json](purchase_results.json)。各组用户行序一致，均通过固定版本的官方格式验证器。

| 表征 | 维数 | `churn` AUROC | `propensity_category` 官方分数 | `propensity_sku` 官方分数 |
| --- | ---: | ---: | ---: | ---: |
| 原始聚合基线 | 208 | 0.7285 | 0.7889 | 0.7799 |
| 原基线独立标准化 | 208 | 0.7269 | 0.7890 | 0.7821 |
| 历史信息增强并标准化 | 673 | 0.7286 | 0.7961 | 0.7881 |
| 统计、PQ 与图特征并标准化 | 544 | 0.7293 | 0.8054 | 0.7972 |
| 上述特征加历史预训练并标准化 | 1001 | 0.7299 | 0.8053 | 0.7985 |
| 同一历史内容改为分组缩放 | 544 | 0.7305 | 0.8058 | 0.7963 |
| 分离行为历史与分组缩放 | 1203 | 0.7302 | 0.8058 | 0.7927 |
| 分离行为历史加多窗口延迟预训练 | 1797 | 0.7340 | 0.8067 | **0.8057** |
| 再加活跃节律与历史复购分数 | 1874 | 0.7348 | 0.8068 | 0.8054 |
| 再加购买/加购商品共现向量 | 2002 | **0.7349** | **0.8069** | **0.8057** |

独立标准化使 churn 下降 0.0016，SKU 分数提高 0.0022，说明标准化并非普遍有利。增强表征相对原基线的三项变化为 +0.0001、+0.0072、+0.0082：主要改善出现在两个 propensity 任务，churn 基本持平。每行使用同一份表征完成三项任务，没有拼接不同表征的单项最好成绩。增强实验同时改动多个特征块并做标准化，不能将涨分单独归因于某一块。

544 维相对旧 673 维的三项变化为 +0.0007、+0.0093、+0.0091；统计、PQ 和图特征在这一组同时改动，不能单独归因。1001 维与 544 维的共同 544 列逐元素完全一致，新增预训练块后变化为 **+0.0006、-0.0001、+0.0013**：churn/SKU 略升，category 略降，两套保留为不同取舍。两组均未达到 `churn ≥ 0.7500`、`category ≥ 0.8150`、`SKU ≥ 0.8150` 中的任何一项目标。

第三轮分组缩放相对旧 544 维的变化为 +0.0012/+0.0004/−0.0009；再改为 1203 维历史后，变化为 −0.0003/0/−0.0036，新增特征未带来整体提升。拼接 594 维多窗口预训练后，相对同一 1203 维历史提高 **+0.0038/+0.0009/+0.0130**，相对旧 1001 维提高 **+0.0041/+0.0014/+0.0072**。后续 2002 维相对 1797 维的变化为 **+0.0009/+0.0002/0**，商品在四位官方分数下持平；仍分别距目标 0.0151/0.0081/0.0093，不混用不同表征的单项最好成绩。

各组均使用 RTX 4060 Laptop GPU、Windows、PyTorch `2.5.0+cu118`、入口种子 42、相同官方模型和每任务三轮训练，以及下文的等价目标缓存；分数定义见下文。原基线完整格式验证与三任务训练耗时 2783.939 秒，独立标准化实验约 2809 秒（按日志与结果文件时间估算），673 维实验单调计时为 2747.804 秒，544 维为 3405.726 秒。单次计时存在未控制的并发负载，不能据此认定维数增加或布局改动导致训练提速。成绩来自单个种子和公开验证划分，尚未检验统计显著性，也不代表隐藏任务或线上排行榜成绩。

1001 维评测在 churn 完成后被会话中断。churn 的三轮精确 Neptune 浮点记录经未修改的官方聚合器恢复为 0.7299、最佳第 3 轮；随后仅将尚未完成的 category/SKU 按原参数分别运行三轮。官方模型每任务重设种子 1278，顺序加载器使用 0 个工作进程，任务构造未发现依赖前一任务随机状态的采样。恢复两任务分别耗时 1251.872、1246.732 秒；初段单调计时已丢失，**1001 维总耗时记为 null**，不把中断等待算作训练时间。完整恢复证据见结果 JSON。

原基线表征构建耗时 109.418 秒；独立标准化是在既有向量上追加 7.002 秒处理；增强表征完整构建耗时 319.357 秒。增强构建的这次实测使用 C-order 临时矩阵；随后改为 F-order 以适应按列累加，只核对了输出逐元素一致，尚未测量全量构建提速。

完整 SKU 验证集的 100 类中有 3 类没有正例；保留了固定版本 TorchMetrics 的官方宏平均 AUROC 处理，未剔除类别或改变指标。小批量 sanity validation 也可能出现无正例提示，类别支持数已记录在成绩文件中。

## 多窗口与延迟预测流程

**1203 维历史与 1797 维延迟组合已完成全量构建、核验和三项正式评测；默认 393 维 immediate 路径仅完成实现测试，尚无该路径的全量正式分数。** 本轮将历史行为拆分为购买、加购与近 7 日 `cart−remove−buy` 余额代理，加入活跃度速度比、独立购买日节律、精确 URL 和带 IDF 的图特征；历史块为 1203 维。余额是事件计数代理，不是真实订单余额。意见核验与设计依据见[第三轮诊断](../reports/third_review.md)。

已完成的 544 维缩放对照直接复用下文构建的原始历史矩阵：

```powershell
python -m extension.standardize --embeddings-directory outputs/extension/history --output-directory outputs/extension/history_block --method block
```

先在完整 `input/` 内拟合独立历史块，保留同一份原始矩阵供历史单独对照和预训练拼接使用：

```powershell
python -m extension.behavior_features --data-directory data/extension --output-directory outputs/extension/behavior --batch-size 100000 --seed 42
python -m extension.standardize --embeddings-directory outputs/extension/behavior --output-directory outputs/extension/behavior_scaled --method block --batch-size 5000
```

准备阶段需要 `extension/requirements.txt` 中的特征依赖；训练和提取另需 PyTorch，内部验证还使用 scikit-learn。在已有官方训练环境补齐特征依赖时，可运行 `python -m pip install -r extension/requirements.txt -c extension/official-requirements.txt`，保留官方固定的 NumPy、pandas 等版本。下面三个阶段可分别执行；`prepare` 自动构建其内部的 earlier、later、full 特征与序列缓存，无需手动先建这些子目录：

```powershell
python -m extension.pretrain_encoder --stage prepare --mode multiwindow --data-directory data/extension --cache-directory outputs/extension/delayed_cache --output-directory outputs/extension/delayed_encoder --pseudo-days 14 --sequence-length 64 --prepare-batch-size 50000 --validation-fraction 0.1 --class-selection training_targets --include-delayed-window --seed 42
python -m extension.pretrain_encoder --stage train --data-directory data/extension --cache-directory outputs/extension/delayed_cache --output-directory outputs/extension/delayed_encoder --epochs 10 --patience 2 --batch-size 1024 --learning-rate 0.001 --churn-weight-mode unit --churn-multiplier 3 --device cuda --seed 42
python -m extension.pretrain_encoder --stage extract --data-directory data/extension --cache-directory outputs/extension/delayed_cache --output-directory outputs/extension/delayed_encoder --batch-size 1024 --device cuda
```

多窗口预训练有两套明确的拟合范围。其内部特征词表、PQ 基底、图 IDF 与投影基底只在最早输入 A 拟合，B 及完整历史推理均复用同一份 `feature_state.npz`；`training_targets` 仅用训练用户在两个历史伪目标窗内的购买选择辅助头类别，排除验证用户，且不读取官方 `target/`。省略该选项时采用 `prefix`，只按 A 的历史购买选类。共享用户划分覆盖所有配对，内部标准化只统计训练用户的 A/B 输入各一次，正例权重按预测头组分别拟合。上面的独立 `behavior` 历史块则在完整 `input/` 拟合；两套基底无需相同，最终按用户行序拼接，不把对应因子当成同一坐标。

默认多窗口模式用 A/B 各自紧接输入的 14 日目标训练 immediate 头；延迟选项再以 A 的输入、序列及买家掩码配 B 的目标，使用独立的 delayed 头。当前数据的 A 输入止于 10 月 12 日，A 目标为 10 月 13–26 日；B 输入止于 10 月 26 日，B 目标为 10 月 27 日–11 月 9 日。每组头均含 1 个 churn、100 个 category、100 个 SKU 输出。提取时使用完整历史，输出瓶颈与 sigmoid 分数；这些分数未经概率校准，不能将加权 BCE 输出直接解释为真实购买概率。

| 多窗口路径 | 预训练输出 | 与同一 1203 维历史块拼接 |
| --- | ---: | ---: |
| 默认 immediate | 128 + 64 + 201 = 393 | 1596 |
| `--include-delayed-window` | 128 + 64 + 201 + 201 = 594 | 1797 |

要运行尚未全量评测的默认 393 维对照，移除准备命令的 `--include-delayed-window`，并将三条命令中的缓存/模型目录分别换为 `outputs/extension/immediate_cache`、`outputs/extension/immediate_encoder`；其余参数保持一致。训练与提取会从缓存和 checkpoint 识别模式。准备阶段必须保留 `--mode multiwindow`，否则仍进入下文旧 457 维 legacy 路径。

将延迟预训练块与前面独立构建的历史块拼接，输出官方要求的 `float16` 1797 维矩阵：

```powershell
python -m extension.standardize --embeddings-directory outputs/extension/behavior --pretrained-directory outputs/extension/delayed_encoder --output-directory outputs/extension/delayed_combined --method block --batch-size 5000
```

393 维对照使用 `--pretrained-directory outputs/extension/immediate_encoder` 和单独的 `outputs/extension/immediate_combined` 输出目录。`--method block` 根据元数据将统计列按列标准化、PQ/图向量按行 L2 缩放、瓶颈按行归一化；已输出的 sigmoid 分数只乘块权重，不再次 sigmoid。参数保存在 `standardization.json`，原始矩阵保持不变。旧 `--method column` 仍是默认值；两条数值路径可能有极少量 `float16` 双重舍入差异，不声称共同统计列逐元素恒等，量化核验见第三轮诊断。

最后将 `delayed_combined`、`immediate_combined` 或 `behavior_scaled` 作为下文评测的 `--embeddings-directory`，各用独立评分目录完成格式验证和三项任务。内部验证 BCE/AUROC 只用于选择候选，不能代替正式分数或保证达标。

本次独立历史构建外层计时 831.067 秒，多窗口准备 2415.347 秒，训练 3301.282 秒、提取 89.522 秒；训练完成 4 轮并采用第 2 轮 checkpoint。1203/1797 维标准化分别为 89.383/239.105 秒，官方三项评测分别为 3453.002/3844.236 秒。单次计时不构成提速结论，嵌套阶段的计时不可重复累加。实际特征环境为 NumPy 2.4.6、pandas 2.3.3、PyArrow 23.0.1；训练和官方评测环境为 NumPy 2.1.0、pandas 2.2.3、PyArrow 19.0.1、PyTorch 2.5.0+cu118。精确源码提交与执行文件哈希见结果 JSON。

两套新表示的用户顺序、前 1011 个历史统计/计数列、列均值与 scale 均完全一致；名称/查询/图块分别有 998/612/8987 个值不同，全部相差一个 `float16` 表示步，无符号变化，最大绝对差为 0.0009765625。这与不同内存布局下归一化求和的舍入差异相符，但不将历史块称为逐元素完全相同。辅助类别覆盖也有取舍：SKU 与官方清单的交集由 24/100 增至 45/100，category 由 96/100 降至 79/100；此比较仅用于诊断，官方清单未参与选类。

## 购买节律与商品共现对照

1874 维与 2002 维均已完成全量构建和固定官方三任务评测。两步正式变化分别为 **+0.0008/+0.0001/−0.0003** 和 **+0.0001/+0.0001/+0.0003**；未达到三项目标，也没有单独评测不含活跃块的 1925 维商品组合。成绩与运行证据见 [purchase_results.json](purchase_results.json)。

从上节生成的 `delayed_combined`（1797 维）开始。第一次对照增加 75 项日级活动统计和两个复购分数。每项分数由 6 个历史截点上的梯度提升树拟合，两个预测区间分别为未来 1–14 天、15–28 天。模型训练使用截点前 112 天内有购买的用户。最早截点的可观察历史短于 112 天。

```powershell
python -m extension.activity_windows --data-directory data/extension --output-directory outputs/extension/activity
python -m extension.combine_profiles --input-directories outputs/extension/delayed_combined outputs/extension/activity --output-directory outputs/extension/activity_combined
```

第二次对照在上述 1874 维上追加购买和加购各 64 维，共 2002 维。两种行为共享历史商品因子，分别按 30 天半衰期池化。保留出现至少三次的商品，每个非零行为块的范数为 1（量化前）。两次拼接都直接复制原有 `float16` 列。

```powershell
python -m extension.product_cooccurrence --data-directory data/extension --output-directory outputs/extension/product_cooccurrence --cutoff-utc 2022-11-09T23:59:59Z
python -m extension.combine_profiles --input-directories outputs/extension/activity_combined outputs/extension/product_cooccurrence --output-directory outputs/extension/product_combined
```

将 `activity_combined` 和 `product_combined` 分别交给下文的 `extension.evaluate`，各使用独立的 `--score-directory`。商品词表与复购训练样本均取自 `input/`。[方法依据与实验范围](../reports/representation_methods.md)

两套官方三任务分别用时 3966.499/4067.043 秒，最佳 epoch 索引分别为 2/2/2 和 1/2/2。拼接与两套评测的连续串行链为 8041.562 秒；会话中断未中止该后台链，未重复训练已完成任务。小幅单种子差值不能据此视为稳定收益，也不能把内部分时段复购 AUC 当作官方流失分数。

## 紧凑表征、历史复购与商品转移

这一组实验复用上文的 `delayed_combined`、`delayed_cache`、`activity` 和 `product_cooccurrence`。所有特征构建命令只读历史输入及其派生缓存。最后一行将得到的同一份画像交给官方三项任务。

```powershell
python -m extension.compact_profiles --profile-directory outputs/extension/delayed_combined --activity-directory outputs/extension/activity --product-directory outputs/extension/product_cooccurrence --output-directory outputs/extension/compact
python -m pip install "lightgbm==4.7.0" "catboost==1.2.10"
python -m extension.repeat_purchases --data-directory data/extension --output-directory outputs/extension/repeat_purchases --compare-models
python -m extension.purchase_targets --data-directory data/extension --cache-directory outputs/extension/delayed_cache --output-directory outputs/extension/purchase_targets --threads 2
python -m extension.purchase_transitions --data-directory data/extension --output-directory outputs/extension/purchase_transitions --threads 2
python -m extension.combine_profiles --input-directories outputs/extension/compact outputs/extension/repeat_purchases outputs/extension/purchase_targets outputs/extension/purchase_transitions --weights 1 1 1.3 2 --output-directory outputs/extension/compact_purchases
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/compact_purchases --score-directory outputs/extension/compact_purchases_scores --cache-targets --num-workers 0 --seed 42
```

| 块 | 维数 | 实际构建方法 |
| --- | ---: | --- |
| 压缩对照 | 1047 | 只从 65,536 个无标签历史画像拟合各块的小型协方差矩阵。购买、加购、最近购物车差额单独投影，保留 512 维历史。原预训练块保留两个瓶颈、两个流失分数，按相同类别将近远期倾向分数取平均，共 394 维。另保留 77 维活动和 64 维商品协同。 |
| 复购 | 31 | 3 项购买基础、8 项购后活动、16 项商品/品类经验率和 4 个历史预测分数。先验只在结果已经观测完整的历史购买机会中估计，并排除收件用户所在折。两棵不同类型的树预测未来 1–7、1–14、1–28、15–28 日复购，取原始分数平均。省略 `--compare-models` 可仅用 scikit-learn 的梯度提升树生成同结构对照。 |
| 宽商品与品类预测 | 128 | 两个历史购买窗口、同一组 90 万训练用户。按购买次数至少 15 次筛得 1,804 个 SKU 和上限 500 个品类，分别训练正负平衡逻辑回归头，并对输出各保留 64 个概率主成分。复用冻结的快照特征，旧 GRU 权重保持固定。 |
| 有向转移 | 64 | 从浏览 URL 与加购 SKU 分别连到同一用户之后 7 天内的首个购买篮子，篮内商品均分一票。拟合时只使用可观察满 7 天的来源事件；提取时用近 14 天行为对行归一化的转移概率作一次传播。稀疏矩阵隐式完成传播和投影。 |

手工计数在压缩时乘 0.8，原监督块沿用输入中已有的 1.3。新商品头乘 1.3，行归一化的转移块乘 2。对不同分量使用独立的历史预测目标，避免重复同一分数来增加列数。PyArrow 只解码需要的 Parquet 列，PCA、概率输出和百万用户拼接分批执行。可用 `--embeddings-directory outputs/extension/compact` 单独评测压缩对照。

还可保持画像宽度和输入顺序，单独检验四个复购分数。两种树的历史输入从 27 项扩充为 102 项，多出的 75 项是按同一截点计算的五种事件活动摘要。在 2022 年 10 月 12 日留出的 13,297 位单购买日买家中，未来第 15–28 天复购 AUC 从 0.65348 升至 0.67262。该消融只衡量历史辅助模型。

```powershell
python -m extension.activity_repeat_scores build --data-directory data/extension --baseline-directory outputs/extension/repeat_purchases --output-directory outputs/extension/activity_scores
python -m extension.activity_repeat_scores replace --baseline-directory outputs/extension/repeat_purchases --scores-directory outputs/extension/activity_scores --output-directory outputs/extension/activity_purchases
python -m extension.combine_profiles --input-directories outputs/extension/compact outputs/extension/activity_purchases outputs/extension/purchase_targets outputs/extension/purchase_transitions --weights 1 1 1.3 2 --output-directory outputs/extension/compact_activity
```

冻结画像后，可以在独立分析环境用 DuckDB 对比购后仍活跃与停止活动两组的验证期复购率。查询直接读取 Parquet，按购买天数与距上次购买的天数分层，并用 Arrow 独立核对人数。这个入口读取验证标签，仅用于事后解释。

```powershell
python -m pip install "duckdb==2.0.0"
python -m extension.purchase_activity_audit --data-directory data/extension --output outputs/extension/purchase_activity.json
```

## 双窗购买预测、全体用户转移和复购增量

官方下游模型使用截点后第 1–14 天的购买标签训练，在第 15–28 天验证。`persistent_purchases` 复用购买版的历史商品词表和投影，在同一历史截点估计延迟购买，把近窗与延迟窗的对数几率按训练样本量加权。样本较少的商品更依赖原近窗预测，最终仍输出 128 维。

`--all-user-graph` 使用所有历史买家估计浏览到购买、加购到购买及购买到后续购买三类关系。前两类取 7 天内的首个购买篮，购买到购买取 1–28 天内的首个购买篮。每个来源事件均须有完整观察窗。图只为指定用户输出 64 维。

```powershell
python -m extension.persistent_purchases --data-directory data/extension --cache-directory outputs/extension/delayed_cache --baseline-directory outputs/extension/purchase_targets --output-directory outputs/extension/persistent_purchases --threads 2
python -m extension.category_horizon_residual --cache-directory outputs/extension/delayed_cache --near-directory outputs/extension/purchase_targets --joint-directory outputs/extension/persistent_purchases --output-directory outputs/extension/preserved_categories
python -m extension.purchase_transitions --data-directory data/extension --output-directory outputs/extension/all_user_transitions --all-user-graph --threads 2
python -m extension.combine_profiles --input-directories outputs/extension/compact outputs/extension/repeat_purchases outputs/extension/preserved_categories outputs/extension/all_user_transitions --weights 1 1 1.3 2 --output-directory outputs/extension/persistent_profiles
```

实际评测用的预测块有 144 列。前 64 列来自双窗 SKU 模型，中间 64 列保留原近窗品类坐标，后 16 列用历史用户拟合的主成分表示双窗与近窗品类预测之差。

复购对照保留购买版全部 1,270 列，再追加 28 列活动分数、线性残差、单日买家交互项和延迟复购的平滑阈值。缩放系数 2 已包含在新增块中。

```powershell
python -m extension.repeat_signals --data-directory data/extension --purchase-directory outputs/extension/repeat_purchases --activity-directory outputs/extension/activity_scores --output-directory outputs/extension/repeat_signals
python -m extension.combine_profiles --input-directories outputs/extension/compact_purchases outputs/extension/repeat_signals --output-directory outputs/extension/expanded_repeat
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/expanded_repeat --score-directory outputs/extension/expanded_repeat_scores --cache-targets --tasks churn --num-workers 0 --seed 42
```

## 历史预训练与 1001 维实验流程

新增流程由三部分组成：`pretrain_encoder` 使用历史内伪未来学习 457 维表征，`history_representation build` 生成 544 维统计/内容/协同特征，`combine` 拼接并按列标准化为 1001 维。544 与 1001 维的正式对照结果见上表。

预训练把完整 `input/` 最后 14 个日历日作为伪目标。当前数据对应 2022-10-27 至 2022-11-09；伪输入止于 10-26 日末。类别/SKU Top100、聚合词表只从伪输入选择，完整历史推理继续使用同一列定义。模型使用残差 MLP 与最近 32 条行为的 GRU，输出 256 维瓶颈和 201 维 logits；churn 损失仅用于伪输入曾购买的用户。内部按用户分层留出，标准化及类别权重仅由训练用户拟合，用验证 BCE 选 checkpoint；这不是官方三项任务得分。官方 `target/` 不参与预训练或特征构建。

在包含 PyTorch 的拓展训练环境执行准备、训练及提取（阶段可分别重跑，无需重复准备缓存）：

```powershell
python -m extension.pretrain_encoder --stage prepare --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --pseudo-days 14 --sequence-length 32
python -m extension.pretrain_encoder --stage train --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --epochs 10 --patience 2 --batch-size 1024 --device cuda --seed 42
python -m extension.pretrain_encoder --stage extract --data-directory data/extension --cache-directory outputs/extension/pretraining_cache --output-directory outputs/extension/pretrained --device cuda
```

在安装 `extension/requirements.txt` 的特征环境中构建其余特征，再拼接：

```powershell
python -m extension.history_representation build --data-directory data/extension --output-directory outputs/extension/history --batch-size 100000
python -m extension.standardize --embeddings-directory outputs/extension/history --output-directory outputs/extension/history_standardized
python -m extension.history_representation combine --history-directory outputs/extension/history --pretrained-directory outputs/extension/pretrained --output-directory outputs/extension/combined
```

544 维包括：96 维统计（40 个事件统计、5 个是否有历史、6 个跨事件/间隔指标、三类商品事件各 15 个有序价格桶统计），256 维历史 Top128 商品/类别精确衰减偏好，64 维名称/搜索 PQ 共现池化，以及 128 维稀疏图用户因子。PQ 位置和码字被视为离散类别，名称和查询分别拟合，不能称为恢复了原始语义码本。默认图包含全部元数据类别及历史高频的 4096 个 SKU、2048 个 URL，使用历史衰减交互、`log1p`、行 L2 后做随机 SVD，并对输出按行归一化；无图边用户为零。

中间历史矩阵为磁盘上的 `float32`，独立标准化或拼接标准化后输出官方要求的 `float16`；`history_standardized` 用于不含预训练块的 544 维对照。裁剪与量化后不保证精确均值 0、方差 1。加 `--long-tail` 可保留额外两组 64 桶，最终是 1129 维：这 128 维没有包含在原清单的 1001 维加总内。完整图的 SVD 仍需额外内存，不建议与预训练数据准备同时运行。

本轮历史特征构建内部耗时 511.159 秒，外层计时 514.621 秒；544 维独立标准化 11.109 秒，1001 维拼接标准化 56.585 秒。预训练准备、训练及提取分别为 627.634、147.545、35.761 秒；训练第 4 轮早停，采用第 2 轮 checkpoint。历史构建的最初内存监控误采到启动器，已将真实峰值设为 null，不提供无效的峰值内存结论。

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

当前下载版本的 input 截至 2022-11-09；本地下游训练目标为 11 月 10–23 日，验证目标为 11 月 24 日–12 月 7 日。官方 DataModule 对两者使用同一批用户和同一份输入表征，验证前不更新表征：对应输入截止后的第 1–14 天和第 15–28 天。延迟伪任务据此设计，尚未证明它能改善成绩。这些是当前本地固定官方评测的数据边界，不代表已核验线上隐藏目标日期。原始时间戳未带时区；代码按 UTC 解释是本项目的约定。

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

省略 `--tasks` 时运行全部三项；也可先单独运行指定任务，保留各次结果，例如：

```text
python -m extension.evaluate --official-directory external/recsys2025 --data-directory data/extension --embeddings-directory outputs/extension/delayed_combined --score-directory outputs/extension/delayed_churn_scores --tasks churn --accelerator gpu --device 0 --num-workers 0 --cache-targets
```

其余任务可指定 `--tasks propensity_category propensity_sku`，使用另一个评分目录。选择任务不改变其模型、指标或每任务三轮协议；跨次汇总应使用同一表征与相同参数，不能拼接不同表征的单项最好分。

实验重跑可加 `--overwrite`，只清除旧的三份结果 JSON，重新写入日志，其他文件不删除。调试修改过的官方代码时可显式加 `--allow-modified-official`；结果会记录真实提交和受版本控制文件的改动，并将 `official_protocol_verified` 设为 `false`。正式性能对比仍使用固定、无改动的官方代码，不能把调试结果与正式结果混为一谈。

官方验证失败时进程返回非零退出码；只有本次所选任务训练成功且对应分数/epoch 文件完整时才记录 `training_completed: true`。必须同时查看 `evaluation.json` 的 `tasks`：单项完成不代表三项均完成；`--validate-only` 的任务列表为空，`training_completed` 为 `false`。官方分数写入 `scores.json`，最佳 epoch 写入 `epochs.json`，执行命令与官方版本写入 `evaluation.json`，终端日志保留在 `validation.log` 和 `training.log`。

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

## 紧凑表征：三任务完整对照

同一公开验证集和固定种子，采用官方三轮评测器。每行三个任务均使用同一份用户表征。

| 表征 | 维数 | Churn | Category | SKU | 完成三项用时（分钟） |
|---|---:|---:|---:|---:|---:|
| 原方案 | 2002 | 0.7349 | 0.8069 | 0.8057 | 67.8 |
| 紧凑购买 | 1270 | 0.7354 | 0.8088 | 0.8071 | 60.1 |
| 紧凑活动 | 1270 | 0.7353 | 0.8084 | 0.8068 | 59.4 |

用时为三个成功任务的实测时间之和，不含中断重试、等待和表征构建。逐项用时、最佳轮次及相对原方案的差值见 [实验记录](compact_results.json)。
