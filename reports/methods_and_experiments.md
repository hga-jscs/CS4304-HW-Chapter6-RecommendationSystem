# Advanced recommendation experiments

This document supplies methodology, architecture figures, and experiment material for the English course report. The project's contribution is the implemented combination and controlled evaluation of established components.

## Evaluation targets

The requested targets are KKBox Public and Private ROC-AUC ≥ 0.7050, Synerise churn AUROC ≥ 0.7500, and both propensity scores ≥ 0.8150. They are acceptance targets, not predicted outcomes. The propensity score is `0.8 × AUROC + 0.1 × Diversity + 0.1 × Novelty`; it must not be reported as plain AUROC.

The preceding recommended models achieved 0.67905/0.68125 on KKBox and 0.7286/0.7961/0.7881 on the three Synerise tasks. Historical results remain in their original records. The new KKBox submission reaches both targets. Both new Synerise representations improve all three scores relative to the preceding representation, but neither reaches any of the three Synerise targets.

## KKBox: separating unlabeled structure from supervised statistics

The transductive branch fits categorical vocabularies and, when enabled, relative frequencies, context/preference probabilities, and sparse interaction factors on the complete train-plus-test interaction universe. The CSV reader excludes the label column from this branch. The same fitted state is used for model training, validation, inference, and optional supervised refitting. The selected 49-feature configuration disables frequency features.

```mermaid
flowchart LR
    X[Full train and test interaction columns] --> U[One unlabeled feature fit]
    U --> C[Categories, relative frequencies and context counts]
    U --> R[Global row-order features]
    U --> G[User-song, artist and genre SVD]
    Y[Supervised training partition and labels] --> T[OOF or preceding-block target means]
    C --> F[Feature matrix]
    R --> F
    G --> F
    T --> F
    F --> M[LightGBM and early stopping]
    Y -->|training labels| M
    V[Held-out labels] --> E[Validation only]
    M --> E
    M --> P[Complete test predictions]
    P --> K[Kaggle Public and Private ROC-AUC]
```

Six file-order proxies describe global position, previous and next user-event row gaps, forward and reverse user counts, and a song's relative frequency in the final 30% of rows. Row distance is not elapsed time. Next-event and reverse-count features use future **unlabeled** input within a batch prediction setting; this experiment does not establish suitability for causal online serving.

Five smoothed label features cover user, song, artist, user/source-system-tab, and user/source-type. Random validation uses five-fold out-of-fold values. Row-order training uses preceding blocks only, with a fixed 0.5 prior before any previous labels exist. External validation and test rows query statistics fitted on the corresponding supervised training partition.

The first ablation uses the final 500,000 training rows, a fixed 400,000/100,000 row-order split, and seed 42. It compares training-only statistics, global unlabeled statistics, row-order features, graph factors, target encoding, preferences, larger trees, and a separate relative-frequency variant. These share validation row identities. Their AUC values are not directly comparable with the earlier random holdout or with a different full-data holdout.

| Sample configuration | Validation ROC-AUC | Selected rounds |
| --- | ---: | ---: |
| Training-only reference | 0.671636410 | 166 |
| Global unlabeled statistics | 0.671505145 | 152 |
| Add row-order features | 0.688348418 | 142 |
| Add transductive graph factors | 0.708989698 | 84 |
| Add preceding-block target encoding | 0.709750066 | 85 |
| Add user preferences | 0.712585319 | 78 |
| Replace tree settings with the proposed larger configuration | 0.707270530 | 95 |
| Add global frequencies to the 63-leaf preference configuration | 0.712614053 | 87 |

All eight sets of 100,000 validation predictions were independently recomputed and matched the same official labels and row IDs (`7,277,418` through `7,377,417`, zero-based). The sequential graph addition improved AUC by approximately 0.02064128. The larger-tree configuration reduced it by approximately 0.00531479 relative to the 63-leaf preference configuration. Frequency features added only 0.00002873 AUC and worsened log loss in this one split; no stable benefit is established, so the simpler no-frequency configuration was selected. The full-data experiment separately compares retaining the holdout model with refitting on all labeled rows, without changing the unlabeled feature universe.

The sample job took 2,328.832 seconds in total. Its largest Linux process peak RSS was 7,579.07 MiB, explaining why an experiment with only 500,000 labeled rows still used Kaggle CPU: its unlabeled graph and vocabulary cover all train and test interactions. These are single-run measurements.

The transductive graph directly addresses a limitation of the previous training-only graph: training interactions were present during graph fitting while held-out interactions were absent. An absent edge between known nodes does not imply a zero reconstructed inner product, but the structural mismatch can still affect the feature distribution. A negative result from that earlier configuration is not a verdict on all collaborative methods.

### Full-data refit comparison

Both full-data jobs use 7,377,418 labeled rows, a 5,901,934/1,475,484 row-order split, 49 features, and seed 42. The model has 63 leaves, learning rate 0.08, a maximum of 1200 iterations, and early stopping patience 30. Both select 165 iterations and achieve validation ROC-AUC 0.7317389601.

| Final supervised training | Public ROC-AUC | Private ROC-AUC | Submission |
| --- | ---: | ---: | --- |
| Keep the holdout model | 0.69701 | 0.69844 | 58464162 |
| Refit on all labeled rows | **0.71053** | **0.71070** | 58464307 |

Both submissions are COMPLETE. The two jobs have identical validation row IDs, labels, elementwise predictions, validation-model SHA-256, execution sources, and dependency versions. Their only configuration difference is `no_refit`. Refitting therefore improves Public/Private scores by 0.01352/0.01226 in this controlled pair. Compared with the preceding recommended model, the new refit model improves the two scores by 0.03148/0.02945 and exceeds both 0.7050 targets. The result does not support the review's unconditional preference for skipping refit.

Training pipeline times are 807.138 seconds without refit and 1169.732 seconds with refit; corresponding inference/check times are 159.299 and 145.872 seconds. The refit stage itself takes 404.814 seconds. Peak Linux RSS is 8536.125 and 9806.129 MiB. These separate cloud runs are not a controlled timing benchmark. Precise measurements and provenance are retained in [the experiment record](basic_transductive_results.json).

## Synerise: historical pseudo-target pretraining

The pseudo-input ends at `2022-10-26 23:59:59.999999999 UTC`; the pseudo-target contains purchases from `2022-10-27 00:00:00` through `2022-11-09 23:59:59`. Both come exclusively from `input/`. Product and category prediction classes are selected from pseudo-input purchases, and aggregate columns remain fixed when the encoder later processes the complete input history.

```mermaid
flowchart TD
    A[Historical input before 27 October] --> B[Aggregate features and latest 32 events]
    B --> C[Residual MLP plus GRU]
    D[Purchases from 27 October through 9 November] --> L[Three-head weighted BCE]
    C --> L
    L --> W[Checkpoint selected on held-out users]
    W --> I[Selected encoder used for inference]
    H[Complete input history through 9 November] --> I
    I --> N[256 bottleneck values plus 201 logits]
    H --> S[96 statistics plus 256 exact popular-item counts]
    H --> Q[64 name/query PQ co-occurrence factors]
    H --> G[128 sparse user graph factors]
    S --> R[544 historical features]
    Q --> R
    G --> R
    N --> J[Concatenate: 1001 dimensions]
    R --> J
    J --> Z[Column standardization, clipping and float16]
    Z --> O[Unmodified official downstream training: three epochs]
    T[Official target windows] --> O
    O --> E[Churn, category and SKU scores]
```

The encoder trains on 900,000 users and selects its checkpoint on 100,000 held-out users. Feature normalization and loss weights use training users only. Churn loss applies only to users who purchased in the pseudo-input. Its positive weight is the training-eligible negative/positive ratio, clipped to `[0.02, 50]`; propensity positive weights are clipped to `[1, 50]`.

Pretraining stopped after four epochs and selected epoch 2 (zero-based index 1), with internal validation weighted BCE 0.13959244. This loss is not an official task score. Of the history-selected 100 SKU classes, 19 have no positive training users and 34 have no positive validation users in the pseudo-target. The vocabulary is retained rather than reselected using future purchases. These history-selected classes can differ from the official downstream class lists; their logits are representation features, not direct official predictions. The final 457-dimensional encoder output covers all one million users.

The 544-dimensional historical representation contains 96 event, price-bucket, and rhythm statistics; 256 exact popular-SKU/category decay counts; separate 32-dimensional name and query pools; and 128 row-normalized sparse graph factors. Average purchase interval is span divided by `count−1` for at least two purchases. Price values are ordinal bucket indices, not money. PQ factors describe position/code co-occurrences and do not recover an unavailable original PQ codebook.

The requested block sizes sum to 1001 dimensions. Adding the separately requested two 64-bin long-tail blocks produces 1129 dimensions; this is an explicit optional variant. Column clipping and float16 rounding do not preserve exact zero means or unit variances.

## Controlled downstream comparison

The formal Synerise comparison uses standardized 544-dimensional history features and standardized 1001-dimensional combined features. All one million client IDs match, all output entries are finite, and the shared 544 columns are exactly equal after float16 standardization. Thus the incremental comparison adds only the pretrained block. Each representation is evaluated on all three tasks using the same official commit, dependency versions, seeds, target cache, and three-epoch protocol.

This isolates the pretrained block within the new feature family. A comparison against the earlier 673-dimensional representation additionally changes the historical feature design; it cannot attribute all differences to pretraining, PQ pooling, or graph factors individually. Results are from one seed and do not establish statistical significance.

| Complete representation | Churn AUROC | Category weighted score | SKU weighted score | Best epoch indices |
| --- | ---: | ---: | ---: | --- |
| Previous enhanced 673 dimensions | 0.7286 | 0.7961 | 0.7881 | 2 / 1 / 2 |
| Historical 544 dimensions | 0.7293 | 0.8054 | 0.7972 | 1 / 1 / 2 |
| Combined 1001 dimensions | 0.7299 | 0.8053 | 0.7985 | 2 / 2 / 2 |
| Combined minus historical | +0.0006 | −0.0001 | +0.0013 | — |
| Requested minimum | 0.7500 | 0.8150 | 0.8150 | Neither new representation meets any target |

Compared with the previous 673-dimensional representation, the combined representation improves churn/category/SKU by 0.0013/0.0092/0.0104. However, pretraining introduces a trade-off relative to the new 544-dimensional control: a small category decrease accompanies the churn and SKU increases. Both complete representations are retained; taskwise maxima from different representations are not presented as one submission. The churn gap remains substantial. Exact scores, provenance, and per-task target checks are retained in [the advanced experiment record](../extension/advanced_results.json).

## Runtime evidence and limitations

Pseudo-window preparation took 627.634 seconds, encoder training 147.545 seconds, and full-user extraction 35.761 seconds, measured around the actual subprocesses. The historical 544-dimensional build took 514.621 seconds including startup (511.159 seconds inside the builder). Its first memory monitor observed the Windows Python launcher rather than its worker, so that value is invalid as a peak-memory estimate and has been marked accordingly. A separate worker snapshot is not substituted for a true peak measurement.

The historical 544-dimensional representation completed its continuous official evaluation in 3405.726 seconds. The combined-representation evaluation was interrupted after completing churn and before completing category. Its three exact churn AUROC values were preserved in the official offline Neptune log and verified against the run identity and log SHA-256. Applying the unmodified official aggregator recovers 0.7299 at zero-based epoch 2; no rounded progress-bar values are used. Category and SKU were restarted separately on the same representation under the same protocol and completed in 1251.872 and 1246.732 seconds. The official model resets the Torch seed for each task, and the data loader is sequential with zero workers. The original interrupted segment has no reliable monotonic duration, so the complete evaluation duration is unavailable. This is a recovered evaluation, not one uninterrupted run.

The code checks target isolation, prefix boundaries, frozen vocabularies, user alignment, graph zero rows, and prediction-row offsets. These tests are complemented by full-data platform and official-evaluator runs; synthetic test AUC values are not reported as model performance.

Method sources: [competition authors' row-order discussion](https://web-ainf.aau.at/pub/jannach/files/Workshop_WSDMCUP_2018.pdf), [winning solution's interaction SVD](https://github.com/lystdo/Codes-for-WSDM-CUP-Music-Rec-1st-place-solution/blob/master/input/training/script/svd_process.py), and [official Synerise data encoding and evaluation implementation](https://github.com/Synerise/recsys2025).
