# MedDINOv3-Stroke
参考 [DINOv3-stroke](https://github.com/Zzz0251/DINOv3-stroke) 的范式，当前版本采用：
- 冻结 MedDINOv3 (ViT-B/16, CT-3M) 作为 backbone
- 使用轻量多标签分类头完成 ICH 任务训练与评估

## 模型架构
当前代码对应两阶段流水线。

### 阶段 A：冻结 MedDINOv3 提特征
输入：单个 NCCT volume（`.nii` / `.nii.gz`）

处理流程：
1. 读取 volume 并按轴切成 2D slices
2. HU 裁剪：`[-1000, 1000]`
3. 归一化：使用固定 `mean/std`
4. 单通道复制为 3 通道以适配 ViT 输入
5. `vit_base(...).forward_features(...)` 提取：
   - `x_norm_clstoken`：每张切片一个 `D=768` 向量
   - `x_norm_patchtokens`：用于可视化 patch 相似度热图

输出：
- `meddinov3_slice_embeddings.npy` (`[N, 768]`)
- `slice_indices.npy`
- `embedding_mean.npy`
- `similarity_map.npy/.png`（示例切片）

### 阶段 B：轻量分类头训练与评估
训练输入：预提取 embedding 与标签（`npy`）

分类头结构：
- `LayerNorm(768)`
- `Linear(768 -> hidden_dim)`
- `GELU + Dropout`
- `Linear(hidden_dim -> C)`

训练目标：
- `BCEWithLogitsLoss(pos_weight=...)`（多标签）
- 默认优化器：`AdamW`
- 评估指标：`macro AUROC` + `per-class AUROC`

说明：
- 当 `--hidden-dim <= 0` 时，退化为单层线性头 `Linear(768 -> C)`

## 代码结构
核心逻辑在 `src/meddinov3_stroke/`，`script/` 仅做入口。

- `src/meddinov3_stroke/config.py`：项目配置中心（默认超参数 + env 路径解析）
- `src/meddinov3_stroke/feature_extractor.py`：MedDINOv3 backbone 抽特征、聚合、热图
- `src/meddinov3_stroke/meddinov3_infer.py`：MedDINOv3 推理主逻辑
- `src/meddinov3_stroke/head_train.py`：轻量头训练
- `src/meddinov3_stroke/head_eval.py`：轻量头测试/评估
- `src/meddinov3_stroke/rsna_pipeline.py`：RSNA CSV 读取、study 级划分、embedding 缓存
- `src/meddinov3_stroke/rsna_split.py`：RSNA 数据划分入口
- `src/meddinov3_stroke/rsna_train.py`：RSNA 一体化训练 pipeline
- `src/meddinov3_stroke/ich_infer.py`：最终推理（出血概率 + 热图）
- `src/meddinov3_stroke/head_model.py`：head 定义、AUROC 与数据校验
- `src/meddinov3_stroke/infer_utils.py`：设备选择、env 与性能统计
- `script/run_meddinov3_infer.py`：推理入口
- `script/train_meddinov3_head.py`：训练入口
- `script/test_meddinov3_head.py`：测试入口
- `script/prepare_rsna_splits.py`：按 study 划分 train/val/test
- `script/train_meddinov3_rsna.py`：从 RSNA CSV + NIfTI 直接训练
- `script/infer_meddinov3_ich.py`：最终部署推理入口

## 环境准备
```bash
uv lock --default-index "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
uv sync
source .venv/bin/activate
```

初始化配置文件：
```bash
cp .env.template .env
```

配置边界约定：
- `.env`：路径、权重路径、下载地址、下载相关参数
- `src/meddinov3_stroke/config.py`：训练/推理默认超参数（可被 CLI 覆盖）

## 模型权重下载
使用内置脚本下载 MedDINOv3 权重到 `modelsweights/`：

```bash
python script/download_models.py --package meddinov3
```

仅检查远程大小（不下载）：
```bash
python script/download_models.py --package meddinov3 --check-size
```

`.env` 中核心键：
- `DATASETS_DIR`
- `MODELS_DIR`
- `MEDDINOV3_CKPT_PATH`
- `MEDDINOV3_URL` / `MEDDINOV3_GDRIVE_FILE_ID` / `MEDDINOV3_HF_REPO`

## 数据说明
原始数据可使用 [RSNA ICH](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data)。

建议流程：
1. DICOM 按 `StudyInstanceUID` 归组
2. 用 `dcm2niix` 转 `nii.gz`
3. 统一方向、spacing（建议重采样）

如果你已有 series 级 CSV（例如 `rsna_hemorrhage_series_labels.csv`），可直接用一体化 pipeline：
- 输入：`nifti_path` + 6 维 series 标签
- pipeline 内部自动：
  - 按 `study_uid` 做 train/val/test 划分
  - 抽取并缓存 volume embedding
  - 训练轻量头
  - 在 test 集评估

## 使用说明
下面是推荐的完整流程。

### 0) 现在可以直接跑的命令
```bash
cd ~/MedDINOv3-Stroke
source .venv/bin/activate
export MPLCONFIGDIR=/tmp/mpl_$USER
mkdir -p "$MPLCONFIGDIR"
```

先训练（CSV + NIfTI 直连）：
```bash
python script/train_meddinov3_rsna.py \
  --series-csv /data/datasets/rsna_hemorrhage_series_labels.csv \
  --output-dir outputs/rsna_pipeline \
  --cache-dir outputs/rsna_cache \
  --checkpoints-dir checkpoints/rsna_ich_head \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --pool-mode mean \
  --batch 8 --resize 224 \
  --epochs 30 --train-batch-size 64 \
  --skip-invalid-nifti \
  --device gpu
```

再做最终病例推理（出血概率 + 热图）：
```bash
python script/infer_meddinov3_ich.py \
  --input /data/datasets/rsna_hemorrhage_nii_1mm/ID_0a0ba0944f/ID_5d5fb15098/_1530143486.nii.gz \
  --checkpoint checkpoints/rsna_ich_head/best.pt \
  --output-dir outputs/ich_case_5d5fb15098 \
  --device gpu \
  --batch 8 --resize 224 --top-k 5
```

这些命令分别会做什么：
- `train_meddinov3_rsna.py`：按 `study_uid` 划分 train/val/test，抽取并缓存 volume embedding，训练 head，并在 test 上评估。
- `infer_meddinov3_ich.py`：对单个 `nii.gz` 输出 6 类出血概率、阈值判定、Top-K 可疑切片和热图。
- 训练权重默认保存到 `checkpoints/rsna_ich_head/best.pt` 与 `checkpoints/rsna_ich_head/last.pt`。

### 1) 推理：单例 CT + 热图
默认读取 `input/` 下第一个 `.nii/.nii.gz`：

```bash
python script/run_meddinov3_infer.py
```

指定输入、输出与设备：
```bash
python script/run_meddinov3_infer.py \
  --input /path/to/case.nii.gz \
  --output-dir outputs/meddinov3_case001 \
  --max-slices 128 \
  --batch 8 \
  --resize 224 \
  --device cpu
```

常用参数：
- `--stride`：切片步长
- `--max-slices`：最多处理切片数，`0` 表示不限制
- `--sim-slice`：生成热图的 slice 索引，`-1` 表示中间切片
- `--sim-patch`：参考 patch，`center` 或 `h,w`

### 2) 训练：轻量分类头（embedding 输入）
```bash
python script/train_meddinov3_head.py \
  --train-embeddings outputs/train/meddinov3_slice_embeddings.npy \
  --train-labels outputs/train/slice_labels.npy \
  --val-embeddings outputs/val/meddinov3_slice_embeddings.npy \
  --val-labels outputs/val/slice_labels.npy \
  --epochs 20 \
  --batch-size 64 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --hidden-dim 256 \
  --dropout 0.2 \
  --output-dir checkpoints/head_train
```

输出：
- `checkpoints/head_train/best.pt`
- `checkpoints/head_train/last.pt`
- `checkpoints/head_train/train_summary.json`

### 3) 测试：轻量分类头
```bash
python script/test_meddinov3_head.py \
  --embeddings outputs/test/meddinov3_slice_embeddings.npy \
  --labels outputs/test/slice_labels.npy \
  --checkpoint checkpoints/head_train/best.pt \
  --output-json outputs/head_eval/metrics.json \
  --output-probs outputs/head_eval/probs.npy
```

输出：
- `outputs/head_eval/metrics.json`：loss、macro AUROC、per-class AUROC
- `outputs/head_eval/probs.npy`：每样本每类别概率

### 4) RSNA 一体化训练（推荐）
给 series CSV，embedding 抽取过程“无感”，并自动做缓存。

```bash
python script/train_meddinov3_rsna.py \
  --series-csv /data/datasets/rsna_hemorrhage_series_labels.csv \
  --output-dir outputs/rsna_pipeline \
  --cache-dir outputs/rsna_cache \
  --checkpoints-dir checkpoints/rsna_ich_head \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --pool-mode mean \
  --batch 8 --resize 224 \
  --epochs 30 --train-batch-size 64 \
  --skip-invalid-nifti
```

输出：
- `outputs/rsna_pipeline/splits/train.csv|val.csv|test.csv`
- `outputs/rsna_pipeline/features/*_embeddings.npy`, `*_labels.npy`
- `checkpoints/rsna_ich_head/best.pt`, `last.pt`
- `outputs/rsna_pipeline/test_metrics.json`
- `outputs/rsna_pipeline/pipeline_summary.json`

如果出现 `Cannot work out file type of "/data/datasets"` 之类错误：
- 说明 CSV 的某些 `nifti_path` 不是 `.nii/.nii.gz` 文件（可能是目录或坏路径）。
- 使用 `--skip-invalid-nifti` 可以跳过坏样本继续训练。
- 训练摘要里的 `invalid_nifti_report` 会记录坏样本统计和示例。

如果只想先划分数据：
```bash
python script/prepare_rsna_splits.py \
  --series-csv /data/datasets/rsna_hemorrhage_series_labels.csv \
  --output-dir outputs/rsna_splits \
  --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 \
  --seed 42
```

### 5) 最终推理（nii.gz -> 出血概率 + 热图）
```bash
python script/infer_meddinov3_ich.py \
  --input /data/datasets/rsna_hemorrhage_nii_1mm/ID_xxx/ID_yyy/_123.nii.gz \
  --checkpoint checkpoints/rsna_ich_head/best.pt \
  --output-dir outputs/ich_case_xxx \
  --device gpu \
  --batch 8 --resize 224 --top-k 5
```

输出：
- `summary.json`：6 类概率、阈值判定、Top-K 可疑切片
- `similarity_map.png`：可疑切片 patch 相似度热图
- `slice_probs.npy`（当 head 输入维度与 slice embedding 一致时）

## 结果文件说明
推理阶段：
- `summary.json`：输入、切片数、embedding 维度、耗时、显存/内存统计
- `similarity_map.npy/.png`：示例切片 patch 级相似度图

训练阶段：
- `train_summary.json`：配置、训练历史、最佳指标和 checkpoint 路径

测试阶段：
- `metrics.json`：最终评估指标

一体化 pipeline：
- `pipeline_summary.json`：划分统计、训练摘要、测试指标

## 备注
- 当前仓库以“冻结 backbone + 简单头”的 baseline 为主，便于快速迭代。
- 若后续需要病例级 MIL、Top-K slice 聚合、Grad-CAM/attention rollout，可在现有结构上扩展。
