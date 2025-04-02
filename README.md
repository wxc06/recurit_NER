```markdown
# BERT-BiLSTM-CRF for Named Entity Recognition

本项目实现了一个基于 BERT 的命名实体识别（NER）系统，融合了 BiLSTM 和 CRF 层，针对招聘场景中的职位信息进行结构化抽取。支持模型训练、验证、预测等完整流程，适用于岗位标准化、人岗匹配、冷启动推荐等下游任务。

## 项目背景

在招聘场景中，职位文本通常表述方式多样、缺乏统一标准。为了从中抽取出如“岗位层级”、“部门方向”等结构化字段，本项目构建了一个端到端的实体识别模型，并结合多个优化模块，提升了对长尾实体、噪声样本的鲁棒性。

## 项目结构
```
```
.
├── config.py                # 路径与超参设置
├── data_process.py         # 原始文本预处理与BIO标签构建
├── data_loader.py          # 自定义Dataset与DataLoader
├── Model_structure.py      # 模型结构：BERT + BiLSTM + CRF
├── Model_training.py       # 训练逻辑（含 early stopping、Focal Loss 支持）
├── train.py                # 训练入口脚本
├── predict.py              # 推理与结果保存（支持CSV/JSON）
├── supplement.py           # F1评估与bad case输出
├── logger_settings.py      # 日志记录配置
└── README.md
```

## 快速开始

### 环境依赖

- Python 3.9+
- PyTorch >= 1.13
- Transformers >= 4.24
- scikit-learn
- tqdm
- fire

```bash
pip install -r requirements.txt  # 示例安装
```

### 数据预处理

将原始数据格式转换为模型输入格式，默认使用 `.npz` 格式加速加载。格式示例：

```python
# 原始输入：
words = ['senior', 'manager']
labels = ['B-RES', 'E-RES']
```

在 `data_process.py` 中完成清洗、分词、标签构建等操作。

### 模型训练

请先在 `config.py` 中设定路径、batch size、epoch 等参数：

```bash
python train.py
```

训练过程中将自动保存表现最优的模型权重。

### 模型预测

支持 CSV 文件输入、结构化预测输出：

```bash
python predict.py \
  --input_dir data/inference_test.csv \
  --col_name title \
  --output_dir output/pred.json
```

输出为标准化的 JSON 格式结果，可直接集成入下游推荐系统或用户画像平台。

---

## 关键特性

- ✅ **实体级别预测精度优化**：F1 在测试集上达 90.5%，相较 BiLSTM-CRF 提升超 12%
- ✅ **Focal Loss 引入**：缓解类间不平衡，提升长尾职位识别能力
- ✅ **结构清晰、便于迁移**：可拓展至多语言 NER、医疗/法律等其他垂直领域
- ✅ **支持标准化输出**：JSON 可兼容推荐系统、标签体系、ES索引等场景

---

## 📁 文件说明

| 文件名 | 描述 |
|--------|------|
| `config.py` | 所有路径与超参统一管理 |
| `train.py` | 主训练脚本 |
| `predict.py` | 推理入口，支持 CLI 输入/输出 |
| `Model_structure.py` | 模型主体：BERT + BiLSTM + CRF |
| `Model_training.py` | 训练、评估、early stopping、Focal Loss 等逻辑 |
| `supplement.py` | F1评估与错误分析 |
| `logger_settings.py` | 日志记录配置模块 |
| `data_loader.py` | 自定义NER任务的Dataloader封装 |
| `data_process.py` | 数据预处理、BIO标签生成 |

---

## 🧠 模型结构图

```text
Input Text
    ↓
Tokenizer + Segment Embedding
    ↓
BERT Encoder
    ↓
BiLSTM Sequence Model
    ↓
CRF Tagging Layer
    ↓
NER Labels (e.g. B-RES, I-FUN)
```

---

## ⚠️ 注意事项

> 出于数据保密要求，原始数据未随代码一并开源。若需测试，请自定义构造少量招聘样本文本及标签（BIO格式）即可。

