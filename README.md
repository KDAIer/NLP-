# 中英机器翻译项目 (ZH→EN Machine Translation)

本项目实现了基于深度学习的中文到英文机器翻译系统，包含两种模型架构：**Transformer** 和 **RNN (Seq2Seq with Attention)**。

---

## 📁 项目结构

```
.
├── config.yaml               # 全局配置文件（模型/训练/数据路径）
├── preprocess.py             # 数据预处理脚本
├── train.py                  # 训练脚本（支持 Transformer 和 RNN）
├── evaluate.py               # 评估脚本（计算 BLEU 分数）
├── check_translations.py     # 翻译结果格式验证
├── tokenizer.py              # 分词器基类和实现
├── utils.py                  # 工具函数（数据集、翻译函数等）
├── README.md                 # 本文件
├── README_en.md              # 英文说明文档
│
├── model/
│   ├── transformer.py        # Transformer 模型实现
│   └── rnn.py                # RNN Seq2Seq 模型实现
│
├── data/
│   ├── train_100k.jsonl      # 大训练集（100k 样本）
│   ├── train_10k.jsonl       # 小训练集（10k 样本）
│   ├── valid.jsonl           # 验证集（500 样本）
│   ├── test.jsonl            # 测试集（200 样本）
│   └── processed/            # 预处理后的数据目录
│
└── runs/                     # 模型检查点保存目录
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch pyyaml jieba sacrebleu tqdm
```

**环境要求：**
- Python 3.10+
- PyTorch 2.0+
- CUDA（可选，用于 GPU 加速）

### 2. 数据准备

将数据文件放入 `data/` 目录下，然后运行预处理：

```bash
python preprocess.py -c config.yaml
```

这将生成：
- `data/processed/train.jsonl` - 预处理后的训练数据
- `data/processed/val.jsonl` - 预处理后的验证数据
- `data/processed/test.jsonl` - 预处理后的测试数据
- `data/processed/src_vocab.pkl` - 源语言（中文）词表
- `data/processed/tgt_vocab.pkl` - 目标语言（英文）词表

### 3. 模型训练

#### 训练 Transformer 模型

```bash
# 确保 config.yaml 中 model_type: transformer
python train.py -c config.yaml
```

#### 训练 RNN 模型

修改 `config.yaml`：
```yaml
model_type: rnn
```

然后运行：
```bash
python train.py -c config.yaml
```

### 4. 模型评估

```bash
python evaluate.py -c config.yaml --ckpt runs/model_epoch_10.pt --save_path translations.json
```

### 5. 验证输出格式

```bash
python check_translations.py translations.json
```

---

## ⚙️ 配置说明

### config.yaml 完整配置

```yaml
# ------------- 分词器 -----------------
tokenizer: tokenizer.JiebaEnTokenizer

# ------------- 模型选择 ----------------
# 可选: "transformer" 或 "rnn"
model_type: transformer

# ------------- 模型结构 ----------------
model:
  # === 通用参数 ===
  enc_layers: 4          # Encoder 层数
  dec_layers: 4          # Decoder 层数
  emb_size: 256          # 词向量维度
  dropout: 0.1           # Dropout 概率
  
  # === Transformer 专用参数 ===
  nhead: 8               # Multi-Head Attention 头数
  ffn_dim: 1024          # Feed-Forward 隐层维度
  
  # === RNN 专用参数 ===
  rnn_type: gru          # RNN 类型: "gru" 或 "lstm"
  hidden_size: 512       # RNN 隐藏层维度
  attention_method: dot  # 注意力方法: "dot", "multiplicative", "additive"

# ------------- 训练超参 ----------------
train:
  batch_size: 64
  epochs: 10
  lr: 0.0003
  weight_decay: 0.0001
  lr_step: 8
  lr_gamma: 0.5
  save_dir: runs
  num_workers: 0

# ------------- 数据路径 ----------------
data:
  raw_train: data/train_10k.jsonl
  raw_val: data/valid.jsonl
  raw_test: data/test.jsonl
  processed_dir: data/processed
  train_processed: data/processed/train.jsonl
  val_processed: data/processed/val.jsonl
  test_processed: data/processed/test.jsonl
  src_vocab: data/processed/src_vocab.pkl
  tgt_vocab: data/processed/tgt_vocab.pkl
  min_freq: 1

# ------------- 其余 --------------------
seed: 3407
```

---

## 🏗️ 模型架构

### 1. Transformer 模型

基于 "Attention Is All You Need" 论文实现，包含：

| 组件 | 说明 |
|------|------|
| **PositionalEncoding** | 正弦/余弦位置编码，注入序列位置信息 |
| **MultiHeadAttention** | 多头自注意力机制，支持 Q/K/V 投影和缩放点积 |
| **EncoderLayer** | 自注意力 + FFN + 残差连接 + 层归一化 |
| **DecoderLayer** | 掩码自注意力 + 交叉注意力 + FFN |
| **Encoder** | N 层 EncoderLayer 堆叠 |
| **Decoder** | N 层 DecoderLayer 堆叠 |

**模型流程：**
```
源序列 → Embedding → PositionalEncoding → Encoder Layers → Memory
目标序列 → Embedding → PositionalEncoding → Decoder Layers (with Memory) → Linear → Softmax
```

### 2. RNN Seq2Seq 模型

基于 GRU/LSTM 的编码器-解码器架构，包含三种注意力机制：

| 注意力方法 | 计算公式 |
|-----------|---------|
| **Dot (点积)** | $score = h_t^T \cdot h_s$ |
| **Multiplicative (乘法)** | $score = h_t^T W h_s$ |
| **Additive (加法)** | $score = v^T \tanh(W_1 h_t + W_2 h_s)$ |

**模型组件：**

| 组件 | 说明 |
|------|------|
| **RNNEncoder** | 2 层单向 GRU/LSTM，将源序列编码为隐藏状态 |
| **Attention** | 支持点积、乘法、加法三种对齐函数 |
| **RNNDecoder** | 带注意力机制的解码器，使用 Teacher Forcing 训练 |

**模型流程：**
```
源序列 → Embedding → RNN Encoder → (encoder_outputs, hidden)
                                         ↓
目标序列 → Embedding → RNN Decoder (with Attention) → Linear → Softmax
```

---

## 📊 使用示例

### 切换模型类型

**使用 Transformer：**
```yaml
model_type: transformer
model:
  enc_layers: 4
  dec_layers: 4
  emb_size: 256
  nhead: 8
  ffn_dim: 1024
```

**使用 GRU + 点积注意力：**
```yaml
model_type: rnn
model:
  enc_layers: 2
  dec_layers: 2
  emb_size: 256
  hidden_size: 512
  rnn_type: gru
  attention_method: dot
```

**使用 LSTM + 加法注意力：**
```yaml
model_type: rnn
model:
  enc_layers: 2
  dec_layers: 2
  emb_size: 256
  hidden_size: 512
  rnn_type: lstm
  attention_method: additive
```

### 训练命令

```bash
# 使用默认配置
python train.py -c config.yaml

# 使用自定义配置
python train.py -c config_rnn.yaml
```

### 评估命令

```bash
# 评估并保存翻译结果
python evaluate.py -c config.yaml --ckpt runs/model_epoch_10.pt --save_path translations.json

# 查看 BLEU 分数和翻译样例
```

---

## 📝 输出格式

评估脚本输出的 `translations.json` 格式：

```json
[
  {
    "src": "今天天气很好",
    "ref": "It is a fine day today",
    "hyp": "The weather is great today",
    "hyp_id": "sha256_hash..."
  },
  ...
]
```

---

## 🔧 常见问题

### Q1: 如何选择 Transformer 还是 RNN？

- **Transformer**：适合较长序列，并行计算效率高，效果通常更好
- **RNN**：参数量较少，适合资源受限环境，更容易理解和调试

### Q2: 如何调整模型大小？

- 减小 `emb_size` 和 `hidden_size` 可以减少参数量
- 减少 `enc_layers` 和 `dec_layers` 可以加快训练速度
- 对于 Transformer，减少 `nhead` 和 `ffn_dim` 也可以减小模型

### Q3: 训练时 OOM（显存不足）怎么办？

1. 减小 `batch_size`
2. 减小模型规模（层数、维度）
3. 使用梯度累积
4. 使用混合精度训练

### Q4: 如何提高 BLEU 分数？

1. 使用更大的训练集（100k 而非 10k）
2. 增加训练轮数
3. 调整学习率和正则化参数
4. 尝试不同的模型架构和注意力机制

---

## 📚 参考资料

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Bahdanau, D., et al. "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015.
3. Luong, M., et al. "Effective Approaches to Attention-based Neural Machine Translation." EMNLP 2015.

---

## 📄 提交清单

提交前请确保包含以下文件：

- [ ] `model/transformer.py` - 完整的 Transformer 实现
- [ ] `model/rnn.py` - 完整的 RNN Seq2Seq 实现
- [ ] `runs/best_model.pt` - 最佳模型检查点
- [ ] `translations.json` - 翻译结果文件
- [ ] 项目报告 (PDF)

验证输出格式：
```bash
python check_translations.py translations.json
```

---

## 👨‍💻 作者

2025 秋季学期 - 人工神经网络期末项目
