"""
基于 RNN (GRU/LSTM) 的 Seq2Seq 机器翻译模型
包含 Encoder、Decoder 和 Attention 机制

架构要求：
- 使用 GRU 或 LSTM（2 层，单向）作为 Encoder 和 Decoder
- 必须独立实现 Attention 机制（支持点积、乘法、加法对齐函数）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# -----------------------------------------------------------------------------
# Attention 机制
# -----------------------------------------------------------------------------
class Attention(nn.Module):
    """注意力机制模块，支持多种对齐函数。
    
    支持的对齐方式：
    - 'dot': 点积注意力 (Dot-Product Attention)
    - 'multiplicative': 乘法注意力 (Luong's multiplicative)
    - 'additive': 加法注意力 (Bahdanau's additive)
    
    Args:
        hidden_dim (int): 隐藏层维度
        method (str): 注意力计算方法，可选 'dot', 'multiplicative', 'additive'
    """
    
    def __init__(self, hidden_dim: int, method: str = 'dot') -> None:
        super().__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        
        # 根据不同的注意力方法定义所需的参数
        if method == 'multiplicative':
            # 乘法注意力：score = h_t^T * W * h_s
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif method == 'additive':
            # 加法注意力：score = v^T * tanh(W_1 * h_t + W_2 * h_s)
            self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算注意力权重和上下文向量。
        
        Args:
            decoder_hidden (Tensor): 解码器当前隐藏状态 (B, hidden_dim)
            encoder_outputs (Tensor): 编码器所有时间步的输出 (B, L_src, hidden_dim)
            mask (Tensor, optional): 源序列的 padding mask (B, L_src)，1 表示有效，0 表示 padding
            
        Returns:
            context (Tensor): 上下文向量 (B, hidden_dim)
            attn_weights (Tensor): 注意力权重 (B, L_src)
        """
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # 计算注意力分数
        if self.method == 'dot':
            # 点积注意力: score = h_t^T * h_s
            # decoder_hidden: (B, hidden_dim) -> (B, 1, hidden_dim)
            # encoder_outputs: (B, L_src, hidden_dim)
            # scores: (B, 1, hidden_dim) @ (B, hidden_dim, L_src) -> (B, 1, L_src)
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),
                encoder_outputs.transpose(1, 2)
            ).squeeze(1)  # (B, L_src)
            
        elif self.method == 'multiplicative':
            # 乘法注意力: score = h_t^T * W * h_s
            # 先对 encoder_outputs 做线性变换
            # (B, L_src, hidden_dim) -> (B, L_src, hidden_dim)
            transformed = self.W(encoder_outputs)
            # scores: (B, 1, hidden_dim) @ (B, hidden_dim, L_src) -> (B, L_src)
            scores = torch.bmm(
                decoder_hidden.unsqueeze(1),
                transformed.transpose(1, 2)
            ).squeeze(1)
            
        elif self.method == 'additive':
            # 加法注意力: score = v^T * tanh(W_1 * h_t + W_2 * h_s)
            # decoder_hidden: (B, hidden_dim) -> (B, 1, hidden_dim) -> (B, L_src, hidden_dim)
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            # 计算 W_1 * h_t + W_2 * h_s
            energy = torch.tanh(self.W1(decoder_expanded) + self.W2(encoder_outputs))
            # scores: (B, L_src, hidden_dim) -> (B, L_src, 1) -> (B, L_src)
            scores = self.v(energy).squeeze(-1)
        
        # 应用掩码（将 padding 位置的分数设为负无穷）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (B, L_src)
        
        # 计算上下文向量
        # attn_weights: (B, 1, L_src) @ encoder_outputs: (B, L_src, hidden_dim) -> (B, 1, hidden_dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, hidden_dim)
        
        return context, attn_weights


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------
class RNNEncoder(nn.Module):
    """基于 GRU/LSTM 的编码器。
    
    Args:
        vocab_size (int): 源语言词表大小
        emb_dim (int): 词嵌入维度
        hidden_dim (int): RNN 隐藏层维度
        num_layers (int): RNN 层数，默认为 2
        dropout (float): Dropout 概率
        rnn_type (str): RNN 类型，'gru' 或 'lstm'
        bidirectional (bool): 是否使用双向 RNN，默认为 False（单向）
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = 'gru',
        bidirectional: bool = False,
        padding_idx: int = 0
    ) -> None:
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 选择 RNN 类型
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"不支持的 RNN 类型: {rnn_type}，请使用 'gru' 或 'lstm'")
        
        # 如果是双向 RNN，需要将输出维度映射回 hidden_dim
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码器前向传播。
        
        Args:
            src (Tensor): 源序列 token ids (B, L_src)
            src_lengths (Tensor, optional): 源序列实际长度 (B,)
            
        Returns:
            outputs (Tensor): 所有时间步的隐藏状态 (B, L_src, hidden_dim)
            hidden (Tensor): 最终隐藏状态
                - GRU: (num_layers, B, hidden_dim)
                - LSTM: tuple of (h_n, c_n)，每个形状为 (num_layers, B, hidden_dim)
        """
        # 词嵌入 + Dropout
        embedded = self.dropout(self.embedding(src))  # (B, L_src, emb_dim)
        
        # 通过 RNN
        if src_lengths is not None:
            # 使用 pack_padded_sequence 处理变长序列
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        # 如果是双向 RNN，需要处理输出和隐藏状态
        if self.bidirectional:
            # 合并双向输出: (B, L_src, hidden_dim * 2) -> (B, L_src, hidden_dim)
            outputs = self.fc(outputs)
            
            # 处理隐藏状态
            if self.rnn_type == 'lstm':
                # hidden 是 (h_n, c_n) 的元组
                h_n, c_n = hidden
                # 将双向的隐藏状态合并: (num_layers * 2, B, hidden_dim) -> (num_layers, B, hidden_dim * 2)
                h_n = self._merge_bidirectional_hidden(h_n)
                c_n = self._merge_bidirectional_hidden(c_n)
                # 线性变换回 hidden_dim
                h_n = torch.tanh(self.fc(h_n))
                c_n = torch.tanh(self.fc(c_n))
                hidden = (h_n, c_n)
            else:
                # GRU 只有 h_n
                hidden = self._merge_bidirectional_hidden(hidden)
                hidden = torch.tanh(self.fc(hidden))
        
        return outputs, hidden
    
    def _merge_bidirectional_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """合并双向 RNN 的隐藏状态。
        
        Args:
            hidden (Tensor): (num_layers * 2, B, hidden_dim)
            
        Returns:
            Tensor: (num_layers, B, hidden_dim * 2)
        """
        # hidden: (num_layers * 2, B, hidden_dim)
        # 重塑为 (num_layers, 2, B, hidden_dim)，然后沿最后一维拼接
        num_layers = hidden.size(0) // 2
        batch_size = hidden.size(1)
        hidden_dim = hidden.size(2)
        
        hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
        # 拼接前向和后向: (num_layers, B, hidden_dim * 2)
        hidden = torch.cat([hidden[:, 0, :, :], hidden[:, 1, :, :]], dim=-1)
        return hidden


# -----------------------------------------------------------------------------
# Decoder
# -----------------------------------------------------------------------------
class RNNDecoder(nn.Module):
    """带注意力机制的 RNN 解码器。
    
    Args:
        vocab_size (int): 目标语言词表大小
        emb_dim (int): 词嵌入维度
        hidden_dim (int): RNN 隐藏层维度
        num_layers (int): RNN 层数
        dropout (float): Dropout 概率
        rnn_type (str): RNN 类型，'gru' 或 'lstm'
        attention_method (str): 注意力计算方法
    """
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        rnn_type: str = 'gru',
        attention_method: str = 'dot',
        padding_idx: int = 0
    ) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 注意力机制
        self.attention = Attention(hidden_dim, method=attention_method)
        
        # 选择 RNN 类型
        # 输入维度 = 词嵌入维度 + 上下文向量维度（来自注意力）
        rnn_input_dim = emb_dim + hidden_dim
        
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=rnn_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的 RNN 类型: {rnn_type}，请使用 'gru' 或 'lstm'")
        
        # 输出投影层：将 RNN 输出 + 上下文向量映射到词表大小
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
    
    def forward(
        self,
        tgt: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """解码器前向传播（训练模式，使用 teacher forcing）。
        
        Args:
            tgt (Tensor): 目标序列 token ids (B, L_tgt)
            hidden: 初始隐藏状态
                - GRU: (num_layers, B, hidden_dim)
                - LSTM: tuple of (h_n, c_n)
            encoder_outputs (Tensor): 编码器输出 (B, L_src, hidden_dim)
            src_mask (Tensor, optional): 源序列 padding mask (B, L_src)
            
        Returns:
            outputs (Tensor): 每个时间步的词表概率 (B, L_tgt, vocab_size)
            hidden: 最终隐藏状态
            attn_weights (Tensor): 注意力权重 (B, L_tgt, L_src)
        """
        batch_size = tgt.size(0)
        tgt_len = tgt.size(1)
        
        # 词嵌入
        embedded = self.dropout(self.embedding(tgt))  # (B, L_tgt, emb_dim)
        
        # 初始化输出和注意力权重收集器
        outputs = []
        attn_weights_list = []
        
        # 获取初始隐藏状态的最后一层（用于注意力计算）
        if self.rnn_type == 'lstm':
            h_t = hidden[0][-1]  # (B, hidden_dim)
        else:
            h_t = hidden[-1]  # (B, hidden_dim)
        
        # 逐时间步解码
        for t in range(tgt_len):
            # 获取当前时间步的嵌入
            emb_t = embedded[:, t, :]  # (B, emb_dim)
            
            # 计算注意力和上下文向量
            context, attn_weight = self.attention(h_t, encoder_outputs, src_mask)
            attn_weights_list.append(attn_weight)
            
            # 将嵌入和上下文向量拼接作为 RNN 输入
            rnn_input = torch.cat([emb_t, context], dim=-1)  # (B, emb_dim + hidden_dim)
            rnn_input = rnn_input.unsqueeze(1)  # (B, 1, emb_dim + hidden_dim)
            
            # 通过 RNN
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            rnn_output = rnn_output.squeeze(1)  # (B, hidden_dim)
            
            # 更新用于注意力的隐藏状态
            if self.rnn_type == 'lstm':
                h_t = hidden[0][-1]
            else:
                h_t = hidden[-1]
            
            # 将 RNN 输出和上下文向量拼接，通过输出层
            output = self.fc_out(torch.cat([rnn_output, context], dim=-1))  # (B, vocab_size)
            outputs.append(output)
        
        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=1)  # (B, L_tgt, vocab_size)
        attn_weights = torch.stack(attn_weights_list, dim=1)  # (B, L_tgt, L_src)
        
        return outputs, hidden, attn_weights
    
    def forward_step(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        context: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单步解码（用于推理时的贪婪/束搜索解码）。
        
        Args:
            tgt_token (Tensor): 当前时间步的 token id (B, 1)
            hidden: 当前隐藏状态
            encoder_outputs (Tensor): 编码器输出 (B, L_src, hidden_dim)
            context (Tensor): 上一步的上下文向量 (B, hidden_dim)
            src_mask (Tensor, optional): 源序列 padding mask
            
        Returns:
            output (Tensor): 当前时间步的词表 logits (B, vocab_size)
            hidden: 更新后的隐藏状态
            context (Tensor): 当前时间步的上下文向量
            attn_weight (Tensor): 注意力权重 (B, L_src)
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(tgt_token.squeeze(1)))  # (B, emb_dim)
        
        # 将嵌入和上下文向量拼接作为 RNN 输入
        rnn_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)  # (B, 1, emb_dim + hidden_dim)
        
        # 通过 RNN
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        rnn_output = rnn_output.squeeze(1)  # (B, hidden_dim)
        
        # 获取用于注意力的隐藏状态
        if self.rnn_type == 'lstm':
            h_t = hidden[0][-1]
        else:
            h_t = hidden[-1]
        
        # 计算新的注意力和上下文向量
        context, attn_weight = self.attention(h_t, encoder_outputs, src_mask)
        
        # 通过输出层
        output = self.fc_out(torch.cat([rnn_output, context], dim=-1))  # (B, vocab_size)
        
        return output, hidden, context, attn_weight


# -----------------------------------------------------------------------------
# Seq2Seq 模型
# -----------------------------------------------------------------------------
class Seq2SeqRNN(nn.Module):
    """完整的 Seq2Seq RNN 模型，封装了编码器、解码器和注意力机制。
    
    提供与 Transformer 模型相同的 API 接口，便于统一训练和评估。
    
    Args:
        num_encoder_layers (int): 编码器 RNN 层数
        num_decoder_layers (int): 解码器 RNN 层数
        emb_size (int): 词嵌入维度
        hidden_size (int): RNN 隐藏层维度
        src_vocab_size (int): 源语言词表大小
        tgt_vocab_size (int): 目标语言词表大小
        dropout (float): Dropout 概率
        rnn_type (str): RNN 类型，'gru' 或 'lstm'
        attention_method (str): 注意力方法
        pad_id (int): padding token 的 id
    """
    
    def __init__(
        self,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        emb_size: int = 256,
        hidden_size: int = 512,
        src_vocab_size: int = 10000,
        tgt_vocab_size: int = 10000,
        dropout: float = 0.1,
        rnn_type: str = 'gru',
        attention_method: str = 'dot',
        pad_id: int = 0
    ) -> None:
        super().__init__()
        
        self.pad_id = pad_id
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type.lower()
        
        # 编码器
        self.encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_size,
            hidden_dim=hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=False,  # 按要求使用单向 RNN
            padding_idx=pad_id
        )
        
        # 解码器
        self.decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_size,
            hidden_dim=hidden_size,
            num_layers=num_decoder_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            attention_method=attention_method,
            padding_idx=pad_id
        )
    
    def _make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """创建源序列的 padding mask。
        
        Args:
            src (Tensor): 源序列 (B, L_src)
            
        Returns:
            Tensor: padding mask (B, L_src)，1 表示有效，0 表示 padding
        """
        return (src != self.pad_id).float()
    
    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor
    ) -> torch.Tensor:
        """模型前向传播（训练模式）。
        
        Args:
            src_ids (Tensor): 源序列 token ids (B, L_src)
            tgt_ids (Tensor): 目标序列 token ids (B, L_tgt)
            
        Returns:
            Tensor: 输出 logits (B, L_tgt, tgt_vocab_size)
        """
        # 创建源序列 mask
        src_mask = self._make_src_mask(src_ids)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src_ids)
        
        # 解码（使用 teacher forcing）
        outputs, _, _ = self.decoder(tgt_ids, hidden, encoder_outputs, src_mask)
        
        return outputs
    
    def encode(
        self,
        src_ids: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码源序列。
        
        Args:
            src_ids (Tensor): 源序列 token ids (B, L_src)
            src_mask (Tensor, optional): 未使用，保持与 Transformer 接口一致
            
        Returns:
            encoder_outputs (Tensor): 编码器输出 (B, L_src, hidden_size)
            hidden: 编码器最终隐藏状态
        """
        encoder_outputs, hidden = self.encoder(src_ids)
        return encoder_outputs, hidden
    
    def decode(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """解码目标序列。
        
        Args:
            tgt_ids (Tensor): 目标序列 token ids (B, L_tgt)
            memory (Tensor): 编码器输出 (B, L_src, hidden_size)
            hidden: 解码器初始隐藏状态
            tgt_mask (Tensor, optional): 未使用
            src_mask (Tensor, optional): 源序列 padding mask
            
        Returns:
            outputs (Tensor): 解码器输出 (B, L_tgt, hidden_size)
            hidden: 解码器最终隐藏状态
        """
        outputs, hidden, _ = self.decoder(tgt_ids, hidden, memory, src_mask)
        return outputs, hidden
    
    def generator(self, dec_out: torch.Tensor) -> torch.Tensor:
        """将解码器输出转换为词表概率（仅用于兼容 Transformer API）。
        
        注意：RNN 解码器已经在内部完成了投影，这里直接返回输入。
        """
        return dec_out


# -----------------------------------------------------------------------------
# 用于推理的辅助函数
# -----------------------------------------------------------------------------
def translate_sentence_rnn(
    sentence_zh: str,
    model: Seq2SeqRNN,
    tokenizer,
    device: torch.device,
    max_len: int = 100
) -> str:
    """使用 RNN 模型进行贪婪解码翻译。
    
    Args:
        sentence_zh (str): 中文原句
        model (Seq2SeqRNN): 训练好的 RNN 模型
        tokenizer: 分词器实例
        device: 计算设备
        max_len (int): 最大生成长度
        
    Returns:
        str: 翻译后的英文句子
    """
    model.eval()
    
    # 编码源句子
    src_ids = torch.tensor(
        tokenizer.encode_src(sentence_zh),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)  # (1, L_src)
    
    # 创建源序列 mask
    src_mask = model._make_src_mask(src_ids)
    
    with torch.no_grad():
        # 编码
        encoder_outputs, hidden = model.encode(src_ids)
        
        # 初始化解码器输入
        tgt_token = torch.tensor(
            [[tokenizer.sos_token_id]],
            dtype=torch.long,
            device=device
        )  # (1, 1)
        
        # 初始化上下文向量（全零）
        context = torch.zeros(1, model.hidden_size, device=device)
        
        # 收集生成的 token
        generated_ids = [tokenizer.sos_token_id]
        
        for _ in range(max_len):
            # 单步解码
            output, hidden, context, _ = model.decoder.forward_step(
                tgt_token, hidden, encoder_outputs, context, src_mask
            )
            
            # 贪婪选择最大概率的 token
            next_id = output.argmax(dim=-1).item()
            generated_ids.append(next_id)
            
            # 检查是否生成结束符
            if next_id == tokenizer.eos_token_id:
                break
            
            # 更新输入 token
            tgt_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
    
    # 解码生成的 token 序列
    return tokenizer.decode_tgt(generated_ids)
