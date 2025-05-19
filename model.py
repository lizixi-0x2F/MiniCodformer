import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Optional, Tuple

# FastGELU实现，比标准GELU更高效
class FastGELU(nn.Module):
    def forward(self, x):
        return F.gelu(x, approximate='tanh')

class EncoderLayer(nn.Module):
    """Mini Transformer Encoder Layer"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            FastGELU(),  # 使用更高效的FastGELU
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        # Self Attention
        residual = hidden_states
        hidden_states, _ = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=attention_mask
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm1(residual + hidden_states)
        
        # Feed Forward
        residual = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(residual + hidden_states)
        
        return hidden_states

class MiniTransformerEncoder(nn.Module):
    """Mini Transformer Encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        
        # 创建位置索引
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)
        
        # 嵌入层
        inputs_embeds = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        
        hidden_states = self.layer_norm(embeddings)
        hidden_states = self.dropout(hidden_states)
        
        # 注意力掩码转换为布尔掩码 (1->False, 0->True)
        if attention_mask is not None:
            attention_mask = attention_mask.eq(0)
        
        # 通过编码器层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        return hidden_states

class LTCNonCausalLayer(nn.Module):
    """Linear Transformer with Continuous Non-Causal Processing Layer"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.ltc_ncp_hidden_size
        self.kernel_size = config.ltc_kernel_size
        
        # 确保kernel_size为奇数（现在在配置中已经确保了）
        padding = self.kernel_size // 2
            
        # Convolution层用于捕获局部上下文
        self.conv = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            padding=padding,  # 确保输出序列长度与输入相同
            groups=1
        )
        
        # 线性层
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size * 4)
        self.linear2 = nn.Linear(self.hidden_size * 4, self.hidden_size)
        
        # LayerNorm
        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.layer_norm2 = nn.LayerNorm(self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # FastGELU激活函数
        self.activation = FastGELU()
        
        # 添加滑动窗口状态缓存属性
        self.cache_len = self.kernel_size - 1
        self.cached_states = None
        
    def forward(self, hidden_states, use_cache=False, extend_cache=False):
        residual = hidden_states
        
        # 保存原始序列长度
        original_seq_len = hidden_states.size(1)
        
        # 卷积部分 - 需要转置为[batch, channels, seq_len]格式
        conv_input = hidden_states.transpose(1, 2)
        
        # 滑动窗口卷积状态缓存处理
        if use_cache:
            if extend_cache and self.cached_states is not None:
                # 只处理最后一个token，连接之前的缓存
                if hidden_states.size(1) == 1:
                    # 连接缓存和当前输入
                    cached_input = self.cached_states
                    combined_input = torch.cat([cached_input, conv_input], dim=2)
                    
                    # 计算完整结果
                    conv_output = self.conv(combined_input)
                    
                    # 只取最后一个位置的输出
                    conv_output = conv_output[:, :, -1:]
                    
                    # 更新缓存（保留最新的cache_len个token）
                    self.cached_states = combined_input[:, :, -self.cache_len:]
                else:
                    # 处理batch情况或首次计算
                    conv_output = self.conv(conv_input)
                    # 更新缓存
                    self.cached_states = conv_input[:, :, -self.cache_len:]
            else:
                # 首次调用或不扩展缓存
                conv_output = self.conv(conv_input)
                # 初始化缓存
                self.cached_states = conv_input[:, :, -self.cache_len:]
        else:
            # 不使用缓存，正常计算
            conv_output = self.conv(conv_input)
        
        # 确保输出序列长度与输入相同
        if not use_cache and conv_output.size(2) != original_seq_len:
            # 如果卷积输出长度不匹配，截断或补齐
            if conv_output.size(2) > original_seq_len:
                # 截断多余部分
                conv_output = conv_output[:, :, :original_seq_len]
            else:
                # 如果长度不足（极少情况），使用填充
                padding = torch.zeros(
                    conv_output.size(0), 
                    conv_output.size(1), 
                    original_seq_len - conv_output.size(2),
                    device=conv_output.device,
                    dtype=conv_output.dtype
                )
                conv_output = torch.cat([conv_output, padding], dim=2)
        
        hidden_states = conv_output.transpose(1, 2)  # 转回[batch, seq_len, channels]
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm1(residual + hidden_states)
        
        # Feed Forward网络
        residual = hidden_states
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)  # 使用FastGELU
        hidden_states = self.linear2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm2(residual + hidden_states)
        
        return hidden_states
        
    def reset_cache(self):
        """重置卷积状态缓存"""
        self.cached_states = None

class LTCNCPDecoder(nn.Module):
    """Linear Transformer with Continuous Non-Causal Processing Decoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            LTCNonCausalLayer(config) for _ in range(config.ltc_ncp_num_layers)
        ])
        
        # 输出投影层 - 确保使用准确的词汇表大小
        self.output_projection = nn.Linear(config.ltc_ncp_hidden_size, config.vocab_size)
        
    def forward(self, hidden_states, use_cache=False, extend_cache=False):
        # 通过LTC-NCP层
        for layer in self.layers:
            hidden_states = layer(hidden_states, use_cache=use_cache, extend_cache=extend_cache)
            
        # 投影到词汇表
        logits = self.output_projection(hidden_states)
        
        return logits
        
    def reset_cache(self):
        """重置所有层的缓存"""
        for layer in self.layers:
            layer.reset_cache()

class StudentTransformer(nn.Module):
    """Only used during training for supervision"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用与Mini Transformer相同配置的Transformer decoder
        self.decoder = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, hidden_states, attention_mask=None):
        # 注意力掩码转换为布尔掩码 (1->False, 0->True)
        if attention_mask is not None:
            # 转换为PyTorch多头注意力需要的格式
            attention_mask = attention_mask.eq(0)
        
        # 通过decoder层
        for layer in self.decoder:
            hidden_states = layer(hidden_states, attention_mask)
            
        # 投影到词汇表
        logits = self.output_projection(hidden_states)
        
        return logits

class DistillationModel(nn.Module):
    """完整的蒸馏模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = MiniTransformerEncoder(config)
        
        # 添加一个投影层以匹配维度
        self.encoder_to_decoder_projection = nn.Linear(
            config.hidden_size, 
            config.ltc_ncp_hidden_size
        )
        
        # LTC-NCP Decoder
        self.ltc_ncp_decoder = LTCNCPDecoder(config)
        
        # 监督用Transformer (仅训练时启用)
        if config.distill_supervision:
            self.supervision_transformer = StudentTransformer(config)
            
            # 权重共享：让supervision_transformer的输出层与encoder的嵌入层共享权重
            if getattr(config, 'share_embeddings', True):
                self.supervision_transformer.output_projection.weight = self.encoder.embeddings.weight
        else:
            self.supervision_transformer = None
            
        # 权重共享：让LTC-NCP解码器的输出层与编码器嵌入层共享权重
        if getattr(config, 'share_embeddings', True):
            self.ltc_ncp_decoder.output_projection.weight = self.encoder.embeddings.weight
            
        # 词嵌入层(与encoder共享)
        self.get_output_embeddings = lambda: self.encoder.embeddings
        
        # 启用TF32加速（如果配置允许且在支持的硬件上）
        if getattr(config, 'enable_tf32', True) and torch.cuda.is_available():
            # 这只会影响Ampere及以上架构的GPU
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # 知识蒸馏超参数
        self.distill_top_k = config.distill_top_k
        self.temperature = config.temperature
        
        # 添加教师输出适配器，使用配置中的词汇表大小
        self.teacher_adapter = TeacherOutputAdapter(config.vocab_size)  # 使用配置中的词汇表大小
        
    def save_pretrained(self, save_directory):
        """保存模型到指定目录"""
        import os
        import json
        import torch
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型配置
        # 创建可序列化的配置字典
        config_dict = {}
        for key, value in vars(self.config).items():
            # 跳过不可序列化的类型
            if key == 'teacher_device' and isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_dict[key] = value
            else:
                # 尝试转换为字符串
                try:
                    config_dict[key] = str(value)
                except:
                    print(f"警告: 配置项 '{key}' 无法序列化，已跳过")
        
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # 保存模型权重
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        
        # 如果是DDP模型，需要获取内部模型
        if hasattr(self, "module"):
            model_to_save = self.module
        else:
            model_to_save = self
            
        # 保存为state_dict，方便加载
        torch.save(model_to_save.state_dict(), model_path)
        
        print(f"模型已保存到 {save_directory}")
        
    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """从预训练目录加载模型"""
        import os
        import json
        import torch
        from dataclasses import dataclass, asdict
        
        # 内部定义ModelConfig类
        @dataclass
        class ModelConfig:
            vocab_size: int = 151669
            hidden_size: int = 2048
            num_hidden_layers: int = 24
            num_attention_heads: int = 16
            intermediate_size: int = 8192
            hidden_dropout_prob: float = 0.1
            attention_probs_dropout_prob: float = 0.1
            max_position_embeddings: int = 2048
            ltc_ncp_hidden_size: int = 2048
            ltc_ncp_num_layers: int = 12
            ltc_kernel_size: int = 3
            distill_supervision: bool = True
            share_embeddings: bool = True
            distill_logits: bool = True
            distill_hiddens: bool = True
            temperature: float = 2.0
            distill_top_k: int = 20
            ignore_padding: bool = True
            pad_token_id: int = 0
            eos_token_id: int = 2
            bos_token_id: int = 1
            model_type: str = "distillation_model"
            
            def to_json_string(self):
                """序列化配置为JSON字符串"""
                import json
                config_dict = {k: v for k, v in vars(self).items()}
                return json.dumps(config_dict, indent=2)
                
            def to_dict(self):
                """将配置转换为字典"""
                return {k: v for k, v in vars(self).items()}
                
            def __getitem__(self, key):
                """支持字典访问方式"""
                return getattr(self, key, None)
                
            def save_pretrained(self, save_directory):
                """保存配置到目录"""
                import os
                import json
                os.makedirs(save_directory, exist_ok=True)
                with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                    f.write(self.to_json_string())
        
        # 加载配置
        with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # 创建配置对象
        config = ModelConfig()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"), 
            map_location=device if device else torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        
        if device:
            model.to(device)
            
        return model
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        teacher_hidden_states=None,
        teacher_logits=None,
        return_dict=True,
        ignore_index=-100,
        top_k=None,
        temperature=2.0,
        hard_weights=None,
        use_soft_labels=True  # 新参数：是否使用软标签蒸馏
    ):
        # 编码器处理输入
        encoder_outputs = self.encoder(input_ids, attention_mask)
        
        # 投影编码器输出到解码器所需的维度
        encoder_outputs = self.encoder_to_decoder_projection(encoder_outputs)
        
        # LTC-NCP解码器处理
        ltc_ncp_logits = self.ltc_ncp_decoder(encoder_outputs)
        
        # 如果启用了监督transformer
        if self.supervision_transformer is not None and self.training:
            supervision_logits = self.supervision_transformer(encoder_outputs, attention_mask)
        else:
            supervision_logits = None
            
        loss = None
        loss_components = {}
        
        if labels is not None:
            # 计算LTC-NCP解码器的损失
            # 如果开启了padding忽略，使用配置中的pad_token_id
            pad_token_id = self.config.pad_token_id if getattr(self.config, 'ignore_padding', True) else -100
            
            # 确保标签在有效范围内
            vocab_size = self.config.vocab_size
            labels_shape = labels.shape
            
            # 第一步：将所有标签设为有效值或忽略索引
            # 任何小于0的值都设为忽略索引
            # 任何大于等于vocab_size的值也设为忽略索引
            valid_labels = torch.where(
                (labels >= 0) & (labels < vocab_size),
                labels,
                torch.tensor(pad_token_id, device=labels.device, dtype=labels.dtype)
            )
            
            # 检查是否有无效标签被替换
            if pad_token_id != -100:
                invalid_count = (labels != valid_labels).sum().item()
                if invalid_count > 0 and not hasattr(self, '_warned_invalid_labels'):
                    print(f"警告：发现{invalid_count}个超出范围的标签，已设置为忽略索引")
                    self._warned_invalid_labels = True
            
            # 标签平滑支持
            label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
            
            try:
                loss_fct = nn.CrossEntropyLoss(
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing
                )
                ltc_ncp_loss = loss_fct(ltc_ncp_logits.view(-1, self.config.vocab_size), valid_labels.view(-1))
                loss_components['ltc_ncp_ce_loss'] = ltc_ncp_loss.item()
            except Exception as e:
                print(f"损失计算出错，尝试使用替代方法: {e}")
                # 如果出错，尝试使用函数式API
                ltc_ncp_loss = F.cross_entropy(
                    ltc_ncp_logits.view(-1, self.config.vocab_size),
                    valid_labels.view(-1),
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing
                )
                loss_components['ltc_ncp_ce_loss'] = ltc_ncp_loss.item()
            
            # 初始化总损失为标准交叉熵损失
            loss = ltc_ncp_loss
            
            # 创建联合KD损失实例
            joint_kd_loss = JointKDLoss(
                temperature=temperature if temperature is not None else self.temperature,
                alpha=0.7,  # 软标签权重
                beta=0.3,   # 硬标签权重
                gamma=0.0   # 特征蒸馏权重 (暂不使用特征蒸馏)
            )
            
            # 知识蒸馏 (teacher-student) 损失
            if teacher_logits is not None and use_soft_labels:
                # 使用adapter处理词汇表大小统一问题
                batch_size, seq_length = input_ids.shape
                
                # 使用adapter调整教师logits的形状以匹配学生输出
                # 这里使用更精确的尺寸处理，而不是简单裁剪
                teacher_logits_adapted = self.teacher_adapter(
                    teacher_logits, 
                    target_batch_size=batch_size,
                    target_seq_length=seq_length
                )
                
                # 应用联合KD损失
                if top_k is not None or self.distill_top_k > 0:
                    actual_top_k = top_k if top_k is not None else self.distill_top_k
                    try:
                        # 对于每个位置，找到教师模型的top-k预测
                        top_k_value = min(actual_top_k, self.config.vocab_size)
                        
                        # 创建遮罩用于筛选top-k词汇
                        batch_size, seq_length = input_ids.shape
                        
                        # 创建蒸馏掩码
                        teacher_masked_logits = torch.zeros_like(teacher_logits_adapted)
                        
                        # 通过循环处理每个序列位置以找到top-k值
                        for b in range(batch_size):
                            for s in range(seq_length):
                                # 获取当前位置教师模型预测的top-k索引
                                values, indices = torch.topk(teacher_logits_adapted[b, s], k=top_k_value)
                                # 复制这些位置的教师输出值到掩码张量中
                                teacher_masked_logits[b, s, indices] = teacher_logits_adapted[b, s, indices]
                        
                        kd_total_loss, kd_losses = joint_kd_loss(
                            student_logits=ltc_ncp_logits,
                        # 应用联合KD损失 (仅包含top-k)
                            teacher_logits=teacher_masked_logits,
                            labels=valid_labels,
                            ignore_index=pad_token_id
                        )
                        
                        # 更新损失组件
                        for k, v in kd_losses.items():
                            loss_components[f'kd_{k}'] = v
                            
                    except Exception as e:
                        print(f"Top-K联合蒸馏计算出错: {e}")
                        # 出错时使用全部词汇表
                        kd_total_loss, kd_losses = joint_kd_loss(
                            student_logits=ltc_ncp_logits,
                            teacher_logits=teacher_logits_adapted,
                            labels=valid_labels,
                            ignore_index=pad_token_id
                        )
                        # 更新损失组件
                        for k, v in kd_losses.items():
                            loss_components[f'kd_{k}'] = v
                else:
                    # 使用全词汇表
                    kd_total_loss, kd_losses = joint_kd_loss(
                        student_logits=ltc_ncp_logits,
                        teacher_logits=teacher_logits_adapted,
                        labels=valid_labels,
                        ignore_index=pad_token_id
                    )
                    # 更新损失组件
                    for k, v in kd_losses.items():
                        loss_components[f'kd_{k}'] = v
                
                # 使用联合KD损失替换原始损失
                loss = kd_total_loss
                
                # 如果有监督transformer，添加其监督损失
                if supervision_logits is not None:
                    supervision_loss_fct = nn.CrossEntropyLoss(
                        ignore_index=pad_token_id,
                        label_smoothing=label_smoothing
                    )
                    supervision_ce_loss = supervision_loss_fct(
                        supervision_logits.view(-1, self.config.vocab_size), 
                        valid_labels.view(-1)
                    )
                    loss_components['supervision_ce_loss'] = supervision_ce_loss.item()
                    
                    # 添加监督损失
                    loss = loss + 0.2 * supervision_ce_loss
                    
                    # 如果使用软标签，也对监督transformer应用KD损失
                    sup_kd_total_loss, sup_kd_losses = joint_kd_loss(
                        student_logits=supervision_logits,
                        teacher_logits=teacher_logits_adapted,
                        labels=valid_labels,
                        ignore_index=pad_token_id
                    )
                    # 更新损失组件
                    for k, v in sup_kd_losses.items():
                        loss_components[f'sup_kd_{k}'] = v
                    
                    # 添加监督KD损失
                    loss = loss + 0.2 * sup_kd_total_loss
        
        # 返回结果
        return {
            "loss": loss,
            "loss_components": loss_components,
            "ltc_ncp_logits": ltc_ncp_logits,
            "supervision_logits": supervision_logits,
        } if return_dict else (loss, ltc_ncp_logits)
        
    def generate(self, input_ids, attention_mask=None, max_length=None, eos_token_id=None, repetition_penalty=None):
        """生成文本"""
        config = self.config
        
        # 设置最大生成长度
        if max_length is None:
            max_length = config.max_position_embeddings - input_ids.shape[1] - 1
        
        # 使用配置中的EOS token ID
        if eos_token_id is None:
            eos_token_id = getattr(config, 'eos_token_id', None)
            
        # 设置重复惩罚
        if repetition_penalty is None:
            repetition_penalty = getattr(config, 'repetition_penalty', 1.0)
            
        # 编码输入 - 只计算一次
        encoder_outputs = self.encoder(input_ids, attention_mask)
        projected_encoder_outputs = self.encoder_to_decoder_projection(encoder_outputs)
        
        # 初始化序列
        batch_size = input_ids.shape[0]
        generated_ids = input_ids.clone()
        
        # 初始化解码器缓存
        self.ltc_ncp_decoder.reset_cache()
        
        # 初始次调用 - 处理整个序列
        logits = self.ltc_ncp_decoder(projected_encoder_outputs, use_cache=True, extend_cache=False)
        
        # 自回归生成
        for _ in range(max_length):
            # 取最后一个位置的预测
            next_token_logits = logits[:, -1, :]
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in generated_ids[i]:
                        if token_id in next_token_logits[i]:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            # 采样下一个token
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 将新token添加到生成序列中
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 检查是否所有样本都生成了EOS
            if eos_token_id is not None and (next_tokens == eos_token_id).all():
                break
            
            # 每次只处理新生成的token
            new_input_ids = next_tokens.unsqueeze(-1)
            new_encoder_outputs = self.encoder(new_input_ids, None)
            new_projected_outputs = self.encoder_to_decoder_projection(new_encoder_outputs)
            
            # 使用缓存进行解码
            logits = self.ltc_ncp_decoder(new_projected_outputs, use_cache=True, extend_cache=True)
            
        return generated_ids 

class TeacherOutputAdapter(nn.Module):
    """
    教师模型输出适配器，用于处理教师模型输出和学生模型输入之间的尺寸不匹配问题
    支持批次大小调整、序列长度调整和词汇表大小调整
    """
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, teacher_logits, target_batch_size, target_seq_length):
        """
        调整教师模型输出尺寸以匹配目标尺寸
        
        Args:
            teacher_logits: 教师模型输出 [batch_size, seq_length, vocab_size]
            target_batch_size: 目标批次大小
            target_seq_length: 目标序列长度
            
        Returns:
            adjusted_logits: 调整后的输出 [target_batch_size, target_seq_length, vocab_size]
        """
        if teacher_logits is None:
            # 如果没有教师输出，返回零张量
            return torch.zeros(
                (target_batch_size, target_seq_length, self.vocab_size),
                dtype=torch.float16,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        
        batch_size, seq_length, vocab_size = teacher_logits.shape
        device = teacher_logits.device
        dtype = teacher_logits.dtype
        
        # 处理词汇表大小不匹配
        # 如果教师模型的词汇表大小与目标不同，进行调整
        if vocab_size != self.vocab_size:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"正在调整教师词汇表大小: {vocab_size} -> {self.vocab_size}")
            
            # 创建新的张量来适配目标词汇表大小
            resized_logits = torch.zeros(
                (batch_size, seq_length, self.vocab_size),
                dtype=dtype,
                device=device
            )
            
            # 复制共同部分
            common_size = min(vocab_size, self.vocab_size)
            resized_logits[:, :, :common_size] = teacher_logits[:, :, :common_size]
            
            # 如果教师词汇表更大，对超出部分的信息进行汇总并加入到最后一个标记
            if vocab_size > self.vocab_size:
                # 计算超出部分的logits平均值，并添加到最后一个标记
                excess_logits = teacher_logits[:, :, common_size:].mean(dim=2, keepdim=True)
                # 将超出部分的信息添加到最后一个标记
                resized_logits[:, :, -1:] += excess_logits
            
            # 使用调整后的张量替换原始张量
            teacher_logits = resized_logits
            vocab_size = self.vocab_size
            
        # 处理批次大小不匹配
        if batch_size != target_batch_size:
            # 扩展或收缩批次维度
            if batch_size > target_batch_size:
                # 如果教师批次更大，裁剪
                teacher_logits = teacher_logits[:target_batch_size]
            else:
                # 如果教师批次更小，复制最后一个批次的样本
                padding = teacher_logits[-1:].expand(target_batch_size - batch_size, seq_length, vocab_size)
                teacher_logits = torch.cat([teacher_logits, padding], dim=0)
        
        # 处理序列长度不匹配
        if seq_length != target_seq_length:
            # 创建适应目标序列长度的新张量
            adjusted_logits = torch.zeros(
                (target_batch_size, target_seq_length, vocab_size),
                dtype=dtype,
                device=device
            )
            
            # 确定要复制的长度
            copy_length = min(seq_length, target_seq_length)
            
            # 复制共同部分
            adjusted_logits[:, :copy_length, :] = teacher_logits[:, :copy_length, :]
            
            # 如果需要扩展序列长度，复制最后的标记
            if target_seq_length > seq_length:
                last_tokens = teacher_logits[:, -1:, :]
                for i in range(seq_length, target_seq_length):
                    adjusted_logits[:, i:i+1, :] = last_tokens
            
            return adjusted_logits
        
        return teacher_logits

# 添加联合知识蒸馏损失函数
class JointKDLoss(nn.Module):
    """
    联合知识蒸馏损失函数，结合了多种蒸馏技术
    """
    def __init__(self, temperature=2.0, alpha=0.5, beta=0.5, gamma=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 软标签权重
        self.beta = beta    # 硬标签权重
        self.gamma = gamma  # 特征蒸馏权重
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        
    def forward(self, student_logits, teacher_logits, student_hidden=None, 
               teacher_hidden=None, labels=None, ignore_index=-100):
        """
        联合KD损失计算
        
        Args:
            student_logits: 学生模型输出logits
            teacher_logits: 教师模型输出logits
            student_hidden: 学生模型隐藏状态 (可选)
            teacher_hidden: 教师模型隐藏状态 (可选)
            labels: 真实标签 (可选)
            ignore_index: 忽略的标签索引
            
        Returns:
            total_loss: 联合KD损失
        """
        total_loss = 0.0
        losses = {}
        
        # 软标签蒸馏 (KL散度损失)
        if teacher_logits is not None:
            # 使用温度缩放logits
            soft_student = F.log_softmax(student_logits / self.temperature, dim=-1)
            soft_teacher = F.softmax(teacher_logits / self.temperature, dim=-1)
            
            # 计算KL散度
            soft_loss = self.kl_div(soft_student, soft_teacher) * (self.temperature ** 2)
            total_loss += self.alpha * soft_loss
            losses['soft_loss'] = soft_loss.item()
        
        # 硬标签监督 (交叉熵损失)
        if labels is not None:
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index
            )
            total_loss += self.beta * hard_loss
            losses['hard_loss'] = hard_loss.item()
        
        # 特征蒸馏 (MSE损失，可选)
        if student_hidden is not None and teacher_hidden is not None:
            # 确保维度匹配
            if student_hidden.size(-1) != teacher_hidden.size(-1):
                # 使用线性投影调整维度
                if not hasattr(self, 'hidden_projection'):
                    self.hidden_projection = nn.Linear(
                        student_hidden.size(-1), teacher_hidden.size(-1)
                    ).to(student_hidden.device)
                student_hidden = self.hidden_projection(student_hidden)
                
            feature_loss = F.mse_loss(student_hidden, teacher_hidden)
            total_loss += self.gamma * feature_loss
            losses['feature_loss'] = feature_loss.item()
        
        return total_loss, losses 
