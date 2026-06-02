# UltraCodec

> 面向 LLM 的超低帧率（< 5 Hz）语音编解码器

UltraCodec 是面向大语言模型（LLM）输入侧效率优化的神经音频编解码器。其核心目标是将语音 token 帧率压缩至 **3.125 Hz**（自适应平均），同时保持重构质量（ViSQOL > 3.5），从而显著降低 LLM 处理语音时的上下文占用、KV-cache 显存与推理时延。

## 核心特性

- **层次化时间压缩编码器（HTCE）**：50 → 25 → 12.5 → 6.25 → 3.125 Hz 四阶段渐进降采样。
- **语义预测量化（SPQ）**：基于 LM-prior 的残差量化，编码端去除可预测冗余。
- **自适应帧率门控（AFR）**：信息密度门控，静音段几乎零 token，关键段峰值至 12.5 Hz。
- **级联式解码器**：3.125 → 12.5 → 50 Hz → 16 kHz 多尺度教师强迫。
- **LLM 联合三阶段训练**：纯 codec 预训练 → LLM-aware 微调 → 联合 LoRA 微调。

## 安装

```bash
git clone <repo>
cd ultracodec
pip install -e .
```

依赖：PyTorch ≥ 2.0、torchaudio、librosa、omegaconf、hydra-core、pytorch-lightning。

完整依赖见 `requirements.txt`。

## 数据集准备

UltraCodec 支持以下数据集：

| 数据集 | 用途 | 自动下载 |
| ------ | ---- | -------- |
| LibriSpeech | 训练 / 测试 | ✓ |
| VCTK | 多说话人评估 | (用户已有) |
| MUSAN | 噪声增强 | ✓ |
| DNS Challenge | 噪声鲁棒性 | ✓ |

**VCTK 路径**：`/data01/audio_group/m24_yuanjiajun/AP-BWE/VCTK-Corpus-0.92/wav_test`，子目录为 `train/`、`train_test/`、`test/`。

一键下载所有非 VCTK 数据集：

```bash
bash scripts/download_data.sh
# 或
python -m ultracodec.data.download --output_dir ./data --datasets librispeech,musan,dns
```

## 快速开始

### Stage 1：纯 codec 预训练

```bash
python scripts/train.py --config configs/train_stage1.yaml
```

### Stage 2：LLM-aware 微调

```bash
python scripts/train.py --config configs/train_stage2.yaml \
    --resume runs/uc_stage1/best.pt
```

### Stage 3：联合 LoRA 微调

```bash
python scripts/train.py --config configs/train_stage3.yaml \
    --resume runs/uc_stage2/best.pt
```

### 评估

```bash
python scripts/evaluate.py --config configs/eval.yaml \
    --checkpoint runs/uc_stage3/best.pt
```

### 编解码演示

```bash
python scripts/encode_decode.py --checkpoint runs/uc_stage3/best.pt \
    --input sample.wav --output sample_recon.wav
```

## 项目结构

```
ultracodec/
├── configs/             # YAML 配置（base + 三阶段 + eval）
├── ultracodec/
│   ├── model/           # 模型实现（后续填充）
│   ├── data/            # 数据集、下载、预处理
│   ├── losses/          # 损失函数
│   ├── metrics/         # 评估指标
│   └── utils/           # 工具（日志、checkpoint、EMA 等）
├── scripts/             # 训练 / 评估 / 演示脚本
└── tests/               # 单元测试
```

## 开发状态

- [x] 项目骨架与数据管线
- [ ] HTCE / SPQ / AFR / Cascaded Decoder 实现
- [ ] 损失函数（多尺度 STFT、对抗、蒸馏）
- [ ] 三阶段训练循环
- [ ] 评估指标（ViSQOL / UTMOS / PESQ / WER）

## 引用

参见 `/Users/judge/Documents/auto/auto_audio_speech/papers/paper3_ultracodec.md`。

## License

Apache-2.0
