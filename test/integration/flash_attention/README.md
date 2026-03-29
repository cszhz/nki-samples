# Flash Attention on AWS Trainium2

Neuron Kernel Interface (NKI) implementation of Flash Attention with Tensor Parallelism support.

## Environment Setup

```bash
# Activate Neuron environment
source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate

# Verify setup
neuron-ls
```

---

## Basic Testing

### 1. Correctness Test
```bash
pytest flash_attention_correctness.py
```

### 2. Simple Benchmark
```bash
# Fixed config: batch=1, heads=4, seq=32768
python flash_attention_benchmark.py result/metrics.json
cat result/metrics.json
```

### 3. All-in-One Script
```bash
# Runs correctness + benchmark
./run.sh result/metrics.json
```

---

## Tensor Parallelism Testing

### Single Test

```bash
# Basic syntax
torchrun --nproc_per_node=<TP> flash_attention_tp.py \
  --mode <MODE> --tp <TP> \
  [--batch-size <N>] [--seq-len <N>]

# Example: GQA with TP=4
torchrun --nproc_per_node=4 flash_attention_tp.py \
  --mode gqa --tp 4 --batch-size 8 --seq-len 8192
```

**Arguments:**
- `--mode`: `mha`, `gqa`, `mqa` (default: `gqa`)
- `--tp`: `1`, `2`, `4`, `8`, or `auto`
- `--batch-size`: Override config batch size
- `--seq-len`: Override config sequence length

**Attention Modes:**
- **MHA**: Q/K/V heads = 32/32/32 (standard Transformer)
- **GQA**: Q/K/V heads = 32/8/8 (Llama-style, 4:1 ratio)
- **MQA**: Q/K/V heads = 32/1/1 (inference optimized)

### Sweep Testing

```bash
# Quick test (6 configs)
python flash_attention_tp.py --sweep quick

# TP scalability (9 configs, tests TP=1/2/4)
python flash_attention_tp.py --sweep tp_scalability

# GQA tuning (12 configs, batch/seq sweep)
python flash_attention_tp.py --sweep gqa_performance

# Full matrix (72 configs: 3 modes × 3 TP × 4 batch × 2 seq)
python flash_attention_tp.py --sweep full
```

**View results:**
```bash
ls sweep_results/
cat sweep_results/sweep_summary.json
```

---

## Configuration

Edit `tp_configs.yaml` to customize:

```yaml
gqa:
  base:
    num_q_heads: 32
    num_k_heads: 8      # GQA: 4:1 ratio
    seq_len: 8192
    batch_size: 8
  tp4:
    q_heads_per_device: 8   # 32/4
    k_heads_per_device: 2   # 8/4
```

**Override via CLI:**
```bash
torchrun --nproc_per_node=4 flash_attention_tp.py \
  --mode gqa --tp 4 --batch-size 12 --seq-len 4096
```

---

## Common Issues

**1. Compilation error: instruction count exceeded**
```bash
# Reduce batch/seq or increase TP
--batch-size 8 --seq-len 8192  # Instead of larger values
```

**2. TP size mismatch**
```bash
# Make sure nproc_per_node matches --tp
torchrun --nproc_per_node=4 ... --tp 4
```

**3. First run takes 10-20 minutes**
- Normal: compilation happens once, subsequent runs use cache

**4. OOM error**
- Reduce batch size or increase TP size

---

## References

- `flash_attention.py`: Core NKI implementation
- `flash_attention_benchmark.py`: Simple fixed-config benchmark
- `flash_attention_tp.py`: Configurable TP testing with sweep support
- `tp_configs.yaml`: Configuration file

**AWS Neuron Docs:**
- [NKI Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)
- [Tensor Parallelism](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-training/app_notes/nxd-training-tp-appnote.html)
