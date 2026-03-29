#!/usr/bin/env python3
"""
Flash Attention TP with Configurable TP Size
支持 TP=1/2/4/8 等多种配置

运行方式:
  # 使用配置文件
  torchrun --nproc_per_node=4 flash_attention_tp_configurable.py --mode gqa --tp 4
  torchrun --nproc_per_node=8 flash_attention_tp_configurable.py --mode gqa --tp 8

  # 自动检测设备数
  torchrun --nproc_per_node=4 flash_attention_tp_configurable.py --mode gqa --tp auto
"""
import torch
import os
import sys
import time
import argparse
import yaml
import json
import subprocess
from pathlib import Path
from datetime import datetime

os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer --distribution-strategy=llm-training"
os.environ["NEURON_FUSE_SOFTMAX"] = "1"

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch.distributed as dist
from flash_attention import nki_flash_attn_func


def load_config(config_file="tp_configs.yaml"):
    """加载配置文件"""
    config_path = Path(__file__).parent / config_file
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_tp_config(config, mode, tp_size):
    """获取指定模式和 TP 大小的配置"""
    mode_config = config[mode]
    tp_key = f"tp{tp_size}"

    if tp_key not in mode_config:
        raise ValueError(f"TP size {tp_size} not configured for mode {mode}")

    tp_config = mode_config[tp_key].copy()
    tp_config.update(mode_config['base'])

    return tp_config


def run_flash_attention_tp(rank, world_size, device, mode, tp_config):
    """运行 Flash Attention TP"""

    if rank == 0:
        print("\n" + "="*80)
        print(f"Flash Attention Tensor Parallelism - {tp_config.get('name', mode.upper())}")
        print("="*80)
        print(f"Configuration:")
        print(f"  Mode: {mode.upper()}")
        print(f"  TP Size: {world_size}")
        print(f"  Total Q heads: {tp_config['num_q_heads']}")
        print(f"  Total K/V heads: {tp_config['num_k_heads']}")
        print(f"  Q heads per device: {tp_config['q_heads_per_device']}")
        print(f"  K/V heads per device: {tp_config['k_heads_per_device']}")
        print(f"  Sequence length: {tp_config['seq_len']}")
        print(f"  Head dimension: {tp_config['head_dim']}")
        print(f"  Batch size: {tp_config['batch_size']}")
        print("="*80)

    torch.manual_seed(42)
    dtype_str = tp_config.get('dtype', 'bfloat16')
    dtype = getattr(torch, dtype_str)

    bs = tp_config['batch_size']
    num_q_heads_total = tp_config['num_q_heads']
    num_kv_heads_total = tp_config['num_k_heads']
    num_q_heads_per_device = tp_config['q_heads_per_device']
    num_kv_heads_per_device = tp_config['k_heads_per_device']
    head_dim = tp_config['head_dim']
    seq_len = tp_config['seq_len']
    kv_replicated = tp_config.get('kv_replicated', False)

    print(f"\n[Rank {rank}/{world_size}] Initializing...")
    print(f"[Rank {rank}] Device: {device}")
    print(f"[Rank {rank}] Processing - Q: {num_q_heads_per_device} heads, "
          f"K/V: {num_kv_heads_per_device} heads")

    # 生成完整输入
    q_full = torch.randn(bs, num_q_heads_total, seq_len, head_dim, dtype=dtype) - 0.5
    k_full = torch.randn(bs, num_kv_heads_total, seq_len, head_dim, dtype=dtype) - 0.5
    v_full = torch.randn(bs, num_kv_heads_total, seq_len, head_dim, dtype=dtype) - 0.5

    # 切分策略
    if kv_replicated:
        # MQA: K/V 复制
        q_head_start = rank * num_q_heads_per_device
        q_head_end = q_head_start + num_q_heads_per_device

        q_local = q_full[:, q_head_start:q_head_end, :, :].contiguous().to(device).requires_grad_()
        k_local = k_full.to(device).requires_grad_()  # 完整复制
        v_local = v_full.to(device).requires_grad_()  # 完整复制

        print(f"[Rank {rank}] MQA mode: K/V replicated across devices")
    else:
        # MHA/GQA: 按比例切分
        q_head_start = rank * num_q_heads_per_device
        q_head_end = q_head_start + num_q_heads_per_device

        kv_head_start = rank * num_kv_heads_per_device
        kv_head_end = kv_head_start + num_kv_heads_per_device

        q_local = q_full[:, q_head_start:q_head_end, :, :].contiguous().to(device).requires_grad_()
        k_local = k_full[:, kv_head_start:kv_head_end, :, :].contiguous().to(device).requires_grad_()
        v_local = v_full[:, kv_head_start:kv_head_end, :, :].contiguous().to(device).requires_grad_()

    print(f"[Rank {rank}] Local shapes: Q={q_local.shape}, K={k_local.shape}, V={v_local.shape}")

    # Warmup
    if rank == 0:
        print("\n[Rank 0] Warming up (compiling)...")
    _ = nki_flash_attn_func(q_local, k_local, v_local, causal=True)
    xm.mark_step()

    # Benchmark
    if rank == 0:
        print("[Rank 0] Running benchmark...")

    start = time.time()

    # Forward
    out_local = nki_flash_attn_func(q_local, k_local, v_local, causal=True)

    # Backward
    loss_local = torch.sum(out_local**2)
    loss_local.backward()
    xm.mark_step()

    # All-reduce loss
    if world_size > 1:
        loss_global = xm.all_reduce(xm.REDUCE_SUM, loss_local)

        # MQA: 需要 all-reduce K/V 梯度
        if kv_replicated:
            k_grad = xm.all_reduce(xm.REDUCE_SUM, k_local.grad)
            v_grad = xm.all_reduce(xm.REDUCE_SUM, v_local.grad)
            xm.mark_step()
    else:
        loss_global = loss_local

    elapsed = time.time() - start

    print(f"\n[Rank {rank}] Results:")
    print(f"  Output shape: {out_local.shape}")
    print(f"  Local loss: {loss_local.item():.2f}")
    print(f"  Global loss: {loss_global.item():.2f}")
    print(f"  Time: {elapsed:.3f}s")

    # Gather outputs
    if world_size > 1:
        print(f"[Rank {rank}] Gathering outputs...")
        out_gathered = xm.all_gather(out_local, dim=1)
        xm.mark_step()
        print(f"[Rank {rank}] Gathered shape: {out_gathered.shape}")
    else:
        out_gathered = out_local

    if rank == 0:
        print("\n" + "="*80)
        print("✓ Test Completed!")
        print("="*80)
        print(f"  Mode: {mode.upper()}")
        print(f"  TP Size: {world_size}")
        print(f"  Q heads: {num_q_heads_total} ({num_q_heads_per_device}/device)")
        print(f"  K/V heads: {num_kv_heads_total} ({num_kv_heads_per_device}/device)")
        if kv_replicated:
            print(f"  K/V strategy: Replicated")
        print(f"  Sequence length: {seq_len}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Global loss: {loss_global.item():.2f}")
        print(f"  Output shape: {out_gathered.shape}")
        print("="*80)

    return {
        'mode': mode,
        'tp_size': world_size,
        'time': elapsed,
        'loss': loss_global.item(),
        'output_shape': out_gathered.shape
    }


def detect_hardware():
    """检测硬件配置"""
    try:
        import subprocess
        result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
        output = result.stdout

        # 简单解析 neuron-ls 输出
        if 'trn2.xlarge' in output or 'NEURON CORES' in output:
            # 计算 NeuronCore 数量
            lines = output.split('\n')
            for line in lines:
                if 'NEURON' in line and 'CORES' in line:
                    parts = line.split('|')
                    if len(parts) > 2:
                        cores = parts[2].strip()
                        if cores.isdigit():
                            return int(cores)

        # 回退：通过 XLA 设备检测
        devices = xm.get_xla_supported_devices()
        return len(devices)
    except:
        # 最后回退
        return 1


def run_sweep_tests(sweep_name, config_file='tp_configs.yaml', output_dir='sweep_results'):
    """Run batch tests based on sweep configurations"""
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if 'sweep_configs' not in config:
        print("Error: No sweep_configs found in config file")
        return

    sweep_cfg = config['sweep_configs']

    # Get test list based on sweep_name
    if sweep_name == 'full':
        # Run all 72 combinations
        test_configs = []
        for mode_key in ['mha_combinations', 'gqa_combinations', 'mqa_combinations']:
            if mode_key in sweep_cfg:
                test_configs.extend(sweep_cfg[mode_key])
    elif sweep_name in ['quick', 'tp_scalability', 'gqa_performance']:
        # Use predefined subset
        if 'recommended_subsets' not in sweep_cfg or sweep_name not in sweep_cfg['recommended_subsets']:
            print(f"Error: Sweep '{sweep_name}' not found in recommended_subsets")
            return
        subset = sweep_cfg['recommended_subsets'][sweep_name]
        test_labels = subset['tests']

        # Find configs by label
        test_configs = []
        for mode_key in ['mha_combinations', 'gqa_combinations', 'mqa_combinations']:
            if mode_key in sweep_cfg:
                for cfg in sweep_cfg[mode_key]:
                    if cfg['label'] in test_labels:
                        test_configs.append(cfg)
    else:
        print(f"Error: Unknown sweep name '{sweep_name}'")
        print(f"Available: full, quick, tp_scalability, gqa_performance")
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Results storage
    results = {
        'sweep_name': sweep_name,
        'start_time': datetime.now().isoformat(),
        'total_tests': len(test_configs),
        'tests': []
    }

    print(f"\n{'='*80}")
    print(f"Starting Sweep: {sweep_name}")
    print(f"Total tests: {len(test_configs)}")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}\n")

    # Run each test
    for idx, test_cfg in enumerate(test_configs, 1):
        mode = test_cfg['mode']
        tp = test_cfg['tp']
        batch = test_cfg['batch']
        seq = test_cfg['seq']
        label = test_cfg['label']

        print(f"\n[{idx}/{len(test_configs)}] Running: {label}")
        print(f"  Mode: {mode}, TP: {tp}, Batch: {batch}, Seq: {seq}")

        # Build command
        cmd = [
            'torchrun',
            f'--nproc_per_node={tp}',
            'flash_attention_tp.py',
            '--mode', mode,
            '--tp', str(tp),
            '--batch-size', str(batch),
            '--seq-len', str(seq)
        ]

        # Run test
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            elapsed = time.time() - start_time

            # Parse output for timing and metrics
            test_result = {
                'label': label,
                'mode': mode,
                'tp': tp,
                'batch': batch,
                'seq': seq,
                'elapsed_time': elapsed,
                'success': result.returncode == 0,
                'stdout': result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,  # Last 1000 chars
                'stderr': result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
            }

            # Extract time from output
            for line in result.stdout.split('\n'):
                if 'Time:' in line and 's' in line:
                    try:
                        time_str = line.split('Time:')[1].strip().replace('s', '')
                        test_result['execution_time'] = float(time_str)
                    except:
                        pass

            results['tests'].append(test_result)

            if result.returncode == 0:
                print(f"  ✓ Completed in {elapsed:.1f}s")
            else:
                print(f"  ✗ Failed (exit code: {result.returncode})")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"  ✗ Timeout after {elapsed:.1f}s")
            results['tests'].append({
                'label': label,
                'mode': mode,
                'tp': tp,
                'batch': batch,
                'seq': seq,
                'elapsed_time': elapsed,
                'success': False,
                'error': 'Timeout (>1800s)'
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results['tests'].append({
                'label': label,
                'mode': mode,
                'tp': tp,
                'batch': batch,
                'seq': seq,
                'success': False,
                'error': str(e)
            })

    # Save results
    results['end_time'] = datetime.now().isoformat()
    results['success_count'] = sum(1 for t in results['tests'] if t['success'])
    results['fail_count'] = len(results['tests']) - results['success_count']

    output_file = output_path / f"{sweep_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Sweep Completed: {sweep_name}")
    print(f"{'='*80}")
    print(f"Total tests: {len(results['tests'])}")
    print(f"Successful: {results['success_count']}")
    print(f"Failed: {results['fail_count']}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Flash Attention Tensor Parallelism Test',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test GQA with TP=4 (recommended)
  torchrun --nproc_per_node=4 flash_attention_tp.py --mode gqa --tp 4

  # Test MHA with auto-detected TP
  torchrun --nproc_per_node=4 flash_attention_tp.py --mode mha --tp auto

  # Test with custom sequence length
  torchrun --nproc_per_node=2 flash_attention_tp.py --mode gqa --tp 2 --seq-len 4096

  # Test TP scalability (run separately)
  torchrun --nproc_per_node=1 flash_attention_tp.py --mode gqa --tp 1
  torchrun --nproc_per_node=2 flash_attention_tp.py --mode gqa --tp 2
  torchrun --nproc_per_node=4 flash_attention_tp.py --mode gqa --tp 4

  # Batch test all 72 combinations
  python flash_attention_tp.py --sweep full

  # Run quick validation sweep (6 tests)
  python flash_attention_tp.py --sweep quick

  # Run TP scalability sweep (9 tests)
  python flash_attention_tp.py --sweep tp_scalability

  # Run GQA performance tuning sweep (12 tests)
  python flash_attention_tp.py --sweep gqa_performance

Configuration:
  Configs are loaded from tp_configs.yaml
  72 combinations available: 3 modes × 3 TP sizes × 4 batch sizes × 2 seq lengths

  Modes:
    mha: Multi-Head Attention (Q/K/V heads equal, good for training)
    gqa: Grouped-Query Attention (K/V heads < Q heads, recommended)
    mqa: Multi-Query Attention (K/V heads = 1, good for inference)

  TP Sizes:
    1: Single device baseline
    2: 2-way tensor parallelism
    4: 4-way tensor parallelism (recommended for trn2.3xlarge)

Note:
  --nproc_per_node must match --tp size for proper distributed execution
        """)
    parser.add_argument('--mode', type=str, default='gqa',
                       choices=['mha', 'gqa', 'mqa'],
                       help='Attention mode: mha (Multi-Head), gqa (Grouped-Query), mqa (Multi-Query)')
    parser.add_argument('--tp', type=str, default='auto',
                       help='TP size: 1, 2, 4, or "auto" to auto-detect from world_size')
    parser.add_argument('--config', type=str, default='tp_configs.yaml',
                       help='Path to YAML config file (default: tp_configs.yaml)')
    parser.add_argument('--seq-len', type=int, default=None,
                       help='Override sequence length from config (e.g., 4096, 8192)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config (e.g., 4, 8, 12, 16)')
    parser.add_argument('--sweep', type=str, default=None,
                       choices=['full', 'quick', 'tp_scalability', 'gqa_performance'],
                       help='Run batch tests: full (72), quick (6), tp_scalability (9), gqa_performance (12)')
    parser.add_argument('--output-dir', type=str, default='sweep_results',
                       help='Output directory for sweep results (default: sweep_results)')
    args = parser.parse_args()

    # Handle sweep mode (batch testing)
    if args.sweep:
        run_sweep_tests(args.sweep, args.config, args.output_dir)
        return

    # 初始化分布式
    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group('xla')
        world_size = xr.world_size()
        rank = xr.global_ordinal()
    else:
        world_size = 1
        rank = 0

    device = xm.xla_device()

    # 处理 TP size
    if args.tp == 'auto':
        tp_size = world_size
        if rank == 0:
            print(f"Auto-detected TP size: {tp_size}")
    else:
        tp_size = int(args.tp)
        if tp_size != world_size:
            if rank == 0:
                print(f"⚠ Warning: Requested TP={tp_size} but world_size={world_size}")
                print(f"Using world_size={world_size}")
            tp_size = world_size

    # 加载配置
    try:
        config = load_config(args.config)
        tp_config = get_tp_config(config, args.mode, tp_size)

        # 覆盖 seq_len (如果指定)
        if args.seq_len:
            tp_config['seq_len'] = args.seq_len
            if rank == 0:
                print(f"Overriding seq_len to {args.seq_len}")

        # 覆盖 batch_size (如果指定)
        if args.batch_size:
            tp_config['batch_size'] = args.batch_size
            if rank == 0:
                print(f"Overriding batch_size to {args.batch_size}")

    except Exception as e:
        if rank == 0:
            print(f"❌ Error loading config: {e}")
            print(f"Make sure {args.config} exists and is valid")
        sys.exit(1)

    # 检查硬件兼容性
    if rank == 0:
        available_cores = detect_hardware()
        print(f"\nHardware Detection:")
        print(f"  Available NeuronCores: {available_cores}")
        print(f"  Requested TP size: {tp_size}")

        if tp_size > available_cores:
            print(f"\n⚠ WARNING:")
            print(f"  Requested TP={tp_size} but only {available_cores} cores available")
            print(f"  This test will still run but may not achieve expected parallelism")
            print(f"  Recommended: Use TP={available_cores} or smaller")
        print()

    # 运行测试
    result = run_flash_attention_tp(rank, world_size, device, args.mode, tp_config)

    # 性能分析 (仅 rank 0)
    if rank == 0:
        print("\n" + "="*80)
        print("Performance Analysis")
        print("="*80)

        # 估算理论加速比
        baseline_time = result['time'] * tp_size  # 估算单设备时间
        actual_speedup = baseline_time / result['time']
        efficiency = (actual_speedup / tp_size) * 100

        print(f"  Estimated single-device time: {baseline_time:.2f}s")
        print(f"  Actual time (TP={tp_size}): {result['time']:.2f}s")
        print(f"  Speedup: {actual_speedup:.2f}x")
        print(f"  Parallel efficiency: {efficiency:.1f}%")

        # 内存节省
        if args.mode == 'mha':
            memory_savings = (1 - 1/tp_size) * 100
            print(f"  Memory savings per device: ~{memory_savings:.1f}%")
        elif args.mode == 'gqa':
            memory_savings = (1 - 1/tp_size) * 100
            print(f"  Memory savings per device: ~{memory_savings:.1f}% (Q and K/V)")
        elif args.mode == 'mqa':
            q_savings = (1 - 1/tp_size) * 100
            print(f"  Memory savings per device: ~{q_savings:.1f}% (Q only)")
            print(f"  Note: K/V replicated but small (1 head)")

        print("="*80)


if __name__ == '__main__':
    main()
