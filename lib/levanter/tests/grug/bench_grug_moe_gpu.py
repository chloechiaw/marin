#!/usr/bin/env python
"""Benchmark grug MoE on GPU comparing ragged_dot backends.

Usage:
    # Full Grug Transformer (matches reviewer's PR 4297 conditions):
    RAGGED_DOT_IMPL=xla uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --mode model --grad --iters 100
    RAGGED_DOT_IMPL=triton uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --mode model --grad --iters 100

    # Isolated MoE FFN forward:
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl xla
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl triton

    # Forward + backward (MoE FFN only):
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl xla --grad
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl triton --grad

    # Isolated ragged_dot kernel:
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl xla --mode kernel
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl triton --mode kernel

    # Skewed routing (80% of tokens to 2 experts — shows Triton advantage):
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl xla --mode kernel --skewed
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --impl triton --mode kernel --skewed

    # Qwen3-30B-A3B geometry (128 experts, topk=8, ep=8) — compare vs Megatron ~12.4ms:
    # tokens=32768 = 4096 tokens/rank * 8 ranks, matching Megatron's per-rank input
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --mode moe_mlp --grad \
        --hidden 2048 --intermediate 768 --experts 128 --topk 8 \
        --tokens 32768 --ep-size 8 --warmup 5 --iters 20

    # Qwen3-235B-A22B geometry — compare vs Megatron ~11.75ms:
    uv run python lib/levanter/tests/grug/bench_grug_moe_gpu.py --mode moe_mlp --grad \
        --hidden 4096 --intermediate 1536 --experts 128 --topk 8 \
        --tokens 32768 --ep-size 8 --warmup 5 --iters 20
"""

import argparse
import contextlib
import os
import statistics
import time

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, AxisType, NamedSharding, PartitionSpec as P

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


# H100 SXM peak bf16 Tensor Core throughput (spec sheet).
H100_BF16_PEAK_TFLOPS = 989.0


def make_inputs(*, tokens, hidden_dim, intermediate_dim, num_experts, topk, skewed, dtype=jnp.bfloat16):
    key = jax.random.key(42)
    k_x, k_sel, k_logits, k_w13, k_w2 = jax.random.split(key, 5)

    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=dtype)

    if skewed:
        # 80% of tokens routed to experts 0-1, 20% spread across 2-7.
        # This creates large imbalance — XLA pads all groups to the max.
        probs = jnp.array([0.4, 0.4] + [0.2 / (num_experts - 2)] * (num_experts - 2))
        selected_experts = jax.random.choice(
            k_sel, num_experts, shape=(tokens, topk), p=probs
        ).astype(jnp.int32)
    else:
        selected_experts = jax.random.randint(k_sel, (tokens, topk), 0, num_experts, dtype=jnp.int32)

    combine_logits = jax.random.normal(k_logits, (tokens, topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(dtype)
    w_up_gate = jax.random.normal(k_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=dtype)
    w_down = jax.random.normal(k_w2, (num_experts, intermediate_dim, hidden_dim), dtype=dtype)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def compute_moe_flops(tokens, topk, hidden_dim, intermediate_dim):
    """Compute FLOPs for one MoE FFN forward pass (up+gate matmul + down matmul).

    Each token is routed to `topk` experts. Per expert assignment:
      - up+gate: 2 * hidden * (2*intermediate) FLOPs  (one ragged_dot)
      - down:    2 * intermediate * hidden FLOPs       (one ragged_dot)
    """
    assignments = tokens * topk
    flops_up_gate = 2 * assignments * hidden_dim * (2 * intermediate_dim)
    flops_down = 2 * assignments * intermediate_dim * hidden_dim
    return flops_up_gate + flops_down


def bench_model(args):
    """Benchmark full Grug Transformer training step — matches reviewer's PR 4297 setup.

    Includes forward pass, backward pass, and optimizer update (AdamH),
    matching the real training loop in experiments/grug/moe/train.py.
    """
    import dataclasses as dc

    import jmp
    import optax

    from experiments.grug.moe.model import GrugModelConfig, Transformer
    from experiments.grug.moe.optimizer import GrugMoeAdamHConfig

    num_devices = jax.device_count()
    device_kind = jax.devices()[0].device_kind

    config = GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=512,
        intermediate_dim=1024,
        shared_expert_intermediate_dim=512,
        dense_intermediate_dim=1536,
        num_experts=8,
        num_experts_per_token=2,
        num_layers=10,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=4096,
        head_dim=None,
        initializer_std=0.5 / 512**0.5,
        qk_mult=1.3,
    )

    # Reviewer's optimizer config from launch_h100_pr4297.py.
    opt_config = GrugMoeAdamHConfig(
        learning_rate=0.003,
        adam_lr=0.003,
        beta1=0.96,
        beta2=0.995,
        epsilon=1e-15,
        lr_schedule="linear",
        decay=0.2,
        min_lr_ratio=0.0,
        warmup=0.1,
        max_grad_norm=1,
    )

    # Reviewer's mixed precision: params=float32, compute=bfloat16, output=bfloat16.
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.bfloat16)

    num_train_steps = args.warmup + args.iters
    optimizer = opt_config.build(num_train_steps)

    batch_size = args.batch_size
    seq_len = config.max_seq_len
    total_tokens = batch_size * seq_len
    impl = os.environ.get("RAGGED_DOT_IMPL", "auto")
    z_loss_weight = 1e-4

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {num_devices}x {device_kind}")
    print(f"Model:   Grug MoE ~256M (10 layers, hidden=512, 8 experts, topk=2)")
    print(f"Batch:   {batch_size} x seq_len={seq_len} = {total_tokens} tokens")
    print(f"Optim:   AdamH (lr=0.003, beta1=0.96, beta2=0.995)")
    print(f"MP:      params=f32, compute=bf16, output=bf16")
    print(f"RAGGED_DOT_IMPL: {impl}")
    print(f"Timing:  {args.warmup} warmup + {args.iters} timed iterations")
    print()

    # 3-axis mesh: (data, expert, model) — matches training setup.
    expert_parallel = min(2, num_devices)
    data_parallel = num_devices // expert_parallel
    devices = jax.devices()
    mesh = Mesh(
        np.array(devices).reshape(data_parallel, expert_parallel, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )

    print(f"Mesh:    data={data_parallel}, expert={expert_parallel}, model=1")
    print()

    # Initialize model + optimizer state (matches initial_state in train.py).
    print("Initializing model + optimizer state...")
    with jax.set_mesh(mesh):
        params = mp.cast_to_param(Transformer.init(config, key=jax.random.PRNGKey(42)))
        opt_state = optimizer.init(params)

    step_counter = jnp.array(0, dtype=jnp.int32)
    one = jnp.array(1, dtype=jnp.int32)

    # Create fake batch — shard across (data, expert) axes.
    token_ids = jax.random.randint(
        jax.random.PRNGKey(0), (batch_size, seq_len), 0, config.vocab_size, dtype=jnp.int32
    )
    loss_weight = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    token_ids = jax.device_put(token_ids, batch_sharding)
    loss_weight = jax.device_put(loss_weight, batch_sharding)

    z_loss = z_loss_weight if z_loss_weight > 0 else None

    @jax.jit
    def train_step(params, opt_state, step_num, toks, lw):
        def loss_fn(p):
            compute_params = mp.cast_to_compute(p)
            return compute_params.next_token_loss(
                toks, lw, reduction="mean",
                logsumexp_weight=z_loss,
                return_router_metrics=True,
            )
        (loss, _metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, step_num + one, loss

    # Warmup
    print("Warming up (includes JIT compilation)...")
    with jax.set_mesh(mesh):
        for _ in range(args.warmup):
            params, opt_state, step_counter, loss = train_step(
                params, opt_state, step_counter, token_ids, loss_weight
            )
            jax.block_until_ready(loss)
    print(f"  Loss after warmup: {float(jax.device_get(loss)):.4f}")
    print()

    # Timed
    times = []
    with jax.set_mesh(mesh):
        for _ in range(args.iters):
            t0 = time.perf_counter()
            params, opt_state, step_counter, loss = train_step(
                params, opt_state, step_counter, token_ids, loss_weight
            )
            jax.block_until_ready(loss)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times_ms = [t * 1000 for t in times]
    median_ms = statistics.median(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    examples_per_sec = batch_size / (median_ms / 1000)
    tokens_per_sec = total_tokens / (median_ms / 1000)

    print(f"Results ({args.iters} iterations, RAGGED_DOT_IMPL={impl}, mode=model):")
    print(f"  Median:       {median_ms:.2f} ms")
    print(f"  Mean:         {mean_ms:.2f} ms")
    print(f"  Min:          {min_ms:.2f} ms")
    print(f"  Max:          {max_ms:.2f} ms")
    print(f"  Examples/sec: {examples_per_sec:.2f}")
    print(f"  Tokens/sec:   {tokens_per_sec:,.0f}")


def bench(args):
    impl = args.impl
    num_devices = jax.device_count()
    device_kind = jax.devices()[0].device_kind
    ep_size = args.ep_size

    if ep_size > num_devices:
        raise ValueError(f"ep_size={ep_size} exceeds device count={num_devices}")
    if num_devices % ep_size != 0:
        raise ValueError(f"device count={num_devices} not divisible by ep_size={ep_size}")
    if ep_size > 1 and args.experts % ep_size != 0:
        raise ValueError(f"experts={args.experts} not divisible by ep_size={ep_size}")

    data_parallel = num_devices // ep_size

    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {num_devices}x {device_kind}")
    print(f"Config:  tokens={args.tokens}, hidden={args.hidden}, intermediate={args.intermediate}, "
          f"experts={args.experts}, topk={args.topk}, dtype=bf16")
    print(f"Implementation: {impl}")
    if ep_size > 1:
        print(f"EP:      ep_size={ep_size}, data_parallel={data_parallel}, "
              f"moe_impl={args.moe_impl}")
    print(f"Routing: {'skewed (80% to 2 experts)' if args.skewed else 'uniform random'}")
    print(f"Pass:    {'forward + backward (jax.grad)' if args.grad else 'forward only'}")
    print(f"Timing:  {args.warmup} warmup + {args.iters} timed iterations")
    print()

    devices = jax.devices()
    if ep_size > 1:
        mesh = Mesh(
            np.array(devices).reshape(data_parallel, ep_size, 1),
            axis_names=("data", "expert", "model"),
            axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        )
        print(f"Mesh:    data={data_parallel}, expert={ep_size}, model=1")
    else:
        mesh = Mesh(
            devices,
            axis_names=("data",),
            axis_types=(AxisType.Explicit,),
        )
        print(f"Mesh:    data={num_devices}")
    print()

    x, selected_experts, combine_weights, w_up_gate, w_down = make_inputs(
        tokens=args.tokens,
        hidden_dim=args.hidden,
        intermediate_dim=args.intermediate,
        num_experts=args.experts,
        topk=args.topk,
        skewed=args.skewed,
    )

    if ep_size > 1:
        # With EP: shard tokens across (data, expert) and weights across expert.
        batch_sharding = NamedSharding(mesh, P(("data", "expert")))
        batch_topk_sharding = NamedSharding(mesh, P(("data", "expert"), None))
        expert_sharding = NamedSharding(mesh, P("expert", None, None))
    else:
        # No EP: shard tokens across data, replicate weights.
        batch_sharding = NamedSharding(mesh, P("data"))
        batch_topk_sharding = NamedSharding(mesh, P("data", None))
        expert_sharding = NamedSharding(mesh, P())

    x = jax.device_put(x, batch_sharding)
    selected_experts = jax.device_put(selected_experts, batch_topk_sharding)
    combine_weights = jax.device_put(combine_weights, batch_topk_sharding)
    w_up_gate = jax.device_put(w_up_gate, expert_sharding)
    w_down = jax.device_put(w_down, expert_sharding)

    if args.mode == "moe_mlp":
        moe_impl = args.moe_impl if ep_size > 1 else None
        if args.grad:
            @jax.jit
            def step(x, sel, cw, w13, w2):
                def loss_fn(x_, w13_, w2_):
                    out = moe_mlp(
                        x_, sel, cw, w13_, w2_,
                        activation=ActivationFunctionEnum.silu,
                        mesh=mesh,
                        implementation=moe_impl,
                    )
                    return out.sum()
                grads = jax.grad(loss_fn, argnums=(0, 1, 2))(x, w13, w2)
                return grads[0]  # return one grad to block on
        else:
            @jax.jit
            def step(x, sel, cw, w13, w2):
                return moe_mlp(
                    x, sel, cw, w13, w2,
                    activation=ActivationFunctionEnum.silu,
                    mesh=mesh,
                    implementation=moe_impl,
                )

        run = lambda: step(x, selected_experts, combine_weights, w_up_gate, w_down)
    else:
        if ep_size > 1:
            raise ValueError("kernel mode does not support expert parallelism; use --mode moe_mlp")
        # Isolate just the ragged_dot call for a cleaner kernel-level comparison.
        from levanter.grug.grug_moe import _prepare_moe_dispatch
        x_dispatch, _, _, group_sizes = _prepare_moe_dispatch(
            x, selected_experts, combine_weights, num_experts=args.experts,
        )

        # Show actual group sizes to illustrate skew.
        gs_vals = list(map(int, group_sizes))
        print(f"Group sizes: {gs_vals}  (max={max(gs_vals)}, min={min(gs_vals)}, "
              f"ratio={max(gs_vals)/max(min(gs_vals),1):.1f}x)")
        print()

        @jax.jit
        def step_kernel(x_d, w13, gs):
            return ragged_dot(x_d, w13, gs, implementation=impl)

        run = lambda: step_kernel(x_dispatch, w_up_gate, group_sizes)

    # Warmup (includes compilation)
    # kernel mode runs raw ragged_dot without shard_map, so skip the mesh
    # to avoid ShardingTypeError from ragged_dot_general.
    use_mesh = args.mode == "moe_mlp"
    mesh_ctx = jax.set_mesh(mesh) if use_mesh else contextlib.nullcontext()

    print("Warming up (includes JIT compilation)...")
    with mesh_ctx:
        for _ in range(args.warmup):
            out = run()
            out.block_until_ready()
    print(f"  Output shape: {out.shape}, dtype: {out.dtype}")
    print(f"  finite: {bool(jnp.isfinite(out).all())}")
    print()

    # Timed — run one extra "drain" iteration to absorb any deferred work from
    # the warmup→measurement transition, then measure args.iters clean iterations.
    times = []
    with mesh_ctx:
        # Drain iteration (not counted)
        out = run()
        out.block_until_ready()

        for i in range(args.iters):
            t0 = time.perf_counter()
            out = run()
            out.block_until_ready()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            times.append(t1 - t0)
            print(f"  iter {i:3d}: {elapsed_ms:10.2f} ms")

    times_ms = [t * 1000 for t in times]
    median_ms = statistics.median(times_ms)
    mean_ms = sum(times_ms) / len(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    tokens_per_sec = args.tokens / (median_ms / 1000)

    # MoE FFN params (up_gate + down weights only)
    param_count = args.experts * (args.hidden * 2 * args.intermediate + args.intermediate * args.hidden)
    param_count_m = param_count / 1e6

    # MFU: model FLOPs utilization (backward ≈ 2x forward, so fwd+bwd ≈ 3x forward)
    total_flops = compute_moe_flops(args.tokens, args.topk, args.hidden, args.intermediate)
    if args.grad:
        total_flops *= 3
    achieved_tflops = (total_flops / (median_ms / 1000)) / 1e12
    peak_tflops = H100_BF16_PEAK_TFLOPS * num_devices
    mfu = achieved_tflops / peak_tflops * 100

    grad_tag = "+grad" if args.grad else ""
    ep_tag = f", ep={ep_size}" if ep_size > 1 else ""
    print(f"Results ({args.iters} iterations, impl={impl}, mode={args.mode}{grad_tag}{ep_tag}):")
    print(f"  Median: {median_ms:.2f} ms")
    print(f"  Mean:   {mean_ms:.2f} ms")
    print(f"  Min:    {min_ms:.2f} ms")
    print(f"  Max:    {max_ms:.2f} ms")
    print(f"  Tokens/sec: {tokens_per_sec:,.0f}")
    print(f"  MoE FFN params: {param_count_m:.1f}M")
    print(f"  Achieved: {achieved_tflops:.2f} TFLOPS")
    print(f"  Peak ({num_devices}x {device_kind}): {peak_tflops:.0f} TFLOPS")
    print(f"  MFU: {mfu:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Benchmark grug moe_mlp on GPU")
    parser.add_argument("--tokens", type=int, default=131072,
                        help="Total tokens across all devices (sharded along data axis)")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for model mode (batch_size * seq_len = total tokens)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--impl", type=str, default="auto", choices=["auto", "triton", "xla"],
                        help="ragged_dot implementation to benchmark (for moe_mlp/kernel modes)")
    parser.add_argument("--mode", type=str, default="moe_mlp", choices=["moe_mlp", "kernel", "model"],
                        help="model = full Grug Transformer; moe_mlp = MoE FFN only; kernel = isolated ragged_dot")
    parser.add_argument("--ep-size", type=int, default=1,
                        help="Expert parallelism degree (1 = no EP, 8 = all GPUs for EP)")
    parser.add_argument("--moe-impl", type=str, default="ring", choices=["ring", "ragged_all_to_all"],
                        help="MoE EP implementation strategy (only used when ep_size > 1)")
    parser.add_argument("--skewed", action="store_true",
                        help="Use skewed routing (80%% of tokens to 2 experts) instead of uniform")
    parser.add_argument("--grad", action="store_true",
                        help="Benchmark forward + backward pass (via jax.grad) instead of forward only")

    args = parser.parse_args()
    if args.mode == "model":
        bench_model(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()
