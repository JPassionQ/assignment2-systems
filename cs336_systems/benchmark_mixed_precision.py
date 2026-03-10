import argparse
import timeit
import numpy as np
import torch

# Import the model from your local package
# Assuming this script is run from a directory where cs336_basics is accessible
from cs336_basics.model import BasicsTransformerLM
from contextlib import nullcontext

# Define the model configurations from Table 1
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

def main():
    parser = argparse.ArgumentParser(description="Benchmark BasicsTransformerLM")
    parser.add_argument("--model-size", type=str, choices=MODEL_CONFIGS.keys(), default="small", help="Model size configuration to use.")
    parser.add_argument("--context-length", type=int, default=128, help="Context length for the model.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for benchmarking.")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size.")
    parser.add_argument("--warmup-steps", type=int, default=0, help="Number of warm-up steps before timing.")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of steps to measure.")
    parser.add_argument("--backward", action="store_true", help="Include backward pass in benchmarking.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision using autocast.")
    
    args = parser.parse_args()

    # 1. Initialize the model
    config = MODEL_CONFIGS[args.model_size]
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0
    ).to(args.device)
    
    model.train() if args.backward else model.eval()

    # 2. Generate a random batch of data
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    print(f"Benchmarking '{args.model_size}' model | Context: {args.context_length} | Backward: {args.backward}")
    print(f"Warmup steps: {args.warmup_steps} | Measurement steps: {args.num_steps} | Device: {args.device}\n")

    # 3. Run warm-up steps
    # 设置 Autocast 上下文
    if args.bf16 and args.device == "cuda":
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        ctx = nullcontext()

    for _ in range(args.warmup_steps):
        with ctx:
            out = model(x)
        if args.backward:
            loss = out.mean()
            loss.backward()
            model.zero_grad()
            
        if args.device == "cuda":
            torch.cuda.synchronize()

    # 4. Time the execution of n steps
    fwd_times = []
    bwd_times = []
    for _ in range(args.num_steps):
        # Forward pass
        start_fwd = timeit.default_timer()
        with ctx:
            out = model(x)
        if args.device == "cuda":
            torch.cuda.synchronize()
        end_fwd = timeit.default_timer()
        fwd_times.append(end_fwd - start_fwd)

        # Backward pass
        if args.backward:
            loss = out.mean()
            start_bwd = timeit.default_timer()
            loss.backward()
            if args.device == "cuda":
                torch.cuda.synchronize()
            end_bwd = timeit.default_timer()
            bwd_times.append(end_bwd - start_bwd)
            model.zero_grad()

    # Print results
    fwd_avg = np.mean(fwd_times) * 1000
    fwd_std = np.std(fwd_times) * 1000
    print(f"Forward Pass:  {fwd_avg:.2f} ms ± {fwd_std:.2f} ms")

    if args.backward:
        bwd_avg = np.mean(bwd_times) * 1000
        bwd_std = np.std(bwd_times) * 1000
        print(f"Backward Pass: {bwd_avg:.2f} ms ± {bwd_std:.2f} ms")
        print(f"Total Step:    {(fwd_avg + bwd_avg):.2f} ms ± {(np.std(np.array(fwd_times) + np.array(bwd_times)) * 1000):.2f} ms")

if __name__ == "__main__":
    main()