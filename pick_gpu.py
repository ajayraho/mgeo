#!/usr/bin/env python3
import subprocess

MIN_FREE_MEM = 20000  # MiB

def main():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception as e:
        print(f"# pick_gpu.py: nvidia-smi failed: {e}")
        return

    best_idx = None
    best_free = -1

    for line in out.strip().splitlines():
        parts = line.split(",")
        if len(parts) == 1:
            # format: "idx free"
            idx_str, free_str = parts[0].split()
        else:
            # format "idx, free" (just in case)
            idx_str, free_str = parts[0].strip(), parts[1].strip()

        idx = int(idx_str)
        free = int(free_str)

        if free < MIN_FREE_MEM:
            continue

        if free > best_free:
            best_free = free
            best_idx = idx

    if best_idx is None:
        # No GPU meets threshold: just comment out info
        print(f"# pick_gpu.py: No GPU has at least {MIN_FREE_MEM} MiB free.")
        print("# Available (index freeMiB):")
        for line in out.strip().splitlines():
            print(f"# {line}")
        return

    # Print shell commands to eval
    print(f'export CUDA_VISIBLE_DEVICES={best_idx}')
    print(f'echo "Using GPU {best_idx} with {best_free} MiB free"')

if __name__ == "__main__":
    main()
