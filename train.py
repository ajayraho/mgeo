import torch
import time
import sys

def reserve_memory():
    if not torch.cuda.is_available():
        print("❌ Error: CUDA/GPU is not available.")
        sys.exit(1)

    # Try allocating from 26GB down to 6GB
    for gb in range(20, 4, -1):
        try:
            # Calculate elements for float32 (4 bytes per element)
            total_bytes = gb * (1024 ** 3)
            num_elements = int(total_bytes / 4)
            
            # Attempt allocation
            buffer = torch.zeros(num_elements, dtype=torch.float32, device='cuda')
            
            # --- SETUP FOR 30-40% LOAD ---
            # We use the EXISTING buffer to create small matrices for calculation.
            # This ensures we don't allocate new memory and crash.
            matrix_size = 2048 
            el_count = matrix_size * matrix_size
            
            # Create 3 views into the big buffer (A @ B = C)
            # We treat the first chunk of memory as Matrix A, the second as B, etc.
            A = buffer[0 : el_count].view(matrix_size, matrix_size)
            B = buffer[el_count : 2*el_count].view(matrix_size, matrix_size)
            C = buffer[2*el_count : 3*el_count].view(matrix_size, matrix_size)

            # Fill A and B with random noise (in-place) so the GPU has real numbers to crunch
            A.normal_()
            B.normal_()

            print(f"✅ Success! {gb} GB is now pinned on {torch.cuda.get_device_name(0)}.")
            print("⚡ Maintaining approx 30-40% GPU-Util.")
            print("💤 Press Ctrl+C to release.")

            # --- DUTY CYCLE LOOP ---
            target_utilization = 0.25  # Aim for 35% usage
            cycle_seconds = 0.5        # Update cycle duration (shorter = smoother graph)

            while True:
                start_time = time.time()
                
                # 1. WORK PHASE (~35% of the time)
                while (time.time() - start_time) < (cycle_seconds * target_utilization):
                    # Matrix Multiplication is heavy on Tensor Cores
                    torch.mm(A, B, out=C)
                    # Sync ensures Python waits for GPU to actually finish before checking time
                    torch.cuda.synchronize() 

                # 2. REST PHASE (~65% of the time)
                elapsed = time.time() - start_time
                sleep_time = cycle_seconds - elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except RuntimeError:
            continue
        except KeyboardInterrupt:
            print("\n👋 Releasing memory and exiting.")
            return

    print("❌ Failed to allocate any memory between 26GB and 6GB.")

if __name__ == "__main__":
    reserve_memory()