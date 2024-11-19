import torch
import argparse

def fill_gpu_memory(memory_size_gb):
    # Check if GPU is available
    if torch.cuda.is_available():
        # Set device to GPU
        device = torch.device("cuda")
        print(f"Using {torch.cuda.get_device_name(0)}")

        # Calculate the number of elements to fill up the GPU memory
        num_elements = int((memory_size_gb * (1024 ** 3)) // 4)  # Convert GB to bytes, and then to number of elements (assuming float32)
        tensor = torch.rand(num_elements, device=device)

        while True:
            # Some random GPU operations to keep it busy
            tensor = tensor * 2.0
            tensor = tensor - 0.5
    else:
        print("GPU is not available.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fill up GPU memory with random tensor")
    parser.add_argument("--m", type=float, default=16, help="Size of GPU memory to fill up in GB")
    args = parser.parse_args()

    fill_gpu_memory(args.m)