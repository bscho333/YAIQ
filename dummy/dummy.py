import torch

# Check if GPU is available
if torch.cuda.is_available():
    # Set device to GPU
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name(0)}")

    # Create a random tensor of size 32GB / 4 (since float32 is 4 bytes)
    num_elements = (32 * (1024 ** 3)) // 4
    tensor = torch.rand(num_elements, device=device)

    while True:
        # Some random GPU operations to keep it busy
        tensor = tensor * 2.0
        tensor = tensor - 0.5
else:
    print("GPU is not available.")