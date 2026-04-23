import torch


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("No GPU detected by PyTorch.")
        return

    device_count = torch.cuda.device_count()
    print("GPU count:", device_count)

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = bytes_to_gb(props.total_memory)

        print(f"\nGPU {i}")
        print("Name:", torch.cuda.get_device_name(i))
        print("Total VRAM (GB):", round(total_mem_gb, 2))
        print("Compute capability:", f"{props.major}.{props.minor}")

        allocated_gb = bytes_to_gb(torch.cuda.memory_allocated(i))
        reserved_gb = bytes_to_gb(torch.cuda.memory_reserved(i))

        print("Allocated memory (GB):", round(allocated_gb, 2))
        print("Reserved memory (GB):", round(reserved_gb, 2))

    current_device = torch.cuda.current_device()
    print("\nCurrent device index:", current_device)
    print("Current device name:", torch.cuda.get_device_name(current_device))

    x = torch.randn(2, 2).cuda()
    print("\nGPU tensor test successful:")
    print(x)


if __name__ == "__main__":
    main()
