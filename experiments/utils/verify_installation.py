import torch


def main():

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of recognized GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f'CUDA:{i}: {torch.cuda.get_device_properties(i).name}')

if __name__ == "__main__":
    main()
    