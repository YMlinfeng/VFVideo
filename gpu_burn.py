import torch
import torch.multiprocessing as mp

def burn(gpu_id):
    torch.cuda.set_device(gpu_id)
    a = torch.randn(4096, 4096, device=f"cuda:{gpu_id}")
    b = torch.randn(4096, 4096, device=f"cuda:{gpu_id}")
    c = torch.empty(4096, 4096, device=f"cuda:{gpu_id}")
    while True:
        torch.matmul(a, b, out=c)

if __name__ == "__main__":
    mp.set_start_method("spawn")
    for i in range(torch.cuda.device_count()):
        mp.Process(target=burn, args=(i,)).start()
