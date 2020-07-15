import torch.distributed as dist

def get_rank():
    if not (dist.is_available() and dist.is_initialized()):
        return 0
    else:
        return dist.get_rank()
    