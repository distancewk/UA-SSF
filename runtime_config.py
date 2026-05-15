def resolve_torch_device(cuda_arg, cuda_available):
    cuda_arg = str(cuda_arg).strip().lower()
    if cuda_arg == "cpu":
        return "cpu"
    if cuda_available:
        return f"cuda:{cuda_arg}"
    return "cpu"
