import torch

print('===GPU 信息检测 ===')
print(f'GPU 可用：{torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量：{torch.cuda.device_count()}')
    print(f'当前GPU编号：{torch.cuda.current_device()}')
    print(f'GPU名称：{torch.cuda.get_device_name(0)}')
    print(f'GPU显存总量：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
else:
    print("❌ CUDA不可用，当前是CPU版本torch")

