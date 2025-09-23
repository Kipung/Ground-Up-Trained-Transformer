import torch


print("Torch version:", torch.__version__)
print("Compiled with CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")