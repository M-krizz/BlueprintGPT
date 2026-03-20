import torch

print("=== Improved Model (improved_v1.pt) ===")
ckpt = torch.load('learned/model/checkpoints/improved_v1.pt', map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt.get('epoch')}")
print(f"Loss: {ckpt.get('loss', 0):.4f}")
print(f"Config: {ckpt.get('config')}")

print("\n=== Original Model (kaggle_test.pt) ===")
ckpt2 = torch.load('learned/model/checkpoints/kaggle_test.pt', map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt2.get('epoch')}")
print(f"Loss: {ckpt2.get('loss', 0):.4f}")
print(f"Config: {ckpt2.get('config')}")
