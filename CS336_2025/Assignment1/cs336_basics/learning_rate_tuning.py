from model import SGD
import torch

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# test the 3 lr: 1e1, 1e2, 1e3
opt = SGD([weights], lr=1e1)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.

print("Final loss for lr=1e1:", loss.cpu().item())

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# test the 3 lr: 1e1, 1e2, 1e3
opt = SGD([weights], lr=1e2)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
print("Final loss for lr=1e2:", loss.cpu().item())

weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# test the 3 lr: 1e1, 1e2, 1e3
opt = SGD([weights], lr=1e3)
for t in range(100):
    opt.zero_grad() # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean() # Compute a scalar loss value.
    print(loss.cpu().item())
    loss.backward() # Run backward pass, which computes gradients.
    opt.step() # Run optimizer step.
print("Final loss for lr=1e3:", loss.cpu().item())