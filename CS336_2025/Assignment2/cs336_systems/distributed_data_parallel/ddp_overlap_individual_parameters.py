import torch
import torch.distributed as dist

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
            if not p.requires_grad:
                continue
            def hook(param):
                param.grad.div_(dist.get_world_size())
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(handle)
            
            p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for h in self.handles:
            h.wait()

        self.handles.clear()

