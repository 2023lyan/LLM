import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.handles = []
        self.buckets = []
        self.param_to_bucket = {}
        self.bucket_ready_count = {}
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0)
        
        bucket_size_bytes = int(self.bucket_size_mb * 1024 * 1024)
        curr_bucket, curr_size = [], 0
        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue
            p_bytes = p.numel() * p.element_size()
            if curr_size + p_bytes > bucket_size_bytes:
                self.buckets.append(curr_bucket)
                curr_bucket, curr_size = [], 0
            curr_bucket.append(p)
            curr_size += p_bytes
        if curr_bucket:
            self.buckets.append(curr_bucket)
            
        for i, bucket in enumerate(self.buckets):
            self.bucket_ready_count[i] = 0
            for p in bucket:
                self.param_to_bucket[p] = i
        
        for p in self.module.parameters():
            if not p.requires_grad:
                continue
            
            def hook(param):
                param.grad.div_(dist.get_world_size())
                bucket_index = self.param_to_bucket[param]
                self.bucket_ready_count[bucket_index] += 1
                if self.bucket_ready_count[bucket_index] == len(self.buckets[bucket_index]):
                    grads = [parameter.grad for parameter in self.buckets[bucket_index]]
                    flat = _flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flat, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append((handle, flat, grads))
            
            p.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle, flat, grads in self.handles:
            handle.wait()
            for g, new_g in zip(grads, _unflatten_dense_tensors(flat, grads)):
                g.copy_(new_g)
        self.handles.clear()
        for k in self.bucket_ready_count:
            self.bucket_ready_count[k] = 0
    
