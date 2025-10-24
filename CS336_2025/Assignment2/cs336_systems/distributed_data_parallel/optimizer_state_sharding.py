import torch
import torch.distributed as dist

class Optimizer_Sharder(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: torch.optim.Optimizer, **kwargs: any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.all_params = list(params)
        self.self_params = self.all_params[self.rank::self.world_size]
        self.optimizer = optimizer_cls(self.self_params, **kwargs)
        self.handles = []
        super().__init__(self.all_params, {})

        
    def step(self, closure=None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        for i, params in enumerate(self.all_params):
            handle = dist.broadcast(params.data, src=i % self.world_size, async_op=True)
            self.handles.append(handle)
        for h in self.handles:
            h.wait()
        self.handles.clear()
        
    def add_param_group(self, param_group: dict[str, any]):
        super().add_param_group(param_group)