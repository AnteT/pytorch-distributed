import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize the distributed environment
def init_process(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '172.18.0.2'  # Replace with the address of your master node
    os.environ['MASTER_PORT'] = '60000'  # Replace with the port number used for communication
    print(f'Waiting for remaining nodes...')
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method='tcp://172.18.0.2:6585'
    )
    print(f"Hello from rank: {dist.get_rank()}")

# Example usage
if __name__ == "__main__":
    # Assuming 4 processes in 4 containers
    world_size = 4
    rank = 0  # TODO change me!
    init_process(rank, world_size)
    
# >>> sampler = DistributedSampler(dataset) if is_distributed else None
# >>> loader = DataLoader(dataset, shuffle=(sampler is None),
# ...                     sampler=sampler)
# >>> for epoch in range(start_epoch, n_epochs):
# ...     if is_distributed:
# ...         sampler.set_epoch(epoch)


# python main.py --master-ip $ip_address$ --num-nodes 4 --rank $rank$

# env_dict = [
#     {
#     "MASTER_ADDR": "172.18.0.2"
#     ,"MASTER_PORT": 60_000
#     ,"RANK": 0
#     ,"WORLD_SIZE": 4
#     },
#     {
#     "MASTER_ADDR": "172.18.0.3"
#     ,"MASTER_PORT": 60_001
#     ,"RANK": 1
#     ,"WORLD_SIZE": 4
#     },
#     {
#     "MASTER_ADDR": "172.18.0.4"
#     ,"MASTER_PORT": 60_002
#     ,"RANK": 2
#     ,"WORLD_SIZE": 4
#     },
#     {
#     "MASTER_ADDR": "172.18.0.5"
#     ,"MASTER_PORT": 60_003
#     ,"RANK": 3
#     ,"WORLD_SIZE": 4
#     }
# ]

# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.net1 = nn.Linear(10, 10)
#         self.relu = nn.ReLU()
#         self.net2 = nn.Linear(10, 5)

#     def forward(self, x):
#         return self.net2(self.relu(self.net1(x)))


# def demo_basic(local_world_size, local_rank):

#     # setup devices for this process. For local_world_size = 2, num_gpus = 8,
#     # rank 0 uses GPUs [0, 1, 2, 3] and
#     # rank 1 uses GPUs [4, 5, 6, 7].
#     n = torch.cuda.device_count() // local_world_size
#     device_ids = list(range(local_rank * n, (local_rank + 1) * n))

#     print(
#         f"[{os.getpid()}] rank = {dist.get_rank()}, "
#         + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids} \n", end=''
#     )

#     model = ToyModel().cuda(device_ids[0])
#     ddp_model = DDP(model, device_ids)

#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     optimizer.zero_grad()
#     outputs = ddp_model(torch.randn(20, 10))
#     labels = torch.randn(20, 5).to(device_ids[0])
#     loss_fn(outputs, labels).backward()
#     optimizer.step()


# def spmd_main(local_world_size, local_rank):
#     # These are the parameters used to initialize the process group
    
#     if sys.platform == "win32":
#         # Distributed package only covers collective communications with Gloo
#         # backend and FileStore on Windows platform. Set init_method parameter
#         # in init_process_group to a local file.
#         if "INIT_METHOD" in os.environ.keys():
#             print(f"init_method is {os.environ['INIT_METHOD']}")
#             url_obj = urlparse(os.environ["INIT_METHOD"])
#             if url_obj.scheme.lower() != "file":
#                 raise ValueError("Windows only supports FileStore")
#             else:
#                 init_method = os.environ["INIT_METHOD"]
#         else:
#             # It is a example application, For convience, we create a file in temp dir.
#             temp_dir = tempfile.gettempdir()
#             init_method = f"file:///{os.path.join(temp_dir, 'ddp_example')}"
#         dist.init_process_group(backend="gloo", init_method=init_method, rank=int(env_dict["RANK"]), world_size=int(env_dict["WORLD_SIZE"]))
#     else:
#         print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
#         dist.init_process_group(backend="gloo")

#     print(
#         f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
#         + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
#     )

#     demo_basic(local_world_size, local_rank)

#     # Tear down the process group
#     dist.destroy_process_group()


# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # # This is passed in via launch.py
    # parser.add_argument("--local_rank", type=int, default=0)
    # This needs to be explicitly passed in
    # parser.add_argument("--local_world_size", type=int, default=1)
    # args = parser.parse_args()
    # # The main entry point is called directly without using subprocess
    # spmd_main(args.local_world_size, args.local_rank)

"""
python example.py
launch.py --nnode=1 --node_rank=0 --nproc_per_node=8 example.py --local_world_size=8
"""

"""
python -m torch.distributed.launch --nproc-per-node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node-rank=0 --master-addr="192.168.1.1"
           --master-port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
"""