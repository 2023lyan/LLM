import os
import typing
import numpy as np
import numpy.typing as npt
import torch
import argparse
import pathlib
from tqdm import tqdm
import sys
from einops import rearrange
import logging
import wandb
from jaxtyping import Int
from torch import Tensor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import model

os.environ["http_proxy"] = "socks5h://127.0.0.1:10808"
os.environ["https_proxy"] = "socks5h://127.0.0.1:10808"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="batch size used for training the model (train.py)"
    )
    parser.add_argument(
        "--context_length",
        default=256,
        type=int,
        help="context length used for training the model (train.py)"
    )
    parser.add_argument(
        "--dataset",
        default = 0,
        type=int,
        help="dataset used for training the model (train.py). 0 for TinyStories, 1 for OWT"
    )
    parser.add_argument(
        "--vocab_size",
        default=10000,
        type=int,
        help="vocabulary size used for training the model (train.py)"
    )
    parser.add_argument(
        "--num_layers",
        default=4,
        type=int,
        help="number of layers in the transformer model (train.py)"
    )
    parser.add_argument(
        "--d_model",
        default=512,
        type=int,
        help="dimension of the model in the transformer (train.py)"
    )
    parser.add_argument(
        "--num_heads",
        default=16,
        type=int,
        help="number of attention heads in the transformer (train.py)"
    )
    parser.add_argument(    
        "--d_ff",
        default=1344,
        type=int,
        help="dimension of the feed-forward network in the transformer (train.py)"
    )
    parser.add_argument(
        "--rope_theta",
        default=10000.0,
        type=float,
        help="theta value for the rotary positional embedding in the transformer (train.py)"
    )
    parser.add_argument(
        "--num_epoch",
        default=20000,
        type=int,
        help="number of epochs to train the model (train.py)"
    )
    parser.add_argument(
        "--lr_min",
        default=1e-5,
        type=float,
        help="minimum learning rate for the learning rate schedule (train.py)"
    )
    parser.add_argument(
        "--lr_max",
        default=2e-4,
        type=float,
        help="number of iterations for the learning rate warmup (train.py)"
    )
    parser.add_argument(
        "--warmup_steps",
        default=1000,
        type=int,
        help="number of iterations for the learning rate warmup (train.py)"
    )
    parser.add_argument(
        "--decay_steps",
        default=15000,
        type=int,
        help="number of iterations for the learning rate decay (train.py)"
    )
    parser.add_argument(
        "--max_norm",
        default=0.5,
        type=float,
        help="maximum norm for gradient clipping (train.py)"
    )
    parser.add_argument(
        "--beta1",
        default=0.9,
        type=float,
        help="beta1 for AdamW optimizer (train.py)"
    )
    parser.add_argument(
        "--beta2",
        default=0.999,
        type=float,
        help="beta2 for AdamW optimizer (train.py)"
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-2,
        type=float,
        help="weight decay for AdamW optimizer (train.py)"
    )
    args = parser.parse_args()
    return args

DATA_PATH = pathlib.Path(__file__).resolve().parent.parent / "data"
TINYSTORY_TRAIN = DATA_PATH / "tinystories_train_tokens.npy"
OWT_TRAIN = DATA_PATH / "owt_train_tokens.npy"
TINYSTORY_VALID = DATA_PATH / "tinystories_valid_tokens.npy"
OWT_VALID = DATA_PATH / "owt_valid_tokens.npy"
CKPT_PATH = pathlib.Path(__file__).resolve().parent.parent / "checkpoints"
os.makedirs(CKPT_PATH, exist_ok=True)

def set_logger(log_name, log_dir, args):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(
            log_dir,
            f"{args.lr_max}_{args.lr_min}_{args.batch_size}_{args.context_length}_{args.dataset}.log"
        )
    )
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

logger = set_logger("train_logger", "logs/train", parse_args())

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="2023lyan-sjtu-hpc-center",
    # Set the wandb project where this run will be logged.
    project="Implementation of Transformer Language Model",
    # Track hyperparameters and run metadata.
    config={
        "batch_size": parse_args().batch_size,
        "context_length": parse_args().context_length,
        "dataset": parse_args().dataset,
        "vocab_size": parse_args().vocab_size,
        "num_layers": parse_args().num_layers,
        "d_model": parse_args().d_model,
        "num_heads": parse_args().num_heads,
        "d_ff": parse_args().d_ff,
        "rope_theta": parse_args().rope_theta,
        "num_epoch": parse_args().num_epoch,
        "lr_min": parse_args().lr_min,
        "lr_max": parse_args().lr_max,
        "warmup_steps": parse_args().warmup_steps,
        "decay_steps": parse_args().decay_steps,
        "max_norm": parse_args().max_norm,
        "device": device,
        "tokenizer_type": parse_args().dataset,  # 0 for TinyStories, 1 for OWT
    },
)

class DataLoader():
    def __init__(self, batch_size: int, context_length: int, device: str, dataset: npt.NDArray):
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
        self.dataset = dataset
    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        start = np.random.randint(low = 0, high = len(self.dataset) - self.context_length, 
                                  size=self.batch_size)
        inputs = np.stack([self.dataset[i:i + self.context_length] for i in start])
        labels = np.stack([self.dataset[i + 1:i + self.context_length + 1] for i in start])
        inputs = torch.tensor(inputs, dtype=torch.long, device=self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        return (inputs, labels)
    
class Checkpoint_Manager():
    def __init__(self):
        pass
    def save_checkpoint(self, model: torch.nn.Module,
                        optimizer: torch.optim.Optimizer, 
                        iteration: int, 
                        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration
        }
        torch.save(checkpoint, out)
    def load_checkpoint(self, 
                        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
                        model: torch.nn.Module, 
                        optimizer: torch.optim.Optimizer) -> int:
        checkpoint = torch.load(src)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['iteration']

class Trainer():
    def __init__(self,
                 model: model.Transformer_LM,
                 optimizer: torch.optim.Optimizer,
                 loss: model.Cross_Entropy_Loss,
                 dataloader: DataLoader,
                 device: str,
                 scheduler: model.Learning_Rate_Schedule,
                 clipper: model.Gradient_Clipping,
                 valid: DataLoader,
                 manager: Checkpoint_Manager = Checkpoint_Manager()):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.device = device
        self.losses = []
        self.valid_losses = {}
        self.scheduler = scheduler
        self.clipper = clipper
        self.manager = manager
        self.valid = valid

    def train(self, num_epochs: int):
        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            inputs, labels = self.dataloader.get_batch()
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(
                rearrange(outputs, "batch_size sequence_length vocab_size -> (batch_size sequence_length) vocab_size"),
                rearrange(labels, "batch_size sequence_length -> (batch_size sequence_length)")
            )
            loss.backward()
            lr = self.scheduler.get_lr(epoch + 1)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.clipper.clip(self.model.parameters())
            self.optimizer.step()
            
            loss_msg = f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}, LR: {lr:.6e}"
            run.log({"Iteration": epoch + 1, "Loss": loss.item(), "Learning Rate": lr})
            print(loss_msg, flush=True)
            
            self.losses.append(loss.item())
            if (epoch + 1) % 1000 == 0:
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
                validation_loss = []
                for _ in range(400):
                    input_valid, labels_valid = self.valid.get_batch()
                    input_valid, labels_valid = input_valid.to(self.device), labels_valid.to(self.device)
                    validation_loss.append(self.model.compute_validation_loss(input_valid, labels_valid).to("cpu"))
                run.log({"Iteration": epoch + 1, "Validation Loss": np.mean(validation_loss).item()})
                logger.info(f"Validation Loss: {np.mean(validation_loss).item()}")
                print(f"Validation Loss: {np.mean(validation_loss).item()}", flush=True)
                self.valid_losses[epoch + 1] = np.mean(validation_loss).item()
            if (epoch + 1) % 1000 == 0:
                self.manager.save_checkpoint(model = self.model, optimizer=self.optimizer, iteration=epoch + 1, out = CKPT_PATH / f"transformer_{parse_args().dataset}_epoch_{epoch + 1}.pt")
        with torch.no_grad():
            torch.save(self.model.state_dict(), CKPT_PATH / f"transformer_{parse_args().dataset}_{parse_args().lr_max}_final.pt")
            np.save(CKPT_PATH / f"transformer_{parse_args().dataset}_final_losses_{parse_args().lr_max}.npy", np.array(self.losses))
            np.save(CKPT_PATH / f"transformer_{parse_args().dataset}_final_valid_losses_{parse_args().lr_max}.npy", np.array(list(self.valid_losses.values())))

if __name__ == "__main__":
    args = parse_args()
    batch_size = args.batch_size
    context_length = args.context_length
    dataset_choice = args.dataset
    vocab_size = args.vocab_size
    num_layers = args.num_layers
    d_model = args.d_model
    num_heads = args.num_heads
    d_ff = args.d_ff
    rope_theta = args.rope_theta
    num_epoch = args.num_epoch
    transformer_model = model.Transformer_LM(
        vocab_size=vocab_size,
        context_length=context_length,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=torch.float32
    )
    if dataset_choice == 0:
        dataset = np.memmap(TINYSTORY_TRAIN, dtype=np.uint16, mode='r')
        valid_dataset = np.memmap(TINYSTORY_VALID, dtype=np.uint16, mode='r')
    elif dataset_choice == 1:
        dataset = np.memmap(OWT_TRAIN, dtype=np.uint16, mode='r')
        valid_dataset = np.memmap(OWT_VALID, dtype=np.uint16, mode='r')
    else:
        raise ValueError("Invalid dataset choice. Use 0 for TinyStories or 1 for OWT.")
    dataloader = DataLoader(batch_size=batch_size, context_length=context_length, device=device, dataset=dataset)
    valid = DataLoader(batch_size=batch_size, context_length=context_length, device=device, dataset=valid_dataset)
    optimizer = model.AdamW(
        params=transformer_model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    loss = model.Cross_Entropy_Loss()
    scheduler = model.Learning_Rate_Schedule(
        lr_min=args.lr_min,
        lr_max=args.lr_max,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps
    )
    clipper = model.Gradient_Clipping(max_norm=args.max_norm)
    trainer = Trainer(model=transformer_model, 
                      optimizer=optimizer, 
                      loss = loss, 
                      dataloader=dataloader, 
                      device=device, 
                      scheduler=scheduler, 
                      valid = valid,
                      clipper=clipper)
    print("Starting training...", flush=True)
    trainer.train(num_epochs=num_epoch)  # Adjust the number of epochs as needed
    print("Training complete.", flush=True)
    run.finish()
