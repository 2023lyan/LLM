from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    )
tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
train_batch = load_dataset("allenai/tulu-3-sft-personas-math-filtered")["train"]
input_ids = train_batch["input_ids"].to(device)
labels = train_batch["labels"].to(device)

output_dir = "../"
model.save_pretrained(save_directory=output_dir)
tokenizer.save_pretrained(save_directory=output_dir)
gradient_accumulation_steps = 4
optimizer = torch.optim.AdamW(lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
data_loader = None
for idx, (inputs, labels) in enumerate(data_loader):
    # Forward pass.
    logits = model(input_ids).logits
    loss = F.cross_entropy(logits, labels) / gradient_accumulation_steps
    # Backward pass.
    loss.backward()
    if (idx + 1) % gradient_accumulation_steps == 0:
        # Update weights every `gradient_accumulation_steps` batches.
        optimizer.step()
        # Zero gradients every `gradient_accumulation_steps` batches.
        optimizer.zero_grad()
    