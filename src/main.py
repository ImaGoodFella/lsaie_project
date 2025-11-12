import os
import deepspeed
import torch

from model import Transformer, TransformerModelArgs
from dataset import ParquetDataset, CollatorForCLM  
from transformers import AutoTokenizer


deepspeed_config = {
    "train_micro_batch_size_per_gpu": 1
}

rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


model_config = TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        vocab_size=tokenizer.vocab_size,
        seq_len=args.sequence_length,
    )

model = Transformer(model_config)

model_engine, _, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config
)



train_dataset = ParquetDataset(
    data_files=["/capstor/store/cscs/ethz/large-sc-2/datasets/train_data.parquet"],  # Update with actual parquet file paths
    tokenizer=tokenizer,
    max_length=2048,
    num_proc=4
)

# Create collator
collator = CollatorForCLM(tokenizer=tokenizer)

# Create dataloader with your dataset
train_dataloader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collator,  # Use your custom collator
    num_workers=2,
    pin_memory=True
)


tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")


for batch in train_dataloader:
    device = torch.device("cuda", rank)
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)

    outputs = model_engine(input_ids, labels=labels)
    model_engine.backward(outputs.loss)
    model_engine.step()