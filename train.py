import os
import time
import json  # <-- IMPORT
import deepspeed  # <-- IMPORT
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  # <-- IMPORT
from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs
from utils import build_lr_scheduler, clip_grad_norm_, get_args, get_num_params, get_num_flop_per_token, init_logger, logger, PRECISION_STR_TO_DTYPE, set_default_dtype

def train(args):
  logger.info(f"Experiment args: {args}")
  
  # --- DEEPSPEED INIT ---
  if args.deepspeed:
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    world_size = torch.distributed.get_world_size()
  else:
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    world_size = 1
  # -----------------------

  model_dtype = PRECISION_STR_TO_DTYPE[args.model_dtype]

  # Set up DataLoader
  logger.info("Setting up DataLoaders...")
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
  train_ds = ParquetDataset(args.dataset, tokenizer, args.sequence_length, args.batch_size*args.training_steps)
  train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)

  # --- SAMPLER FOR DISTRIBUTED TRAINING ---
  train_sampler = DistributedSampler(train_ds) if args.deepspeed else None
  train_dl = DataLoader(train_ds,
                        batch_size=args.batch_size,
                        collate_fn=train_collator,
                        sampler=train_sampler) # <-- ADD SAMPLER
  # ----------------------------------------
  
  train_dl_iterator = iter(train_dl)

  # Set up Model
  logger.info("Setting up Model...")
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
  with set_default_dtype(model_dtype):
    model = Transformer(model_config) # Don't send to(device) yet
  
  # --- MODEL, OPTIMIZER, SCHEDULER SETUP ---
  if args.deepspeed:
    logger.info("Using DeepSpeed")
    
    # Load and update config with runtime args
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)
    
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size
    ds_config["train_batch_size"] = args.batch_size * world_size
    ds_config["optimizer"]["params"]["lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_max_lr"] = args.learning_rate
    ds_config["scheduler"]["params"]["warmup_num_steps"] = args.lr_warmup_steps
    ds_config["gradient_clipping"] = args.grad_max_norm
    if args.model_dtype == "bf16":
        ds_config["bf16"] = {"enabled": True}
    elif args.model_dtype == "fp16":
        ds_config["fp16"] = {"enabled": True}
        ds_config.pop("bf16", None)

    # Note: torch.compile is not compatible with ZeRO-2/3 yet.
    if args.compile:
        logger.warning("torch.compile is not used when DeepSpeed is enabled.")
        
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )
    # model_engine handles device placement
    
  else:
    logger.info("Using native PyTorch")
    model = model.to(device)
    if args.compile:
      logger.info("Using `torch.compile`")
      model = torch.compile(model, fullgraph=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer)
    lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)
  # ---------------------------------------------
  
  model.train() # Set to train mode

  # Utils
  num_flop_per_token = get_num_flop_per_token(
      get_num_params(model, exclude_embedding=True),
      model_config,
  )

  ntokens_since_last_log = 0
  ntraining_tokens_since_last_log = 0
  time_last_log = time.perf_counter()

  logger.info("Starting training!")
  train_step = 0
  
  if args.deepspeed:
      steps_per_rank = max(1, args.training_steps // world_size)
  else:
      steps_per_rank = args.training_steps
  
  while train_step < steps_per_rank:
    train_step += 1
    input_ids, labels = next(train_dl_iterator)
    ntokens_since_last_log += args.batch_size * args.sequence_length * world_size # Account for all GPUs
    num_items_in_batch = labels.ne(-100).sum()
    ntraining_tokens_since_last_log += num_items_in_batch * world_size # Account for all GPUs
    
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    # --- DEEPSPEED TRAINING STEP ---
    if args.deepspeed:
        logits = model_engine(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
        loss = loss / num_items_in_batch
        del logits
        
        model_engine.backward(loss)
        model_engine.step() # This handles grad clipping, optimizer step, and lr scheduler step
        
    # --- NATIVE PYTORCH TRAINING STEP ---
    else:
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum")
        loss = loss / num_items_in_batch
        del logits
        loss.backward()
        clip_grad_norm_(model.parameters(), args.grad_max_norm)
        optimizer.step()
        lr_scheduler.step()
    # ----------------------------------

    # Logging
    if (train_step == 1 or train_step % args.logging_frequency == 0):
      # Only log from rank 0 to avoid spam
      if args.local_rank <= 0: # This works for both deepspeed (-1) and native (0)
        time_delta = time.perf_counter() - time_last_log
        tps = ntokens_since_last_log / time_delta 
        # MFU calculation: 989e12 is H100 SXM FP16/BF16 TFLOPS (989) * 1e12
        # This should be 989 TFLOPS * 4 GPUs = 3956e12
        mfu = 100 * num_flop_per_token * tps / (989e12 * world_size)
        tflops = num_flop_per_token * tps / 1e12
        training_tps = ntraining_tokens_since_last_log / time_delta

        logger.info(f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second (per-GPU): {tps/world_size:.2f} | Total TFLOPs: {tflops:.2f} | MFU (%): {mfu:.2f}")
        ntokens_since_last_log = 0
        ntraining_tokens_since_last_log = 0
        time_last_log = time.perf_counter()
    
    # Profiling
    if args.profile and args.profile_step_end == train_step:
      logger.info("Ending profiling range")


  logger.info("Training completed")

if __name__ == "__main__":
  init_logger()
  args = get_args()
  train(args)