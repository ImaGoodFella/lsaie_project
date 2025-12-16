import os
import time
import json

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed

from transformers import AutoTokenizer

from dataset import CollatorForCLM, ParquetDataset
from model import Transformer, TransformerModelArgs
from utils import (
    build_lr_scheduler,
    clip_grad_norm_,
    get_args,
    get_num_params,
    get_num_flop_per_token,
    init_logger,
    logger,
    PRECISION_STR_TO_DTYPE,
    set_default_dtype,
)


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
    train_ds = ParquetDataset(
        args.dataset,
        tokenizer,
        args.sequence_length,
        args.batch_size * args.training_steps,
    )
    train_collator = CollatorForCLM(args.sequence_length, tokenizer.pad_token_id)

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

    if args.deepspeed:
        # Load DeepSpeed config to check ZeRO settings
        with open(args.deepspeed_config, 'r') as f:
            ds_config_dict = json.load(f)
    else:
        ds_config_dict = {}

    zero_stage = ds_config_dict.get('zero_optimization', {}).get('stage', 0)
    # Only use zero.Init for Stage 3
    if args.deepspeed and zero_stage == 3:
        logger.info("Using ZeRO Stage 3 partition-at-init")
        with deepspeed.zero.Init(config_dict_or_path=ds_config_dict):
            with set_default_dtype(model_dtype):
                model = Transformer(model_config)
    else:
        with set_default_dtype(model_dtype):
            model = Transformer(model_config)

    # Calculate num_params BEFORE DeepSpeed initialization
    # (Stage 3 shards parameters, so counting after init would give 1/world_size of actual params)
    num_params = get_num_params(model, exclude_embedding=True, is_deepspeed=args.deepspeed)
    num_flop_per_token = get_num_flop_per_token(num_params, model_config)
    logger.info(f"Model parameters (excluding embedding): {num_params:,}")
    logger.info(f"FLOPs per token: {num_flop_per_token:,}")

    # --- MODEL, OPTIMIZER, SCHEDULER SETUP --- DeepSpeed
    if args.deepspeed:
        logger.info("Using DeepSpeed")

        # Load DeepSpeed config to check optimizer settings
        with open(args.deepspeed_config, 'r') as f:
            ds_config_dict = json.load(f)
        
        # Check if optimizer offload to CPU is enabled
        zero_config = ds_config_dict.get('zero_optimization', {})
        offload_optimizer = zero_config.get('offload_optimizer', {})
        optimizer_offload_enabled = offload_optimizer.get('device') == 'cpu'
        
        # Use DeepSpeedCPUAdam if optimizer offload is enabled, else let DeepSpeed create it
        if optimizer_offload_enabled:
            logger.info("Using DeepSpeedCPUAdam (optimizer offload to CPU enabled)")
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                model.parameters(), 
                lr=0.001,  # Default optimizer LR
                betas=(0.9, 0.999)
            )
        else:
            logger.info("Letting DeepSpeed create optimizer from config")
            optimizer = None

        # Initialize DeepSpeed - use args parameter like finetune_zero3.py
        model_engine, optimizer, train_dl, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            training_data=train_ds,
            collate_fn=train_collator
        )

        # Enable DeepCompile if specified in config
        if 'compile' in ds_config_dict:
            logger.info("Enabling Compile in DeepSpeed")
            model_engine.compile()

        train_dl_iterator = iter(train_dl)
    else:
        logger.info("Using native PyTorch")

        model = model.to(device)

        if args.compile:
            logger.info("Using `torch.compile`")
            model = torch.compile(model, fullgraph=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, fused=args.fused_optimizer
        )
        lr_scheduler = build_lr_scheduler(optimizer, args.lr_warmup_steps)

    model_engine.train() if args.deepspeed else model.train()

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

        # Profiling
        if args.profile and args.profile_step_start == train_step:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # --- FOR DEEPSPEED ---
        if args.deepspeed:
            batch = next(train_dl_iterator)
            input_ids = batch[0].to(model_engine.device)
            labels = batch[1].to(model_engine.device)
            
            actual_batch_size = input_ids.shape[0]
            ntokens_since_last_log += actual_batch_size * args.sequence_length * world_size
            num_items_in_batch = labels.ne(-100).sum()
            ntraining_tokens_since_last_log += num_items_in_batch * world_size

            # Forward pass
            logits = model_engine(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
            )
            loss = loss / num_items_in_batch
            del logits

            # Backpropagation
            model_engine.backward(loss)

            # Steps
            model_engine.step()

        # --- FOR NATIVE Pytorch ---
        else:
            input_ids, labels = next(train_dl_iterator)
            ntokens_since_last_log += args.batch_size * args.sequence_length * world_size
            num_items_in_batch = labels.ne(-100).sum()
            ntraining_tokens_since_last_log += num_items_in_batch * world_size

            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            logits = model(input_ids)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1).float(), labels.flatten(0, 1), reduction="sum"
            )
            loss = loss / num_items_in_batch
            del logits

            # Backpropagation
            loss.backward()

            # Steps
            clip_grad_norm_(model.parameters(), args.grad_max_norm)
            optimizer.step()
            lr_scheduler.step()

        # Logging
        if train_step == 1 or train_step % args.logging_frequency == 0:
            time_delta = time.perf_counter() - time_last_log
            # tokens per second per device, abbreviated as tps
            tps = ntokens_since_last_log / time_delta
            mfu = 100 * num_flop_per_token * tps / (989e12 * world_size)
            tflops = num_flop_per_token * tps / (1e12 * world_size)
            training_tps = ntraining_tokens_since_last_log / time_delta

            logger.info(
                f"Step: {train_step} | Loss: {loss.item():.2f} | Tokens per second: {tps:.2f} | Training tokens per second (%): {100*training_tps/tps:.2f} | MFU (%): {mfu:.2f} | TFLOPs: {tflops:.2f}"
            )
            ntokens_since_last_log = 0
            ntraining_tokens_since_last_log = 0
            time_last_log = time.perf_counter()

        # Profiling
        if args.profile and args.profile_step_end == train_step:
            torch.cuda.cudart().cudaProfilerStop()

    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    args = get_args()
    train(args)
