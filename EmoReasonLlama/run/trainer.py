from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import random
import time
import math
import warnings
import logging
import shutil
import sys
import numpy as np
from packaging import version
from pathlib import Path
from transformers.utils import logging as hf_logging
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import transformers
from transformers.utils import is_accelerate_available, is_sagemaker_mp_enabled, is_datasets_available, is_apex_available
from transformers import Trainer
from transformers import TrainerCallback
from transformers import GenerationConfig
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import (
    TRAINING_ARGS_NAME, TRAINER_STATE_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, SCALER_NAME)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics, 
    is_torch_tpu_available
)
from transformers.integrations import hp_params
from transformers.trainer_pt_utils import find_batch_size, get_model_param_count, reissue_pt_warnings, IterableDatasetShard, DistributedLengthGroupedSampler, DistributedSamplerWithLoop, LengthGroupedSampler, SequentialDistributedSampler, ShardSampler
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.pytorch_utils import is_torch_less_than_1_11
from peft.peft_model import PeftModel
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from EmoReasonLlama.run.inference_by_generate import evaluate_for_emo_reason, save_predictions_and_metrics

if is_datasets_available():
    import datasets

if is_apex_available():
    from apex import amp

skip_first_batches = None
if is_accelerate_available():
    from accelerate import __version__ as accelerate_version

    if version.parse(accelerate_version) >= version.parse("0.16"):
        from accelerate import skip_first_batches

hf_logging.set_verbosity_info() # 将log的级别设为INFO
logger = hf_logging.get_logger("transformers") # 获取到与transformers库中相同的logger

def add_log_to_file(log_dir):
    _LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
    _DATE_FMT = '%m/%d/%Y %H:%M:%S'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'log.txt')
    fh = logging.FileHandler(log_file_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


class EmoReasonTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        '''
        重写__init__()的原因：
        1. transformers.trainer文件中实例化的logger，
            只实例化了StreamHandler()，并使用logger.addHandler加入了该对象（也就是只将log输出到控制台），
            而没有实例化FileHandler()对象（即没将log写入文件）
            我们希望能将log文件存到args.logging_dir中。

        2. 为了实现lora训练模式下，在ckpt目录中保存adapter_model，需要额外传入lora_args参数

        3. 回调函数只有`on_init_end()`，没有on_init_begin。而super().__init__()中也是涉及了一些log操作的。所以没法通过回调函数实现。

        4. 为支持multitask training，新增`aux_train_dataset_list`和`aux_collate_fn_list`
        '''
        log_dir = kwargs['args'].logging_dir
        add_log_to_file(log_dir)

        if 'lora_args' in kwargs.keys():
            self.lora_args = kwargs['lora_args']
            del kwargs['lora_args']
        else:
            self.lora_args = None

        if 'aux_train_dataset_list' in kwargs.keys():
            self.aux_train_dataset_list = kwargs['aux_train_dataset_list']
            del kwargs['aux_train_dataset_list']
        else:
            self.aux_train_dataset_list = None
        if 'aux_collate_fn_list' in kwargs.keys():
            self.aux_collate_fn_list = kwargs['aux_collate_fn_list']
            del kwargs['aux_collate_fn_list']
        else:
            self.aux_collate_fn_list = None

        super().__init__(*args, **kwargs)

        if isinstance(self.model, PeftModel):
            assert self.lora_args, 'The `lora_args` param is needed for the `EmoReasonTrainer` when using a lora model.'


    def _get_train_sampler_for_dataset(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.world_size <= 1:
                return RandomSampler(train_dataset, generator=generator)
            elif (
                self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
                and not self.args.dataloader_drop_last
            ):
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
            else:
                return DistributedSampler(
                    train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )


    def get_train_dataloader_for_dataset(self, train_dataset, data_collator) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self._train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler_for_dataset(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


    def training_step(self, model: nn.Module, main_task_inputs: Dict[str, Union[torch.Tensor, Any]], aux_task_inputs_list: List[Dict[str, Union[torch.Tensor, Any]]], step: int) -> torch.Tensor:
        '''
        rewrite for multitask training
        '''
        # perform the main task:
        main_tr_loss_step = super().training_step(model, main_task_inputs)
        # update the main task loss and task steps
        if (self.args.logging_nan_inf_filter
            and not is_torch_tpu_available()
            and (torch.isnan(main_tr_loss_step) or torch.isinf(main_tr_loss_step))
            ):
                # if loss is nan or inf simply add the average of previous logged losses
                self.task_loss_dict[self.main_task] += self.task_loss_dict[self.main_task] / (1 + self.task_step_dict[self.main_task])
        else:
            self.task_loss_dict[self.main_task] += main_tr_loss_step.clone()
        self.task_step_dict[self.main_task] += 1

        total_tr_loss_step = main_tr_loss_step
        for aux_task, aux_distrib, aux_position, aux_task_inputs in zip(self.aux_task_list, self.aux_task_distrib_list, self.aux_task_position_list, aux_task_inputs_list):
            if step % aux_distrib == aux_position:
                # perform the aux task:
                aux_tr_loss_step = super().training_step(model, aux_task_inputs)
                total_tr_loss_step = total_tr_loss_step + aux_tr_loss_step
                # update the aux task loss and task steps
                if (self.args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(aux_tr_loss_step) or torch.isinf(aux_tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        self.task_loss_dict[aux_task] += self.task_loss_dict[aux_task] / (1 + self.task_step_dict[aux_task])
                else:
                    self.task_loss_dict[aux_task] += aux_tr_loss_step.clone()
                self.task_step_dict[aux_task] += 1
            
        return total_tr_loss_step


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # get dataloaders
        self.main_task = args.main_task
        self.aux_task_list = args.aux_task_list
        self.aux_task_distrib_list = args.aux_task_distrib_list
        self.aux_task_position_list = args.aux_task_position_list
        self.aux_task_weight_list = args.aux_task_weight_list
        self.main_train_dataloader = self.get_train_dataloader_for_dataset(self.train_dataset, self.data_collator)
        self.aux_train_dataloader_list = []
        if self.aux_task_list and len(self.aux_task_list):
            for aux_train_dataset, aux_data_collator in zip(self.aux_train_dataset_list, self.aux_collate_fn_list):
                aux_train_dataloader = self.get_train_dataloader_for_dataset(aux_train_dataset, aux_data_collator)
                self.aux_train_dataloader_list.append(aux_train_dataloader)
        
        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(self.main_train_dataloader):
            len_dataloader = len(self.main_train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(self.main_train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(self.main_train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = self.main_train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        # -------------- hzp add -------------
        self.task_loss_dict = {}
        self.task_step_dict = {}
        self._total_task_loss_scalar_dict = {}
        for task_type in [self.main_task] + self.aux_task_list:
            self.task_loss_dict[task_type] = torch.tensor(0.0).to(args.device)
            self.task_step_dict[task_type] = 0
            self._total_task_loss_scalar_dict[task_type] = 0.0
        # ------------------------------------

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for loader in [self.main_train_dataloader] + self.aux_train_dataloader_list:
                    is_random_sampler = hasattr(loader, "sampler") and isinstance(
                        loader.sampler, RandomSampler
                    )
                    if is_torch_less_than_1_11 or not is_random_sampler:
                        # We just need to begin an iteration to create the randomization of the sampler.
                        # That was before PyTorch 1.11 however...
                        for _ in loader:
                            break
                    else:
                        # Otherwise we need to call the whooooole sampler cause there is some random operation added
                        # AT THE VERY END!
                        _ = list(loader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            for loader in [self.main_train_dataloader] + self.aux_train_dataloader_list:
                if isinstance(loader, DataLoader) and isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(epoch)
                elif hasattr(loader, "dataset") and isinstance(loader.dataset, IterableDatasetShard):
                    loader.dataset.set_epoch(epoch)

            # do not support the torch_tpu

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len_dataloader
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                self.main_train_dataloader = skip_first_batches(self.main_train_dataloader, steps_trained_in_current_epoch)
                for aux_loader in self.aux_train_dataloader_list:
                    aux_loader = skip_first_batches(aux_loader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for main_task_inputs, *aux_task_inputs_list in zip(self.main_train_dataloader, *self.aux_train_dataloader_list):
                # add `task_name` to each batch
                main_task_inputs['task_name'] = self.main_task
                for aux_task, aux_task_inputs in zip(self.aux_task_list, aux_task_inputs_list):
                    aux_task_inputs['task_name'] = aux_task

                step += 1
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    (total_batched_samples % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, main_task_inputs, aux_task_inputs_list, step)
                else:
                    tr_loss_step = self.training_step(model, main_task_inputs, aux_task_inputs_list, step)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                
                self.current_flos += float(self.floating_point_ops(main_task_inputs))
                for aux_distrib, aux_position, aux_task_inputs in zip(self.aux_task_distrib_list, self.aux_task_position_list, aux_task_inputs_list):
                    if step % aux_distrib == aux_position:
                        self.current_flos += float(self.floating_point_ops(aux_task_inputs))
                
                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            # if is_torch_tpu_available():
                            #     gradients = xm._fetch_gradients(self.optimizer)
                            #     xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    # elif is_torch_tpu_available():
                    #     if self.do_grad_scaling:
                    #         self.scaler.step(self.optimizer)
                    #         self.scaler.update()
                    #     else:
                    #         xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                # if is_torch_tpu_available():
                #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                #     xm.master_print(met.metrics_report())
                # else:
                logger.warning(
                    "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                    "configured. Check your training configuration if this is unexpected."
                )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            # if is_torch_tpu_available():
            #     xm.rendezvous("load_best_model_at_end")
            if args.local_rank != -1:
                dist.barrier()
            # elif is_sagemaker_mp_enabled():
            #     smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        for task_type in [self.main_task] + self.aux_task_list:
            self._total_task_loss_scalar_dict[task_type] += self.task_loss_dict[task_type].item()
            if task_type != self.main_task:
                distrib = self.aux_task_distrib_list[self.aux_task_list.index(task_type)]
            else:
                distrib = 1.0
            metrics["{}_loss".format(task_type)] = self._total_task_loss_scalar_dict[task_type] / self.state.global_step / distrib

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def compute_loss(self, model, inputs, return_outputs=False):
        '''
        hzp modified from `transformers.Trainer.compute_loss()`
        如果Dataset使用的是自己定义的格式，且collate_fn()返回的项中除了input_ids，labels，attention_mask还包含其它项，那么就需要在将输入送入模型之前，去掉其它项
        
        ---------
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        '''
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        valid_forward_keys = ["input_ids", "attention_mask", "position_ids", "past_key_values", "inputs_embeds", "labels", "use_cache", "output_attentions", "output_hidden_states", "return_dict"] # 参考具体的model
        valid_model_inputs = {k: v for k, v in inputs.items() if k in valid_forward_keys} # hzp add，去掉自定义的数据加载过程所读进来的其它项
        outputs = model(**valid_model_inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)

            # multiply the task loss weight when training
            task_name = inputs['task_name']
            if task_name != self.main_task and task_name in self.aux_task_list:
                task_weight = self.args.aux_task_weight_list[self.aux_task_list.index(task_name)]
                loss = loss * task_weight
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def _save_peft(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        assert isinstance(self.model, PeftModel)
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        
        # We do not save the tokenizer here since it's the same to the base model.

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


    def save_peft_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        '''
        hzp add for save the lora model, inspired by the `trainer.save_model()` func
        ---
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        '''
        if output_dir is None:
            output_dir = self.args.output_dir
        
        # only consider using deepspeed, do not support tpu, mp, DDP(fsdp)
        assert self.deepspeed
        
        # check if zero3 mode enabled for getting the state_dict
        if self.hf_deepspeed_config_orig.is_zero3():
            # use deepspeed engine internal function to gather state dict
            # state_dict_zero3 contains whole parameters of base and lora adapters
            # we will not extract lora parameters since peft save_pretrained will do that
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/peft_model.py#L125
            # https://github.com/huggingface/peft/blob/3714aa2fff158fdfa637b2b65952580801d890b2/src/peft/utils/save_and_load.py#L19
            state_dict_zero3 = self.model_wrapped._zero3_consolidated_16bit_state_dict()
            if self.args.local_rank == 0:
                state_dict = state_dict_zero3
        else:
            # in other mode we use original code from fastchat team, to make sure our change is minimum
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), self.lora_args.lora_bias
            )

        if self.args.should_save:
            self._save_peft(output_dir, state_dict)

        # 在`trainer.save_model()`中针对is_deepspeed_zero3_enabled的情况做了特殊处理，删掉`trainer._save()`中用`torch.save()`保存的ckpt，转而使用`trainer.deepspeed.save_checkpoint()`保存
        #   但这里，因为我们会统一对lora的情况用peft库中实现的`save_pretrained()`函数保存adapter_model，我估计这函数应该已经针对zero3的情况做处理了，所以这里就不加类似的判断和特殊处理了。

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")


    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()
        
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if isinstance(self.model, PeftModel):
            '''
            save the adapter_model when use the lora mode
            '''
            self.save_peft_model(output_dir, _internal_call=True)
            # 这里就不加`self.deepspeed.save_checkpoint(output_dir)`了，原因：
            #   该函数同时在global_stepxx目录里面保存bf16_..._optim_states.pt和mp_rank_..._states.pt，其中后者应该就是整个模型的权重。
            #   保存前者是为了能load已有的ckpt和优化器状态继续训练，但目前transformers.trainer中也没支持peft model的继续训练。     
            #   所以就先不支持这个功能了。为了节省空间这里也不加上述函数了，要不还得把完整模型权重保存一遍。
        else:
            self.save_model(output_dir, _internal_call=True)
            if self.deepspeed:
                # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
                # config `stage3_gather_16bit_weights_on_model_save` is True
                self.deepspeed.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
        #     smp.barrier()
        #     if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
        #         smp.save(
        #             opt_state_dict,
        #             os.path.join(output_dir, OPTIMIZER_NAME),
        #             partial=True,
        #             v3=smp.state.cfg.shard_optimizer_state,
        #         )
        #     if self.args.should_save:
        #         with warnings.catch_warnings(record=True) as caught_warnings:
        #             torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        #         if self.do_grad_scaling:
        #             torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        if self.args.should_save and not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # if is_torch_tpu_available():
        #     rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)


    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        '''
        hzp modified from `transformers.Trainer._maybe_log_save_evaluate()`
        修改主要体现在：
        - 适配重写后的`self.evaluate()`返回值
        - 在`self._save_checkpoint()`之后，加入对于当前模型预测结果以及预测指标的保存流程
        '''
        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)

            # add log of each task's loss
            for task_type in [self.main_task] + self.aux_task_list:
                if self.task_step_dict[task_type] > 0:
                    task_loss_scalar = self._nested_gather(self.task_loss_dict[task_type]).mean().item()
                    self._total_task_loss_scalar_dict[task_type] += task_loss_scalar
                    self.task_loss_dict[task_type] -= self.task_loss_dict[task_type] # reset task loss to zero
                    logs["{}_loss".format(task_type)] = round(task_loss_scalar * self.args.gradient_accumulation_steps / self.task_step_dict[task_type], 4) # 总的loss是以算上梯度累积后的总batch算的
                    self.task_step_dict[task_type] = 0

            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                gene_dict_lists = {} # hzp add
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_gene_dict_list, dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    gene_dict_lists[eval_dataset_name] = dataset_gene_dict_list # hzp add
                    metrics.update(dataset_metrics)
            else:
                gene_dict_lists, metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # ------- save predictions and evaluation metrics of current model ------
            run_dir = self._get_output_dir(self._trial)
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}" # copied from `self._save_checkpoint()`
            model_saved_dir = os.path.join(run_dir, checkpoint_folder) # 上面`self._save_checkpoint()`刚刚把当前模型存入的目录
            os.makedirs(model_saved_dir, exist_ok=True)
            
            if isinstance(self.eval_dataset, dict):
                for eval_set_name in gene_dict_lists.keys():
                    dataset_gene_dict_list = gene_dict_lists[eval_set_name]
                    dataset_metrics = {}
                    for key in list(metrics.keys()):
                        if key.startswith(f"eval_{eval_set_name}_"):
                            ori_key = key.split(f"eval_{eval_set_name}_")[-1]
                            dataset_metrics[ori_key] = metrics[key]
                    save_predictions_and_metrics(eval_set_name, model_saved_dir, dataset_gene_dict_list, dataset_metrics, self.args.save_predictions, self.args.save_metrics)
            else:
                eval_set_name = self.eval_dataset.set_name
                dataset_gene_dict_list = gene_dict_lists
                dataset_metrics = metrics
                save_predictions_and_metrics(eval_set_name, model_saved_dir, dataset_gene_dict_list, dataset_metrics, self.args.save_predictions, self.args.save_metrics)
            # ------------------------------------------------

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        # else: # 执行时不加“--save_strategy "epoch"”，control.should_save为False
        #     ori_args_should_save = self.args.should_save
        #     # 下边这行目前不行，报错：“AttributeError: can't set attribute 'should_save'”，应该是不能修改类的属性，先不搞了
        #     self.args.should_save = False # 不保存模型权重部分，执行`_save_checkpoint()`的其它部分（比如更新metric）
        #     self._save_checkpoint(model, trial, metrics=metrics)
        #     self.control = self.callback_handler.on_save(self.args, self.state, self.control)
        #     self.args.should_save = ori_args_should_save


    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        hzp modified from `transformers.Trainer.evaluate()`
        修改主要体现在：适配简化后的`evaluation_loop()`的输入和输出

        ------
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        # eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        eval_loop = self.evaluation_loop
        if self.args.use_legacy_prediction_loop:
            logger.warning(f"We only overwrite `evaluation_loop` here, so use `evaluation_loop`.")

        # output = eval_loop(
        #     eval_dataloader,
        #     description="Evaluation",
        #     # No point gathering the predictions if there are no metrics, otherwise we defer to
        #     # self.args.prediction_loss_only
        #     prediction_loss_only=True if self.compute_metrics is None else None,
        #     ignore_keys=ignore_keys,
        #     metric_key_prefix=metric_key_prefix,
        # )
        gene_dict_list, metrics_dict, num_samples = eval_loop(
            eval_dataloader,
            description="Evaluation",
            metric_key_prefix=metric_key_prefix
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        # if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        #     start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        metrics_dict.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics_dict)

        # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
        #     # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
        #     xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics_dict)

        self._memory_tracker.stop_and_update_metrics(metrics_dict)

        # return output.metrics
        return gene_dict_list, metrics_dict

    def evaluation_loop(
        self, 
        dataloader: DataLoader, 
        description: str,
        # prediction_loss_only: bool | None = None, 
        # ignore_keys: List[str] | None = None, 
        metric_key_prefix: str = "eval"
    ) -> EvalLoopOutput:
        """
        hzp modified from `transformers.Trainer.evaluation_loop()`
        原函数是从eval_dataloader中加载一个batch的数据，然后利用`self.prediction_step()`实现batch inference。
        但我们这里的任务不太适合batch inference，而应该利用`model.generate()`做逐sample的推理，所以这里按该要求修改了本函数。

        -------
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        # batch_size = self.args.eval_batch_size
        if self.args.eval_batch_size > 1:
            self.args.eval_batch_size = 1
            logger.warning(f"Find eval_batch_size > 1. However, we only support inference sample by sample now, so set the eval_batch_size to 1.")

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        # logger.info(f"  Batch size = {batch_size}")
        logger.info(f"  Batch size = {1}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        if args.eval_accumulation_steps is not None:
            args.eval_accumulation_steps = None
            logger.warning(f"We do not support `eval_accumulation_steps` now. Set it to `None`.")

        # use `model.generate()` to inference and evaluate
        generation_config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
        )
        gene_dict_list, metrics_dict = evaluate_for_emo_reason(eval_dataset, model, args.device, generation_config, args.max_new_tokens)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        else:
            raise Exception('eval dataset has no valid length')

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics_dict.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics_dict[f"{metric_key_prefix}_{key}"] = metrics_dict.pop(key)

        return gene_dict_list, metrics_dict, num_samples



class EvaluationCallback(TrainerCallback):
    '''
    使用Callbacks的目的：在现有的`Trainer.inner_training_loop()`中，在每个epoch结束后，启用log，evaluate以及save的过程

    --update 07.19:
    trainer中默认会采用DefaultFlowCallback，这里面重写了`on_epoch_end`，只要指定了args.logging_strategy，args.evaluation_strategy以及args.save_strategy = "epoch"，就会分别把下面的三个control设成True
    所以在设置好了上面说的三个args参数的情况下，可以把当前这个Callback类去掉了
    '''
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        训练过程的每个epoch结束后，把当前训练情况存入log，对所有指定的集合各做一遍目标任务上的评估，并考虑保存该epoch对应的模型。
        '''
        control.should_log = True # 把训练损失以及当前的学习率存入log
        control.should_evaluate = True # 执行`Trainer.evaluate()`，对给定的数据集进行评估
        control.should_save = True # 保存训练中的checkpoint
        return control


class LogCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):
        '''
        在Trainer中，self.log()中会唤起`on_log()`事件，需要在`on_log()`中具体实现如何对传入的dict形式的log数据生成对应的日志
        '''
        logger.info(logs)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        '''
        在训练结束后，需要在log中报道一下验证集上指定评价指标下最好的结果，以及指标对应的checkpoint
        '''
        logger.info('========== Finish ==========')
        logger.info('Metric for best model: {}'.format(args.metric_for_best_model))
        logger.info('The best metric (on dev set): {}'.format(state.best_metric)) # 执行时若不加“--save_strategy "epoch"”，这行和下一行输出的结果都是None
        logger.info('The best model checkpoint: {}'.format(state.best_model_checkpoint))

