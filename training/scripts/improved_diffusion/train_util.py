import copy
import functools
import glob
import os
import random
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

# Lazy import tqdm to avoid errors when not installed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from . import dist_util, logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import sys
visual_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../visual"))
sys.path.insert(0, visual_path)
from visualize_with_mask import save_denoising_progress_mask_only, load_data_with_mask, find_image_and_mask_pairs

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        lr_scheduler="linear",  # Learning rate scheduler: "linear", "cosine", or "cosine_warmup"
        lr_min=0.0,  # Minimum learning rate (for cosine annealing)
        lr_warmup_steps=0,  # Learning rate warmup steps
        viz_respacing=None,
        viz_gt_dir=None,
        viz_mask_dir=None,
        viz_gt_path=None,
        viz_mask_path=None,
        viz_schedule_jump_params=None,
        viz_output_dir=None,
        # New: model and diffusion parameters (for visualization)
        model_config=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.lr_scheduler = lr_scheduler  # "linear", "cosine", or "cosine_warmup"
        self.lr_min = lr_min  # Minimum learning rate
        self.lr_warmup_steps = lr_warmup_steps  # Warmup steps

        # Save model config (for visualization)
        self.model_config = model_config or {}

        # Visualization related parameters
        # Option 1: Use respacing (standard DDPM sampling, consistent with visualize_single_checkpoint_with_mask.sh option 1)
        # viz_respacing=None means use all steps (e.g., 1000 steps)
        # viz_respacing='250' means use 250 steps for accelerated visualization (timesteps will be resampled to 250 steps)
        self.viz_respacing = viz_respacing  # Timestep respacing for visualization, e.g., '250' for acceleration (set to None to use all steps)
        self.viz_num_images = 1  # Number of images to select for each visualization
        self.viz_fixed_indices = None  # Fixed image indices to ensure the same images are visualized each time

        # Visualization data directories (load GT and masks from specified directories)
        self.viz_gt_dir = viz_gt_dir  # GT image directory
        self.viz_mask_dir = viz_mask_dir  # Mask directory
        self.viz_gt_path = viz_gt_path  # Fixed GT image path
        self.viz_mask_path = viz_mask_path  # Fixed mask path
        self.viz_output_dir = viz_output_dir  # Visualization output directory name
        
        if viz_respacing is not None and viz_respacing != "None" and viz_respacing != "":
            self.viz_respacing = viz_respacing
        
        # Jump sampling parameters
        # If explicitly passed empty string "", use standard DDPM sampling (1000 steps)
        # If jump parameters are passed, use jump sampling (RePaint)
        # If no parameters are passed (None), use default jump sampling
        if viz_schedule_jump_params is not None and viz_schedule_jump_params != "None":
            if viz_schedule_jump_params == "":
                # Empty string means no jump sampling, use standard DDPM
                self.viz_schedule_jump_params = None
            elif isinstance(viz_schedule_jump_params, str):
                import ast
                try:
                    self.viz_schedule_jump_params = ast.literal_eval(viz_schedule_jump_params)
                except:
                    print(f"Warning: Unable to parse viz_schedule_jump_params: {viz_schedule_jump_params}, using default jump sampling")
                    self.viz_schedule_jump_params = {
                        't_T': 250,
                        'n_sample': 1,
                        'jump_length': 1,
                        'jump_n_sample': 5
                    }
            elif isinstance(viz_schedule_jump_params, dict):
                self.viz_schedule_jump_params = viz_schedule_jump_params
            else:
                # Other types, use default jump sampling
                self.viz_schedule_jump_params = {
                    't_T': 250,
                    'n_sample': 1,
                    'jump_length': 1,
                    'jump_n_sample': 5
                }
        else:
            # None or "None", use default jump sampling
            self.viz_schedule_jump_params = {
                't_T': 250,
                'n_sample': 1,
                'jump_length': 1,
                'jump_n_sample': 5
            }

        self.viz_image_pairs = None  # Image-mask pair list
        self.viz_selected_pair = None  # Currently selected image-mask pair (randomly selected first time, then kept constant)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()
        self.last_viz_batch = None
        
        self._viz_diffusion = None  # 1000-step diffusion object for visualization (lazy creation)
        # Progress bar related
        self.use_progress_bar = TQDM_AVAILABLE and dist.get_rank() == 0
        self.progress_bar = None
        self.start_time = None
        self.last_log_time = None

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
                logger.log("Successfully loaded optimizer state")
            except ValueError as e:
                logger.log(f"Warning: Could not load optimizer state: {e}")
                logger.log("This usually happens when the model architecture changed.")
                logger.log("Continuing with a fresh optimizer state (learning will restart from scratch)")
                logger.log("Model weights are still loaded - only optimizer state is reset")

    def _setup_fp16(self):
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        # Initialize progress bar
        total_steps = self.lr_anneal_steps if self.lr_anneal_steps else None
        if self.use_progress_bar:
            initial_step = self.step + self.resume_step
            self.progress_bar = tqdm(
                initial=initial_step,
                total=total_steps,
                desc="Training Progress",
                unit="step",
                ncols=120,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                dynamic_ncols=True
            )
            self.start_time = time.time()
            self.last_log_time = time.time()
            if initial_step > 0:
                logger.log(f"Resuming training from step {initial_step}...")
        
        try:
            while (
                not self.lr_anneal_steps
                or self.step + self.resume_step < self.lr_anneal_steps
            ):
                batch, cond = next(self.data)
                self.run_step(batch, cond)

                # Save current batch for visualization
                if self.step % self.save_interval == 0:
                    self.last_viz_batch = batch.clone()

                if self.step % self.log_interval == 0:
                    logger.dumpkvs()
                    # Visualization moved to save_interval, not executed at log_interval

                    # Update progress bar information
                    if self.use_progress_bar:
                        current_time = time.time()
                        elapsed = current_time - self.start_time
                        if self.last_log_time:
                            step_time = (current_time - self.last_log_time) / self.log_interval
                            # Get latest loss values (from logger)
                            log_dict = logger.getkvs()
                            loss_str = ""
                            if "loss" in log_dict:
                                loss_val = log_dict["loss"]
                                if isinstance(loss_val, (int, float)):
                                    loss_str = f"loss={loss_val:.4f}"
                            if "mse" in log_dict:
                                mse_val = log_dict["mse"]
                                if isinstance(mse_val, (int, float)):
                                    if loss_str:
                                        loss_str += f", mse={mse_val:.4f}"
                                    else:
                                        loss_str = f"mse={mse_val:.4f}"

                            # Update progress bar description
                            current_step = self.step + self.resume_step
                            desc = f"Training Progress [step={current_step}"
                            if total_steps:
                                desc += f"/{total_steps}, {current_step*100//total_steps}%"
                            if loss_str:
                                desc += f", {loss_str}"
                            desc += f", {step_time*1000:.0f}ms/step]"
                            self.progress_bar.set_description(desc)
                            # Update postfix information (displayed on the right side of progress bar)
                            postfix = {}
                            if loss_str:
                                postfix['loss'] = loss_str
                            postfix['speed'] = f"{step_time*1000:.0f}ms/step"
                            self.progress_bar.set_postfix(postfix)
                        self.last_log_time = current_time

                if self.step % self.save_interval == 0:
                    self.save()
                    # Perform visualization after saving (using separate process on CPU)
                    if dist.get_rank() == 0:
                        self.run_visualization()
                    # # Inline visualization (disabled, using separate process instead)
                    # if dist.get_rank() == 0:
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        if self.use_progress_bar:
                            self.progress_bar.close()
                        return

                # Update progress bar
                if self.use_progress_bar:
                    self.progress_bar.update(1)

                self.step += 1
        finally:
            # Close progress bar
            if self.use_progress_bar and self.progress_bar:
                self.progress_bar.close()
        
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        # print("self.use_fp16",self.use_fp16)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch, cond):
        zero_grad(self.model_params)
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            if self.use_fp16:
                loss_scale = 2 ** self.lg_loss_scale
                (loss * loss_scale).backward()
            else:
                loss.backward()

    def optimize_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        sqsum = 0.0
        # print(self.master_params)
        for p in self.master_params:
            # Skip parameters without gradients (e.g., frozen positional embeddings in DiT)
            if p.grad is None:
                continue
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return

        current_step = self.step + self.resume_step

        # Warmup phase: linearly increase learning rate
        if self.lr_warmup_steps > 0 and current_step < self.lr_warmup_steps:
            # Linearly increase from 0 to target learning rate
            warmup_frac = current_step / self.lr_warmup_steps
            lr = self.lr * warmup_frac
        else:
            # Learning rate scheduling after warmup
            if self.lr_warmup_steps > 0:
                # Calculate progress after warmup (relative to remaining steps)
                remaining_steps = self.lr_anneal_steps - self.lr_warmup_steps
                steps_after_warmup = current_step - self.lr_warmup_steps
                frac_done = steps_after_warmup / remaining_steps if remaining_steps > 0 else 1.0
            else:
                frac_done = current_step / self.lr_anneal_steps

            if self.lr_scheduler == "cosine" or self.lr_scheduler == "cosine_warmup":
                # Cosine annealing: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * frac_done))
                import math
                frac_done = min(frac_done, 1.0)  # Ensure not exceeding 1
                lr = self.lr_min + 0.5 * (self.lr - self.lr_min) * (1 + math.cos(math.pi * frac_done))
            else:
                # Linear decay: lr = lr * (1 - frac_done)
                lr = self.lr * (1 - frac_done)

        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()

    def _master_params_to_state_dict(self, master_params):
        if self.use_fp16:
            master_params = unflatten_master_params(
                tuple(self.model.parameters()), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params
    def run_visualization(self):
        """
        Call visualization script to visualize the just-saved weights
        """
        import subprocess
        
        try:
            # Get the just-saved weight file path
            current_step = self.step + self.resume_step

            # Build checkpoint path (EMA weights)
            ema_rate_str = self.ema_rate[0] if isinstance(self.ema_rate, list) else self.ema_rate
            checkpoint_path = os.path.join(
                get_blob_logdir(),
                f"ema_{ema_rate_str}_{current_step:06d}.pt"
            )

            # Check if file exists
            if not os.path.exists(checkpoint_path):
                logger.log(f"Warning: Checkpoint file does not exist: {checkpoint_path}")
                # Try using regular weights
                checkpoint_path = os.path.join(
                    get_blob_logdir(),
                    f"model{current_step:06d}.pt"
                )
                if not os.path.exists(checkpoint_path):
                    logger.log("Warning: Cannot find available checkpoint file, skipping visualization")
                    return

            logger.log(f"Starting visualization of checkpoint: {checkpoint_path}")

            # Build visualization command - use -m module mode to avoid relative import issues
            # Need to run in scripts directory
            scripts_dir = os.path.dirname(os.path.dirname(__file__))  # Parent directory (scripts/)

            # Select visualization image: prioritize fixed path, if not exist then select from backup directory
            viz_gt_path = self.viz_gt_path or ''
            viz_mask_path = self.viz_mask_path or ''

            # Check if fixed path exists
            use_fixed_path = False
            if self.viz_gt_path and self.viz_mask_path:
                if os.path.exists(self.viz_gt_path) and os.path.exists(self.viz_mask_path):
                    use_fixed_path = True
                    viz_gt_path = self.viz_gt_path
                    viz_mask_path = self.viz_mask_path
                    logger.log(f"Using fixed validation image: {os.path.basename(self.viz_gt_path)}")
                else:
                    logger.log(f"Warning: Fixed image does not exist")
                    logger.log(f"  GT path: {self.viz_gt_path} (exists: {os.path.exists(self.viz_gt_path)})")
                    logger.log(f"  Mask path: {self.viz_mask_path} (exists: {os.path.exists(self.viz_mask_path)})")

            # If fixed path does not exist, try selecting from backup directory
            if not use_fixed_path and self.viz_gt_dir and self.viz_mask_dir:
                logger.log(f"Trying to select image from backup directory...")
                # On first load, find all image-mask pairs
                if self.viz_image_pairs is None:
                    self.viz_image_pairs = find_image_and_mask_pairs(self.viz_gt_dir, self.viz_mask_dir)
                    if self.viz_image_pairs:
                        logger.log(f"Found {len(self.viz_image_pairs)} image-mask pairs")

                # Randomly select one pair on first time, then keep it constant
                if self.viz_image_pairs:
                    if self.viz_selected_pair is None:
                        import random
                        self.viz_selected_pair = random.choice(self.viz_image_pairs)
                        logger.log(f"Randomly selected image from backup directory: {self.viz_selected_pair[2]}")

                    viz_gt_path, viz_mask_path, filename = self.viz_selected_pair
                    logger.log(f"Using backup directory image: {filename}")
                else:
                    logger.log(f"Warning: No image-mask pairs found in backup directory")
                    logger.log(f"  GT directory: {self.viz_gt_dir}")
                    logger.log(f"  Mask directory: {self.viz_mask_dir}")

            # If still no image found, skip visualization
            if not viz_gt_path or not viz_mask_path:
                logger.log("Warning: No visualization image found, skipping visualization")
                return

            # Build base command
            cmd = [
                'python', '../../visual/visualize_with_mask.py',
            ]

            # ========== Add model parameters (consistent with training config) ==========
            # Get model parameters from saved config
            if self.model_config:
                # Model architecture parameters
                if 'image_size' in self.model_config:
                    cmd.extend(['--image_size', str(self.model_config['image_size'])])
                if 'num_channels' in self.model_config:
                    cmd.extend(['--num_channels', str(self.model_config['num_channels'])])
                if 'num_res_blocks' in self.model_config:
                    cmd.extend(['--num_res_blocks', str(self.model_config['num_res_blocks'])])
                if 'learn_sigma' in self.model_config:
                    cmd.extend(['--learn_sigma', str(self.model_config['learn_sigma'])])
                if 'num_in_channels' in self.model_config:
                    cmd.extend(['--num_in_channels', str(self.model_config['num_in_channels'])])
                if 'num_out_channels' in self.model_config:
                    cmd.extend(['--num_out_channels', str(self.model_config['num_out_channels'])])
                if 'model_type' in self.model_config:
                    cmd.extend(['--model_type', str(self.model_config['model_type'])])

                # DiT specific parameters
                if 'dit_depth' in self.model_config:
                    cmd.extend(['--dit_depth', str(self.model_config['dit_depth'])])
                if 'dit_hidden_size' in self.model_config:
                    cmd.extend(['--dit_hidden_size', str(self.model_config['dit_hidden_size'])])
                if 'dit_patch_size' in self.model_config:
                    cmd.extend(['--dit_patch_size', str(self.model_config['dit_patch_size'])])
                if 'dit_mlp_ratio' in self.model_config:
                    cmd.extend(['--dit_mlp_ratio', str(self.model_config['dit_mlp_ratio'])])

                # Diffusion parameters (critical! Must be consistent with training)
                if 'diffusion_steps' in self.model_config:
                    cmd.extend(['--diffusion_steps', str(self.model_config['diffusion_steps'])])
                if 'noise_schedule' in self.model_config:
                    cmd.extend(['--noise_schedule', str(self.model_config['noise_schedule'])])

                logger.log(f"Visualization using model config: diffusion_steps={self.model_config.get('diffusion_steps')}, noise_schedule={self.model_config.get('noise_schedule')}, model_type={self.model_config.get('model_type')}")
            else:
                logger.log("Warning: No model config provided, visualization may use default parameters")

            # ========== Add image and checkpoint paths ==========
            # Determine output directory (use directory name passed from script, or use default name)
            if self.viz_output_dir and self.viz_output_dir != '':
                output_dir_name = self.viz_output_dir
            else:
                # Default: automatically determine based on sampling method
                if self.viz_schedule_jump_params:
                    output_dir_name = 'val_denoising_progress_with_mask_jump'
                else:
                    output_dir_name = 'val_denoising_progress_with_mask_ddpm'

            cmd.extend([
                '--checkpoint_path', checkpoint_path,
                '--gt_path', viz_gt_path,
                '--mask_path', viz_mask_path,
                '--seed', '42',
                '--output_dir', os.path.join(get_blob_logdir(), output_dir_name),
            ])

            # Add respacing parameter
            if self.viz_respacing:
                cmd.extend(['--timestep_respacing', self.viz_respacing])

            # Add jump sampling parameters
            if self.viz_schedule_jump_params:
                import json
                cmd.extend(['--schedule_jump_params', json.dumps(self.viz_schedule_jump_params)])

            # Execute visualization (async, non-blocking training)
            logger.log(f"Executing visualization command: {' '.join(cmd)}")
            process = subprocess.Popen(cmd, cwd=scripts_dir)  # Direct output, show real-time progress bar

            # Wait for completion (no timeout limit)
            returncode = process.wait()
            if returncode == 0:
                logger.log(f"Visualization completed: step {current_step}")
            else:
                logger.log(f"Visualization failed (return code {returncode})")

            # Wait for subprocess to release GPU memory, then clean training process CUDA cache
            import time
            import gc
            time.sleep(2)  # Wait for CUDA driver to release subprocess GPU memory
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()
                logger.log("Training process CUDA cache cleared")

        except Exception as e:
            logger.log(f"Error during visualization: {str(e)}")
            import traceback
            logger.log(traceback.format_exc())




def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
