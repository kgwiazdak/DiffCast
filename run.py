import os
import os.path as osp
import math
import time
import argparse
import logging 
import re
import yaml
import cProfile
from tqdm import tqdm
from datetime import timedelta

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from datasets.get_datasets import get_dataset
from utils.metrics import Evaluator
from utils.tools import print_log, cycle, show_img_info

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "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"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"

def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--backbone',
        default='phydnet',
        type=str,
        choices=['simvp', 'phydnet', 'smaat'],
        help='backbone model for deterministic prediction',
    )
    parser.add_argument('--use_diff', action="store_true", default=False,        help='Weather use diff framework, as for ablation study')
    
    parser.add_argument("--seed",           type=int,   default=0,              help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='basic_exps',   help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default=None,           help="additional note for experiment")


    # --------------- Dataset ---------------
    parser.add_argument("--dataset",        type=str,   default='sevir',        help="dataset name")
    parser.add_argument("--img_size",       type=int,   default=128,            help="image size")
    parser.add_argument("--img_channel",    type=int,   default=1,              help="channel of image")
    parser.add_argument("--seq_len",        type=int,   default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",      type=int,   default=5,              help="number of frames to input")
    parser.add_argument("--frames_out",     type=int,   default=20,             help="number of frames to output")    
    parser.add_argument("--num_workers",    type=int,   default=4,              help="number of workers for data loader")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=6,              help="batch size")
    parser.add_argument("--epochs",         type=int,   default=20,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=200000,          help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    parser.add_argument(
        "--backbone_ckpt",
        type=str,
        default=None,
        help="checkpoint containing a deterministic backbone state to load before optional DiffCast wrapping",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        default=False,
        help="freeze the deterministic backbone after loading/building it; useful for residual-only DiffCast training",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=3,
        help="maximum number of saved checkpoints to keep; set to 0 to disable pruning",
    )
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="dataset split used by --eval; use valid for hyperparameter selection and test only for final reporting",
    )
    parser.add_argument(
        "--eval_conditioning_mode",
        type=str,
        default="autoregressive",
        choices=["autoregressive", "oracle_gt"],
        help="conditioning mode used only during diffusion eval diagnostics",
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=1.0,
        help="inference-only scale applied to the diffusion residual before adding it to backbone mu",
    )
    parser.add_argument(
        "--eval_residual_scales",
        type=str,
        default=None,
        help="comma-separated residual scales evaluated in one diffusion pass, e.g. 0,0.25,0.5,0.75",
    )
    parser.add_argument(
        "--eval_max_batches",
        type=int,
        default=None,
        help="optional maximum number of eval batches; use only for quick diagnostics, not final reporting",
    )
    parser.add_argument(
        "--skip_eval_image_save",
        action="store_true",
        default=False,
        help="skip saving per-sample eval images to speed up calibration sweeps",
    )
    parser.add_argument("--wandb_state",    type=str,   default='disabled',      help="wandb state config")
    parser.add_argument(
        "--smaat_teacher_forcing",
        type=float,
        default=0.5,
        help="teacher forcing ratio for the autoregressive SmaAt backbone",
    )
    parser.add_argument(
        "--smaat_loss",
        type=str,
        default="mse",
        choices=["mse", "l1", "charbonnier"],
        help="deterministic loss for the SmaAt backbone",
    )
    parser.add_argument(
        "--diff_teacher_forcing_ratio",
        type=float,
        default=1.0,
        help="deprecated: kept for CLI compatibility, ignored by the current DiffCast training path",
    )
    parser.add_argument(
        "--train_sampling_timesteps",
        type=int,
        default=8,
        help="number of DDIM steps used by the training-time conditioning sampler",
    )
    parser.add_argument(
        "--det_loss_weight",
        type=float,
        default=0.5,
        help="weight for the deterministic backbone loss in DiffCast training",
    )
    parser.add_argument(
        "--diff_loss_weight",
        type=float,
        default=0.5,
        help="weight for the diffusion residual loss in DiffCast training",
    )

    args = parser.parse_args()
    return args


class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        self._preparation()
        
        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   'wandb'
        )
        
        # Config log tracker 'wandb' from accelerate
        self.accelerator.init_trackers(
            project_name=self.exp_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": 
                {
                "mode": self.args.wandb_state,
                # 'resume': self.args.ckpt_milestone
                }
                         }   # disabled, online, offline
        )
        
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, self.scheduler,
            self.train_loader, self.valid_loader, self.test_loader
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device

    def _resolve_checkpoint_path(self, milestone):
        milestone = str(milestone)
        if milestone.endswith(".pt") or osp.sep in milestone or "/" in milestone or "\\" in milestone:
            return osp.abspath(milestone)
        return osp.abspath(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"))

    def _strip_state_prefix(self, state_dict, prefix):
        stripped = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if not stripped:
            return None
        return stripped

    def _load_module_checkpoint(self, module, checkpoint_path, label):
        resolved_path = self._resolve_checkpoint_path(checkpoint_path)
        if not osp.exists(resolved_path):
            raise FileNotFoundError(f"{label} checkpoint does not exist: {resolved_path}")

        data = torch.load(resolved_path, map_location="cpu")
        state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
        if not isinstance(state_dict, dict):
            raise ValueError(f"{label} checkpoint at {resolved_path} does not contain a state dict")

        candidates = [state_dict]
        for prefix in ("module.", "_orig_mod.", "backbone_net.", "module.backbone_net.", "_orig_mod.backbone_net."):
            stripped = self._strip_state_prefix(state_dict, prefix)
            if stripped is not None:
                candidates.append(stripped)

        errors = []
        for candidate in candidates:
            try:
                module.load_state_dict(candidate, strict=True)
                print_log(f"Loaded {label} checkpoint from {resolved_path}", self.is_main)
                return resolved_path
            except RuntimeError as exc:
                errors.append(str(exc))

        raise RuntimeError(
            f"Failed to load {label} checkpoint from {resolved_path}. "
            f"Tried raw state and common prefixes. Last error: {errors[-1] if errors else 'unknown'}"
        )

    def _freeze_module(self, module, label):
        for param in module.parameters():
            param.requires_grad = False
        print_log(f"Froze {label} parameters", self.is_main)
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.model_name = ('Diff' if self.args.use_diff else 'Single') + self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================

        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        if self.args.dataset != 'sevir':
            # preload big batch data for gradient accumulation
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
        
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================

        if self.args.backbone == 'simvp':
            from models.simvp import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'phydnet':
            from models.phydnet import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "device": self.device
            }
            model = get_model(**kwargs)

        elif self.args.backbone == 'smaat':
            from models.smaat import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "teacher_forcing_ratio": self.args.smaat_teacher_forcing,
                "loss_name": self.args.smaat_loss,
            }
            model = get_model(**kwargs)
            
        else:
            raise NotImplementedError

        if self.args.backbone_ckpt is not None:
            self._load_module_checkpoint(model, self.args.backbone_ckpt, f"{self.args.backbone} backbone")

        if self.args.freeze_backbone and not self.args.use_diff:
            raise ValueError("--freeze_backbone requires --use_diff; otherwise no trainable model remains")

        if self.args.freeze_backbone:
            self._freeze_module(model, f"{self.args.backbone} backbone")
        
        if self.args.use_diff:
            from diffcast import get_model
            kwargs = {
                'img_channels' : self.args.img_channel,
                'dim' : 64,
                'dim_mults' : (1,2,4,8),
                'T_in': self.args.frames_in,
                'T_out': self.args.frames_out,
                'sampling_timesteps': 250,
                'train_sampling_timesteps': self.args.train_sampling_timesteps,
                'diff_teacher_forcing_ratio': self.args.diff_teacher_forcing_ratio,
            }
            diff_model = get_model(**kwargs)
            diff_model.load_backbone(model)
            model = diff_model
            if self.is_main and self.args.diff_teacher_forcing_ratio != 1.0:
                print_log(
                    "Warning: --diff_teacher_forcing_ratio is deprecated and ignored in the current DiffCast training path.",
                    self.is_main,
                )
            
        self.model = model
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)


    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch)
        
        self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }

        final_path = osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt")
        tmp_path = f"{final_path}.tmp.{os.getpid()}"

        if osp.exists(tmp_path):
            os.remove(tmp_path)

        try:
            torch.save(data, tmp_path)
            os.replace(tmp_path, final_path)
            self._prune_old_checkpoints(keep=self.args.max_checkpoints)
        except Exception as exc:
            if osp.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(
                f"Failed to save checkpoint to {final_path}. Check filesystem health, free space, and quota."
            ) from exc

        print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)

    def _prune_old_checkpoints(self, keep):
        if keep is None or keep <= 0:
            return

        checkpoint_files = []
        for name in os.listdir(self.ckpt_path):
            match = re.fullmatch(r"ckpt-(\d+)\.pt", name)
            if match is None:
                continue
            checkpoint_files.append((int(match.group(1)), osp.join(self.ckpt_path, name)))

        if len(checkpoint_files) <= keep:
            return

        checkpoint_files.sort(key=lambda item: item[0], reverse=True)
        for _, old_path in checkpoint_files[keep:]:
            os.remove(old_path)
            print_log(f"Remove old checkpoint {old_path}", self.is_main)

    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================
        resolved_path = self._resolve_checkpoint_path(milestone)
        if not osp.exists(resolved_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")

        device = self.accelerator.device
        data = torch.load(resolved_path, map_location=device)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        
        self.optimizer.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'])

        self.cur_step = data['step']
        # Checkpoints are saved at the end of an epoch, so resume from the next one.
        self.cur_epoch = data['epoch'] + 1
        print_log(
            f"Load checkpoint {resolved_path} (requested: {milestone}) and resume at epoch {self.cur_epoch}, step {self.cur_step}",
            self.is_main,
        )
        
    
    def train(self):
        # set global step as traing process
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps,
            disable=not self.is_main,
        )
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):
            self.cur_epoch = epoch
            self.model.train()
            
            for i, batch in enumerate(self.train_loader):
                # train the model with mixed_precision
                with self.accelerator.autocast():

                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])
                    
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                for k,v in loss_dict.items():
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 20 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon, _ = self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                for i in range(radar_ori.shape[0]):
                                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(self.sanity_path, f"{i}/vil"),data_type='vil')

                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            # save checkpoint and do test every epoch
            self.save()
            print_log(f" ========= Finisth one Epoch ==========", self.is_main)

        self.accelerator.end_training()
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch      # [B, T, C, H, W]

    def _predict_model(self, *args, **kwargs):
        model = self.accelerator.unwrap_model(self.model)
        return model.predict(*args, **kwargs)
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        _, loss = self._predict_model(
            frames_in=frames_in,
            frames_gt=frames_out,
            compute_loss=True,
            T_out=self.args.frames_out,
            teacher_forcing_ratio=self.args.smaat_teacher_forcing,
            diff_teacher_forcing_ratio=self.args.diff_teacher_forcing_ratio,
            det_loss_weight=self.args.det_loss_weight,
            diff_loss_weight=self.args.diff_loss_weight,
        )
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        if isinstance(loss, tuple):
            total_loss, det_loss, diff_loss, *extra = loss
            loss_dict = {'total_loss': total_loss, 'det_loss': det_loss, 'diff_loss': diff_loss}
            extra_names = (
                'gt_res_abs_mean',
                'pred_res_abs_mean',
                'gt_res_std',
                'pred_res_std',
                'residual_amp_ratio',
            )
            for name, value in zip(extra_names, extra):
                loss_dict[name] = value
            return loss_dict
        loss_dict = {'total_loss': loss}
        if self.args.backbone == 'smaat' and not self.args.use_diff:
            raw_model = self.accelerator.unwrap_model(self.model)
            stats = getattr(raw_model, "last_rollout_stats", None)
            if stats:
                for key in ("pred_min", "pred_max", "pred_mean", "pred_std", "teacher_forcing_ratio"):
                    if key in stats:
                        loss_dict[key] = torch.tensor(stats[key], device=loss.device)
        return loss_dict
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False, return_mu=False, return_residual=False):
        sample_model = self.ema.ema_model if use_ema else self.accelerator.unwrap_model(self.model)
        sample_fn = sample_model.predict
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        radar_mu = None
        radar_residual = None
        if self.args.use_diff and return_mu and hasattr(sample_model, "sample"):
            radar_pred, radar_mu, radar_residual = sample_model.sample(
                radar_input,
                T_out=self.args.frames_out,
                frames_gt=radar_gt,
                conditioning_mode=self.args.eval_conditioning_mode,
                residual_scale=self.args.residual_scale,
            )
        else:
            radar_pred, _ = sample_fn(radar_input,compute_loss=False)
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()
        if radar_mu is not None:
            radar_mu = self.accelerator.gather(radar_mu).detach().cpu().numpy()
        if radar_residual is not None:
            radar_residual = self.accelerator.gather(radar_residual).detach().cpu().numpy()

        if return_residual:
            return radar_gt, radar_pred, radar_mu, radar_residual
        return radar_gt, radar_pred, radar_mu

    def _summarize_evaluator(self, evaluator):
        avg_csi, avg_far, avg_pod, avg_hss = [], [], [], []
        avg_csi44, avg_csi16 = [], []
        for threshold in evaluator.thresholds:
            hits = np.nan_to_num(np.array(evaluator.metrics[threshold]["hits"]))
            misses = np.nan_to_num(np.array(evaluator.metrics[threshold]["misses"]))
            falsealarms = np.nan_to_num(np.array(evaluator.metrics[threshold]["falsealarms"]))
            correctnegs = np.nan_to_num(np.array(evaluator.metrics[threshold]["correctnegs"]))

            csi = np.nan_to_num(
                np.mean(hits, axis=0)
                / (np.mean(hits, axis=0) + np.mean(misses, axis=0) + np.mean(falsealarms, axis=0))
            )
            far = np.nan_to_num(
                np.mean(falsealarms, axis=0)
                / (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0))
            )
            pod = np.nan_to_num(
                np.mean(hits, axis=0)
                / (np.mean(hits, axis=0) + np.mean(misses, axis=0))
            )
            hss = np.nan_to_num(
                2
                * (
                    np.mean(hits, axis=0) * np.mean(correctnegs, axis=0)
                    - np.mean(misses, axis=0) * np.mean(falsealarms, axis=0)
                )
                / (
                    (np.mean(hits, axis=0) + np.mean(misses, axis=0))
                    * (np.mean(misses, axis=0) + np.mean(correctnegs, axis=0))
                    + (np.mean(hits, axis=0) + np.mean(falsealarms, axis=0))
                    * (np.mean(falsealarms, axis=0) + np.mean(correctnegs, axis=0))
                )
            )

            hits44 = np.array(evaluator.metrics[threshold]["hits44"])
            misses44 = np.array(evaluator.metrics[threshold]["misses44"])
            falsealarms44 = np.array(evaluator.metrics[threshold]["falsealarms44"])

            hits16 = np.array(evaluator.metrics[threshold]["hits16"])
            misses16 = np.array(evaluator.metrics[threshold]["misses16"])
            falsealarms16 = np.array(evaluator.metrics[threshold]["falsealarms16"])

            avg_csi.append(np.mean(csi))
            avg_far.append(np.mean(far))
            avg_pod.append(np.mean(pod))
            avg_hss.append(np.mean(hss))
            avg_csi44.append(
                np.mean(hits44) / (np.mean(hits44) + np.mean(misses44) + np.mean(falsealarms44))
            )
            avg_csi16.append(
                np.mean(hits16) / (np.mean(hits16) + np.mean(misses16) + np.mean(falsealarms16))
            )

        return {
            "csi": float(np.nan_to_num(np.mean(avg_csi))),
            "far": float(np.nan_to_num(np.mean(avg_far))),
            "pod": float(np.nan_to_num(np.mean(avg_pod))),
            "hss": float(np.nan_to_num(np.mean(avg_hss))),
            "csi_pool_4x4": float(np.nan_to_num(np.mean(avg_csi44))),
            "csi_pool_16x16": float(np.nan_to_num(np.mean(avg_csi16))),
        }

    def _make_residual_stats_accumulator(self):
        return {
            "pred_abs_sum": 0.0,
            "gt_abs_sum": 0.0,
            "res_mae_sum": 0.0,
            "pred_sum": 0.0,
            "gt_sum": 0.0,
            "pred_sq_sum": 0.0,
            "gt_sq_sum": 0.0,
            "cross_sum": 0.0,
            "count": 0,
        }

    def _update_residual_stats(self, stats, pred_res, gt_res):
        pred_res = pred_res.astype(np.float64, copy=False)
        gt_res = gt_res.astype(np.float64, copy=False)
        stats["pred_abs_sum"] += float(np.abs(pred_res).sum())
        stats["gt_abs_sum"] += float(np.abs(gt_res).sum())
        stats["res_mae_sum"] += float(np.abs(pred_res - gt_res).sum())
        stats["pred_sum"] += float(pred_res.sum())
        stats["gt_sum"] += float(gt_res.sum())
        stats["pred_sq_sum"] += float(np.square(pred_res).sum())
        stats["gt_sq_sum"] += float(np.square(gt_res).sum())
        stats["cross_sum"] += float((pred_res * gt_res).sum())
        stats["count"] += pred_res.size

    def _finalize_residual_stats(self, stats):
        if stats["count"] <= 0:
            return None
        pred_abs = stats["pred_abs_sum"] / stats["count"]
        gt_abs = stats["gt_abs_sum"] / stats["count"]
        res_mae = stats["res_mae_sum"] / stats["count"]
        pred_mean = stats["pred_sum"] / stats["count"]
        gt_mean = stats["gt_sum"] / stats["count"]
        pred_var = max(stats["pred_sq_sum"] / stats["count"] - pred_mean ** 2, 0.0)
        gt_var = max(stats["gt_sq_sum"] / stats["count"] - gt_mean ** 2, 0.0)
        pred_std = pred_var ** 0.5
        gt_std = gt_var ** 0.5
        cov = stats["cross_sum"] / stats["count"] - pred_mean * gt_mean
        return {
            "pred_abs": pred_abs,
            "gt_abs": gt_abs,
            "abs_ratio": pred_abs / max(gt_abs, 1e-8),
            "std_ratio": pred_std / max(gt_std, 1e-8),
            "res_mae": res_mae,
            "res_corr": cov / max(pred_std * gt_std, 1e-8),
        }
    
    def _get_eval_split(self, split):
        if split == "valid":
            return self.valid_loader, self.valid_path, "Valid"
        if split == "test":
            return self.test_loader, self.test_path, "Test"
        raise ValueError(f"Unknown eval split: {split}")

    def _get_eval_residual_scales(self):
        if not self.args.use_diff or self.args.eval_residual_scales is None:
            return [self.args.residual_scale]

        scales = []
        for raw_scale in self.args.eval_residual_scales.split(","):
            raw_scale = raw_scale.strip()
            if not raw_scale:
                continue
            scales.append(float(raw_scale))

        if not scales:
            raise ValueError("--eval_residual_scales was provided but no numeric scales were parsed")

        return scales

    def _make_eval_bundle(self, seq_len, save_path):
        return {
            "eval": Evaluator(
                seq_len=seq_len,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_path,
            ),
            "frag1_eval": Evaluator(
                seq_len=min(self.args.frames_in, seq_len),
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_path,
            ),
            "residual_stats": self._make_residual_stats_accumulator(),
            "frag1_residual_stats": self._make_residual_stats_accumulator(),
        }

    def _update_eval_bundle(self, bundle, radar_ori, radar_pred, radar_mu, frag1_len):
        bundle["eval"].evaluate(radar_ori, radar_pred)
        frag1_ori = radar_ori[:, :frag1_len]
        frag1_pred = radar_pred[:, :frag1_len]
        bundle["frag1_eval"].evaluate(frag1_ori, frag1_pred)
        if radar_mu is None:
            return

        pred_res = (radar_pred - radar_mu).astype(np.float64, copy=False)
        gt_res = (radar_ori - radar_mu).astype(np.float64, copy=False)
        self._update_residual_stats(bundle["residual_stats"], pred_res, gt_res)

        frag1_mu = radar_mu[:, :frag1_len]
        frag1_pred_res = (frag1_pred - frag1_mu).astype(np.float64, copy=False)
        frag1_gt_res = (frag1_ori - frag1_mu).astype(np.float64, copy=False)
        self._update_residual_stats(bundle["frag1_residual_stats"], frag1_pred_res, frag1_gt_res)

    def _print_eval_bundle(self, result_label, label, bundle):
        res = bundle["eval"].done()
        print_log(f"{label} {result_label} Results: {res}")
        frag1_res = self._summarize_evaluator(bundle["frag1_eval"])
        print_log(f"Frag1 {label} {result_label} Results: {frag1_res}")
        finalized_residual_stats = self._finalize_residual_stats(bundle["residual_stats"])
        finalized_frag1_residual_stats = self._finalize_residual_stats(bundle["frag1_residual_stats"])
        if finalized_residual_stats is not None:
            print_log(
                f"{label} Residual Stats: "
                f"mean(abs(pred-mu))={finalized_residual_stats['pred_abs']:.6f}, "
                f"mean(abs(gt-mu))={finalized_residual_stats['gt_abs']:.6f}, "
                f"abs_ratio={finalized_residual_stats['abs_ratio']:.6f}, "
                f"std_ratio={finalized_residual_stats['std_ratio']:.6f}, "
                f"res_mae={finalized_residual_stats['res_mae']:.6f}, "
                f"res_corr={finalized_residual_stats['res_corr']:.6f}"
            )
        if finalized_frag1_residual_stats is not None:
            print_log(
                f"Frag1 {label} Residual Stats: "
                f"mean(abs(pred-mu))={finalized_frag1_residual_stats['pred_abs']:.6f}, "
                f"mean(abs(gt-mu))={finalized_frag1_residual_stats['gt_abs']:.6f}, "
                f"abs_ratio={finalized_frag1_residual_stats['abs_ratio']:.6f}, "
                f"std_ratio={finalized_frag1_residual_stats['std_ratio']:.6f}, "
                f"res_mae={finalized_frag1_residual_stats['res_mae']:.6f}, "
                f"res_corr={finalized_frag1_residual_stats['res_corr']:.6f}"
            )
    
    def test_samples(self, milestone, do_test=False, split=None):
        # init test data loader
        split = split or ("test" if do_test else "valid")
        data_loader, sample_root, result_label = self._get_eval_split(split)
        eval_scales = self._get_eval_residual_scales()
        multi_scale_eval = self.args.use_diff and len(eval_scales) > 1
        # init sampling method
        self.model.eval()
        # init test dir config
        cnt = 0
        save_dir = osp.join(sample_root, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        if self.is_main:
            print_log(f"Eval split: {split}", self.is_main)
            if self.args.use_diff:
                print_log(
                    f"Eval conditioning mode: {self.args.eval_conditioning_mode}; residual_scale: {self.args.residual_scale}",
                    self.is_main,
                )
                if multi_scale_eval:
                    print_log(f"Eval residual scales in one pass: {eval_scales}", self.is_main)
            if self.args.eval_max_batches is not None:
                print_log(f"Eval max batches: {self.args.eval_max_batches}", self.is_main)
            if self.args.skip_eval_image_save:
                print_log("Eval image saving disabled", self.is_main)

            frag1_len = min(self.args.frames_in, self.args.frames_out)
            if multi_scale_eval:
                scale_bundles = {
                    scale: self._make_eval_bundle(self.args.frames_out, save_dir)
                    for scale in eval_scales
                }
                eval = None
                frag1_eval = None
                residual_stats = None
                frag1_residual_stats = None
            else:
                eval = Evaluator(
                    seq_len=self.args.frames_out,
                    value_scale=self.scale_value,
                    thresholds=self.thresholds,
                    save_path=save_dir,
                )
                frag1_eval = Evaluator(
                    seq_len=frag1_len,
                    value_scale=self.scale_value,
                    thresholds=self.thresholds,
                    save_path=save_dir,
                )
                residual_stats = self._make_residual_stats_accumulator() if self.args.use_diff else None
                frag1_residual_stats = self._make_residual_stats_accumulator() if self.args.use_diff else None

            mu_eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            ) if self.args.use_diff else None
            frag1_mu_eval = Evaluator(
                seq_len=frag1_len,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            ) if self.args.use_diff else None
        # start test loop
        for batch_idx, batch in enumerate(tqdm(data_loader,desc=f'{result_label} Samples', disable=not self.is_main)):
            if self.args.eval_max_batches is not None and batch_idx >= self.args.eval_max_batches:
                break
            # sample
            sample_output = self._sample_batch(
                batch,
                use_ema=True,
                return_mu=self.args.use_diff,
                return_residual=multi_scale_eval,
            )
            if multi_scale_eval:
                radar_ori, radar_recon, radar_mu, radar_residual = sample_output
            else:
                radar_ori, radar_recon, radar_mu = sample_output
                radar_residual = None

            if self.is_main:
                frag1_ori = radar_ori[:, :frag1_len]

                if multi_scale_eval:
                    if radar_mu is None or radar_residual is None:
                        raise ValueError("Multi-scale residual eval requires diffusion mu and residual outputs")
                    for scale, bundle in scale_bundles.items():
                        scaled_recon = np.clip(radar_mu + scale * radar_residual, 0.0, 1.0)
                        self._update_eval_bundle(bundle, radar_ori, scaled_recon, radar_mu, frag1_len)
                    radar_recon_to_save = np.clip(
                        radar_mu + eval_scales[0] * radar_residual,
                        0.0,
                        1.0,
                    )
                else:
                    # evaluate result and save
                    eval.evaluate(radar_ori, radar_recon)
                    frag1_recon = radar_recon[:, :frag1_len]
                    frag1_eval.evaluate(frag1_ori, frag1_recon)
                    radar_recon_to_save = radar_recon
                    if radar_mu is not None:
                        pred_res = (radar_recon - radar_mu).astype(np.float64, copy=False)
                        gt_res = (radar_ori - radar_mu).astype(np.float64, copy=False)
                        self._update_residual_stats(residual_stats, pred_res, gt_res)
                        frag1_mu_for_stats = radar_mu[:, :frag1_len]
                        frag1_pred_res = (frag1_recon - frag1_mu_for_stats).astype(np.float64, copy=False)
                        frag1_gt_res = (frag1_ori - frag1_mu_for_stats).astype(np.float64, copy=False)
                        self._update_residual_stats(frag1_residual_stats, frag1_pred_res, frag1_gt_res)

                if mu_eval is not None and radar_mu is not None:
                    mu_eval.evaluate(radar_ori, radar_mu)
                    frag1_mu = radar_mu[:, :frag1_len]
                    frag1_mu_eval.evaluate(frag1_ori, frag1_mu)
                if not self.args.skip_eval_image_save:
                    for i in range(radar_ori.shape[0]):
                        self.visiual_save_fn(
                            radar_recon_to_save[i],
                            radar_ori[i],
                            osp.join(save_dir, f"{cnt}-{i}/vil"),
                            data_type='vil',
                        )

            self.accelerator.wait_for_everyone()
            cnt += 1
            # if cnt > 10:
            #     break
        # test done
        if self.is_main:
            if multi_scale_eval:
                print_log("========== Multi Residual Scale Results ==========")
                for scale, bundle in scale_bundles.items():
                    self._print_eval_bundle(result_label, f"scale={scale:g}", bundle)
            else:
                res = eval.done()
                print_log(f"{result_label} Results: {res}")
                frag1_res = self._summarize_evaluator(frag1_eval)
                print_log("========== First Fragment Diagnostics ==========")
                print_log(f"Frag1 {result_label} Results: {frag1_res}")
            if mu_eval is not None:
                print_log("========== Backbone Mu Results ==========")
                mu_res = mu_eval.done()
                print_log(f"Mu {result_label} Results: {mu_res}")
                frag1_mu_res = self._summarize_evaluator(frag1_mu_eval)
                print_log(f"Frag1 Mu {result_label} Results: {frag1_mu_res}")
                if multi_scale_eval:
                    for scale, bundle in scale_bundles.items():
                        frag1_res = self._summarize_evaluator(bundle["frag1_eval"])
                        print_log(
                            f"scale={scale:g} Frag1 Delta vs Mu: "
                            f"csi={frag1_res['csi'] - frag1_mu_res['csi']:.6f}, "
                            f"far={frag1_res['far'] - frag1_mu_res['far']:.6f}, "
                            f"pod={frag1_res['pod'] - frag1_mu_res['pod']:.6f}, "
                            f"hss={frag1_res['hss'] - frag1_mu_res['hss']:.6f}"
                        )
                else:
                    print_log(
                        "Frag1 Delta vs Mu: "
                        f"csi={frag1_res['csi'] - frag1_mu_res['csi']:.6f}, "
                        f"far={frag1_res['far'] - frag1_mu_res['far']:.6f}, "
                        f"pod={frag1_res['pod'] - frag1_mu_res['pod']:.6f}, "
                        f"hss={frag1_res['hss'] - frag1_mu_res['hss']:.6f}"
                    )
                    finalized_residual_stats = self._finalize_residual_stats(residual_stats)
                    finalized_frag1_residual_stats = self._finalize_residual_stats(frag1_residual_stats)
                    if finalized_residual_stats is not None:
                        print_log(
                            "Residual Stats: "
                            f"mean(abs(pred-mu))={finalized_residual_stats['pred_abs']:.6f}, "
                            f"mean(abs(gt-mu))={finalized_residual_stats['gt_abs']:.6f}, "
                            f"abs_ratio={finalized_residual_stats['abs_ratio']:.6f}, "
                            f"std_ratio={finalized_residual_stats['std_ratio']:.6f}, "
                            f"res_mae={finalized_residual_stats['res_mae']:.6f}, "
                            f"res_corr={finalized_residual_stats['res_corr']:.6f}"
                        )
                    if finalized_frag1_residual_stats is not None:
                        print_log(
                            "Frag1 Residual Stats: "
                            f"mean(abs(pred-mu))={finalized_frag1_residual_stats['pred_abs']:.6f}, "
                            f"mean(abs(gt-mu))={finalized_frag1_residual_stats['gt_abs']:.6f}, "
                            f"abs_ratio={finalized_frag1_residual_stats['abs_ratio']:.6f}, "
                            f"std_ratio={finalized_frag1_residual_stats['std_ratio']:.6f}, "
                            f"res_mae={finalized_frag1_residual_stats['res_mae']:.6f}, "
                            f"res_corr={finalized_frag1_residual_stats['res_corr']:.6f}"
                        )
            print_log("="*30)

        
    def check_milestones(self, target_ckpt=None, eval_split=None):
        eval_split = eval_split or self.args.eval_split

        mils_paths = os.listdir(self.ckpt_path)
        milestones = []
        for name in mils_paths:
            match = re.fullmatch(r"ckpt-(\d+)\.pt", name)
            if match is not None:
                milestones.append(int(match.group(1)))
        milestones = sorted(milestones, reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)
        
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = osp.splitext(osp.basename(str(target_ckpt)))[0]
            self.test_samples(saved_dir_name, split=eval_split)
            return
        
        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], split=eval_split)
            
def main():
    args = create_parser()
    exp = Runner(args)
    if not args.eval:
        exp.train()
        # Preserve the old auto-eval behavior only for fresh runs.
        if args.ckpt_milestone is None:
            exp.check_milestones(target_ckpt=args.ckpt_milestone)
        return

    exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    # 测试代码各模块执行效率
    # pip install graphviz
    # pip install gprof2dot
    # gprof2dot -f pstats train.profile | dot -Tpng -o result.png
    # cProfile.run('main()', filename='train.profile', sort='cumulative')
    main()
