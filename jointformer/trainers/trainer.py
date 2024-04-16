import os
import time
import math
import torch

from contextlib import nullcontext

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from jointformer.configs.trainer import TrainerConfig
from jointformer.utils.loggers.wandb import WandbLogger
from jointformer.models.prediction import MLM_PREDICTION_MODELS, LM_PREDICTION_MODELS

from jointformer.utils.runtime import set_seed


class Trainer:

    def __init__(
            self, config: TrainerConfig, model: torch.nn, tokenizer, logger: WandbLogger = None,
            train_dataset = None, eval_dataset = None):

        # out dir
        self.out_dir = config.out_dir
        self.resume = config.resume

        # eval
        self.eval_interval = config.eval_interval
        self.log_interval = config.log_interval
        self.eval_iters = config.eval_iters
        self.eval_only = config.eval_only  # if True, script exits right after the first eval
        self.logger = logger

        # load / save
        self.enable_save_checkpoint = config.enable_save_checkpoint
        self.always_save_checkpoint = config.always_save_checkpoint  # if True, always save a checkpoint after each eval
        self.init_from = config.init_from  # 'scratch' or 'resume' or 'gpt2*'

        # adamw optimizer
        self.learning_rate = config.learning_rate  # max learning rate
        self.max_iters = config.max_iters  # total number of training iterations
        self.weight_decay = config.weight_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.grad_clip = config.grad_clip  # clip gradients at this value, or disable if == 0.0

        # learning rate decay settings
        self.decay_lr = config.decay_lr  # whether to decay the learning rate
        self.warmup_iters = config.warmup_iters  # how many steps to warm up for
        self.lr_decay_iters = config.lr_decay_iters  # should be ~= max_iters per Chinchilla
        self.min_lr = config.min_lr  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # DDP settings
        self.is_ddp_run = None
        self.ddp_enabled = config.ddp_enabled
        self.ddp_backend = config.ddp_backend  # 'nccl', 'gloo', etc.

        # runtime
        self.gradient_accumulation_steps = config.gradient_accumulation_steps  # used to simulate larger batch sizes
        self.batch_size = config.batch_size  # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.device = config.device  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        self.device_type = config.device
        self.dtype = config.dtype  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.compile = config.compile  # use PyTorch 2.0 to compile the model to be faster

        # data
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # eval
        self.iter_num = 0
        self.best_val_loss = 1e9
        self.config = config

        # model
        self.model = model

        # scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == 'float16'))

        # optimizer
        self.optimizer = None
        self.correct_bias = config.correct_bias
        self.optimizer_ckpt = None
        self.current_learning_rate = None

        self._ddp_init()
        self._post_init()

    def load_checkpoint(self, path_to_checkpoint: str = None, reset_iter_num: bool = False) -> None:

        out_dir = path_to_checkpoint if path_to_checkpoint is not None else os.path.join(self.out_dir, 'ckpt.pt')
        if not os.path.isfile(out_dir):
            raise FileNotFoundError(f"Checkpoint file {out_dir} does not exist!")

        ckpt = torch.load(out_dir, map_location=next(self.model.parameters()).device)

        # clean ckpt
        state_dict = ckpt['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # load
        self.model.load_state_dict(state_dict, strict=False)
        # self.optimizer.load_state_dict(ckpt['optimizer'])
        self.optimizer_ckpt = ckpt['optimizer']
        if reset_iter_num:
            self.iter_num = 0
            self.best_val_loss = 1e9
        else:
            self.iter_num = ckpt['iter_num']
            self.best_val_loss = ckpt['best_val_loss']

        if 'run_id' in ckpt.keys() and self.logger is not None:
            self.logger.run_id = ckpt['run_id']

        print(f"Successfully loaded checkpoint from {out_dir}...")
        return None

    def save_checkpoint(self, checkpoint_dir: str = None) -> None:

        if self.enable_save_checkpoint:
            out_dir = checkpoint_dir if checkpoint_dir is not None else self.out_dir
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            run_id = None if self.logger is None else self.logger.run_id
            raw_model = self.model.module if self.is_ddp_run else self.model
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': self.iter_num,
                'best_val_loss': self.best_val_loss,
                'run_id': run_id,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f"Successfully saved to {self.out_dir}...")

    def _ddp_init(self) -> None:

        self.is_ddp_run = int(os.environ.get('RANK', -1)) != -1 and self.ddp_enabled
        if self.is_ddp_run:
            # print("DDP enabled!")
            # init_process_group(backend=self.ddp_backend)
            self.ddp_rank = int(os.environ['RANK'])
            self.ddp_local_rank = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.ddp_local_rank}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0  # this process will do loggers, checkpointing etc.
            self.seed_offset = self.ddp_rank  # each process gets a different seed
            # world_size number of processes will be training simultaneously, so we can scale
            # down the desired gradient accumulation iterations per process proportionally
            assert self.gradient_accumulation_steps % self.ddp_world_size == 0
            self.gradient_accumulation_steps //= self.ddp_world_size
        else:
            # if not ddp, we are running on a single gpu, and one process
            # print("Running on a single device!")
            self.master_process = True
            self.seed_offset = 0
            self.ddp_world_size = 1
        tokens_per_iter = self.gradient_accumulation_steps * self.ddp_world_size * self.batch_size * self.model.max_seq_len
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

    def _post_init(self):
        if self.master_process and not os.path.isdir(self.out_dir) and self.enable_save_checkpoint:
            os.makedirs(self.out_dir)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        print(f"Using {self.device_type} device")

    def _train_init(self):

        set_seed(1337 + self.seed_offset)

        self.model.to(self.device)

        self.optimizer = self.model.configure_optimizers(
            self.weight_decay, self.learning_rate, (self.beta1, self.beta2), self.device_type, self.correct_bias)

        if self.optimizer_ckpt:
            self.optimizer.load_state_dict(self.optimizer_ckpt)
            self.optimizer_ckpt = None

        if self.compile and not type(self.model).__name__ == 'OptimizedModule':
            print("Compiling model..")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model)

        # wrap model into DDP container
        if self.is_ddp_run:
            self.model = DDP(self.model, device_ids=[self.ddp_local_rank])

    # learning rate decay scheduler (cosine with warmup)
    def _get_lr(self, it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def set_lr(self, iter_num: int) -> None:
        self.current_learning_rate = self._get_lr(iter_num) if self.decay_lr else self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_learning_rate

    def get_batch(self, split, task):
        inputs = None
        if split == 'train':
            inputs = self.tokenizer.get_inputs(
                dataset=self.train_dataset, task=task, batch_size=self.batch_size, device=self.device)
        if split == 'val':
            inputs = self.tokenizer.get_inputs(
                dataset=self.eval_dataset, task=task, batch_size=self.batch_size, device=self.device)
        return inputs

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()

        ###
        # log the loss of the model->used for model selection // double the num of iters to equal for two type of tasks
        # in a supervised setting some models may have None loss for some tasks
        ###

        for split in ['train', 'val']:
            losses = torch.zeros(2 * self.eval_iters)

            for k in range(self.eval_iters):
                inputs = self.get_batch(split, 'lm')
                with self.ctx:
                    outputs = self.model(
                        task='lm', input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
                losses[k] = torch.Tensor([0.]) if outputs['loss'] is None else outputs['loss'].item()

            for k in range(self.eval_iters, 2 * self.eval_iters):
                inputs = self.get_batch(split, 'mlm')
                with self.ctx:
                    outputs = self.model(
                        task='mlm', input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
                losses[k] = torch.Tensor([0.]) if outputs['loss'] is None else outputs['loss'].item()

            out[split] = losses[losses != 0.].mean()

        ###
        # Log percent of valid molecules sampled
        ###

        valid = []
        for k in range(self.eval_iters):
            idx = torch.ones(size=(self.batch_size, 1), device=self.device) * self.tokenizer.generate_token_id
            idx = idx.long()
            samples = self.model.generate(idx=idx, max_new_tokens=self.tokenizer.max_molecule_length)
            valid.extend(self.tokenizer.is_valid_smiles(samples))
        out['valid'] = sum(valid) / len(valid)

        ###
        # Log supervised loss for fine-tuning purposes
        ###

        model_name = type(self.model._orig_mod).__name__

        if model_name in MLM_PREDICTION_MODELS:
            task = 'mlm'
        elif model_name in LM_PREDICTION_MODELS:
            task = 'lm'
        else:
            out['train_prediction'] = 0.
            out['val_prediction'] = 0.
            return out

        for split in ['train', 'val']:
            losses = torch.zeros(2 * self.eval_iters)
            for k in range(2 * self.eval_iters):
                inputs = self.get_batch(split, task)
                with self.ctx:
                    outputs = self.model(
                        task=task, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
                losses[k] = torch.Tensor([0.]) if outputs['supervised_loss'] is None else outputs['supervised_loss'].item()

            out[split + '_prediction'] = losses[losses != 0.].mean()

        self.model.train()
        return out

    def evaluate(self):
        losses = self.estimate_loss()

        print(
            f"Evaluation at iter {self.iter_num}: train loss {losses['train']:.4f},"
            f" val loss {losses['val']:.4f},"
            f" percent {losses['valid']:.4f}")

        if self.master_process and self.logger is not None:
            self.logger.log({
                    "iter": self.iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "train/loss_prediction": losses['train_prediction'],
                    "val/loss_prediction": losses['val_prediction'],
                    "val/valid": losses['valid'],
                    "lr": self.current_learning_rate,})

        if losses['val'] < self.best_val_loss or self.always_save_checkpoint:
            self.best_val_loss = losses['val']
            if self.iter_num > 0:
                self.save_checkpoint()

    @torch.no_grad()
    def test(self, dataset):
        batch_size = 64
        out = {}
        self._train_init()
        self.logger.init_run()
        self.model.eval()
        loss_l1 = torch.nn.L1Loss(reduction='mean')
        loss_mse = torch.nn.MSELoss(reduction='mean')

        idx = [i for i in range(len(dataset))]
        idx_batched = [idx[i: i + batch_size] for i in range(0, len(idx), batch_size)]

        model_name = type(self.model._orig_mod).__name__

        if model_name in MLM_PREDICTION_MODELS:
            task = 'mlm'
        elif model_name in LM_PREDICTION_MODELS:
            task = 'lm'
        else:
            raise ValueError(f'Model {model_name} not in {MLM_PREDICTION_MODELS, LM_PREDICTION_MODELS}.')

        predictions = []
        targets = []

        for batch in range(len(idx_batched)):
            idx = torch.Tensor(idx_batched[batch]).to(torch.long)
            inputs = self.tokenizer.get_inputs(
                dataset=dataset, task=task, batch_size=self.batch_size, device=self.device, idx=idx)
            with self.ctx:
                outputs = self.model(
                    task=task, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                    labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])

            predictions.extend(dataset.undo_target_transform(outputs['prediction'].cpu()))
            targets.extend(dataset.undo_target_transform(inputs['target'].cpu()))

        predictions = torch.Tensor(predictions)
        targets = torch.Tensor(targets)

        out['MSE'] = loss_mse(predictions, targets).item()
        out['RMSE'] = torch.sqrt(loss_mse(predictions, targets)).item()
        out['MAE'] = loss_l1(predictions, targets).item()
        out['y_pred'] = [item.item() for item in predictions]
        out['y_true'] = [item.item() for item in targets]

        if self.master_process and self.logger is not None:
            self.logger.log({"test/MSE": out['MSE'], "test/RMSE": out['RMSE'], "test/MAE": out['MAE']})

        return out

    @staticmethod
    def get_task(task_p: float):
        p = torch.bernoulli(torch.Tensor([task_p]))
        if p == 1:
            return 'lm'
        elif p == 0:
            return 'mlm'
        else:
            raise ValueError("Wrong task!")

    def train(self):
        self._train_init()
        self.logger.init_run()
        self.model.train()

        task = 'lm'
        inputs = self.get_batch(split='train', task=task)  # fetch the very first batch

        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process

        while True:
            self.set_lr(self.iter_num)

            if self.iter_num % self.eval_interval == 0 and self.master_process:
                self.evaluate()

            if self.iter_num == 0 and self.eval_only:
                break

            for micro_step in range(self.gradient_accumulation_steps):
                if self.is_ddp_run:
                    self.model.require_backward_grad_sync = (micro_step == self.gradient_accumulation_steps - 1)
                with self.ctx:
                    outputs = self.model(
                        task=task, input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'],
                        labels=inputs['labels'], target=inputs['target'], eos_mask=inputs['eos_mask'])
                    loss = outputs['loss'] / self.gradient_accumulation_steps
                task = self.get_task(task_p=self.model.task_p)
                inputs = self.get_batch(split='train', task=task)
                self.scaler.scale(loss).backward()

            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if self.iter_num % self.log_interval == 0 and self.master_process:
                lossf = loss.item() * self.gradient_accumulation_steps
                print(f"iter {self.iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms.", end="\r")
            self.iter_num += 1
            local_iter_num += 1

            # termination conditions
            if self.iter_num > self.max_iters:
                break

        if self.is_ddp_run:
            destroy_process_group()

        print("Training finished.")
