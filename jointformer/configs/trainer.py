import torch
from transformers import PretrainedConfig


class TrainerConfig(PretrainedConfig):

    def __init__(
        self,
        out_dir: str = './results/',
        eval_interval: int = 2000,
        log_interval: int = 1,
        eval_iters: int = 200,
        eval_only: bool = False,
        always_save_checkpoint: bool = False,
        init_from: str = 'scratch',
        learning_rate: float = 6e-4,
        max_iters: int = 600000,
        weight_decay: float = 1e-1,
        beta1: float = 0.9,
        beta2: float = 0.95,
        grad_clip: float = 1.0,
        decay_lr: bool = True,
        warmup_iters: int = 2000,
        lr_decay_iters: int = 600000,
        min_lr: float = 6e-5,
        ddp_enabled: bool = True,
        ddp_backend: str = 'nccl',
        gradient_accumulation_steps: int = 5 * 8,
        batch_size: int = 12,
        device: str = 'cuda',
        dtype: str = 'float32',
        compile: bool = True,
        resume: bool = True,
        wandb_enabled: bool = True,
        wandb_project: str = 'jointformer',
        wandb_run: str = 'jointformer',
        **kwargs,
    ):

        # Out dir
        self.out_dir = out_dir
        self.resume = resume  # resume from last checkpoint

        # eval
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.eval_iters = eval_iters
        self.eval_only = eval_only  # if True, script exits right after the first eval

        # load / save
        self.always_save_checkpoint = always_save_checkpoint  # if True, always save a checkpoint after each eval
        self.init_from = init_from  # 'scratch' or 'resume' or 'gpt2*'

        # adamw optimizer
        self.learning_rate = learning_rate  # max learning rate
        self.max_iters = max_iters  # total number of training iterations
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip  # clip gradients at this value, or disable if == 0.0

        # learning rate decay settings
        self.decay_lr = decay_lr  # whether to decay the learning rate
        self.warmup_iters = warmup_iters  # how many steps to warm up for
        self.lr_decay_iters = lr_decay_iters  # should be ~= max_iters per Chinchilla
        self.min_lr = min_lr  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

        # DDP settings
        self.ddp_enabled = ddp_enabled
        self.ddp_backend = ddp_backend  # 'nccl', 'gloo', etc.

        # runtime
        self.gradient_accumulation_steps = gradient_accumulation_steps  # used to simulate larger batch sizes
        self.batch_size = batch_size  # if gradient_accumulation_steps > 1, this is the micro-batch size
        self.device = device  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
        if dtype == 'auto':
            dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
        self.dtype = dtype  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        self.compile = compile  # use PyTorch 2.0 to compile the model to be faster
        self.wandb_enabled = wandb_enabled
        self.wandb_project = wandb_project
        self.wandb_run = wandb_run
        super().__init__(**kwargs)

    def save(self, save_directory: str) -> None:
        super().save_pretrained(save_directory=save_directory)

    def load(self, config_path: str) -> None:
        super().from_pretrained(pretrained_model_name_or_path=config_path)

