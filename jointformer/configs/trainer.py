import math

from jointformer.configs.base import Config


class TrainerConfig(Config):

    def __init__(
        self,
        compile,
        enable_ddp,
        gradient_accumulation_steps,
        batch_size,
        block_size,
        dtype,
        weight_decay,
        learning_rate,
        beta1,
        beta2,
        grad_clip,
        eval_iters,
        warmup_iters,
        lr_decay_iters,
        min_lr,
        decay_lr,
        always_save_checkpoint,
        save_checkpoint,
        save_checkpoint_every,
        eval_only,
        eval_interval,
        log_interval,
        max_iters,
        max_epochs,
        tasks
    ):
        super().__init__()

        # runtime
        self.compile = compile
        self.enable_ddp = enable_ddp
        self.dtype = dtype
        self.eval_only = eval_only
        self.max_iters = max_iters
        self.max_epochs = max_epochs

        # optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip = grad_clip

        # scheduler
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr

        # evaluation & logging
        self.eval_iters = eval_iters
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.always_save_checkpoint = always_save_checkpoint
        self.save_checkpoint_every = save_checkpoint_every
        self.save_checkpoint = save_checkpoint

        # others
        self.block_size = block_size
        self.tasks = tasks
        self._post_init()

    def _post_init(self):
        self._normalize_task_probabilities()

    def _normalize_task_probabilities(self):
        total = sum(self.tasks.values())
        for task in self.tasks:
            self.tasks[task] = self.tasks[task] / total

    def correct_for_num_train_examples(self, num_train_examples: int):
        if self.max_epochs is not None:
            num_iters_single_epoch = math.ceil(num_train_examples / self.batch_size)
            self.max_iters = num_iters_single_epoch * self.max_epochs
            self.warmup_iters = 0.1 * self.max_iters
            self.lr_decay_iters = self.max_iters
            self.eval_interval = num_iters_single_epoch
            self.log_interval = min(self.log_interval, self.eval_intervl)
        else:
            raise ValueError("Argument `max epochs` not specified in config file.")