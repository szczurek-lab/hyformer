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
        grad_clip
    ):
        super().__init__()
        self.compile = compile
        self.enable_ddp = enable_ddp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.block_size = block_size
        self.dtype = dtype
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.grad_clip = grad_clip
