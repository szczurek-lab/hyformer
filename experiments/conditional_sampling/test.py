import os, logging, argparse, sys

import torch
import torch.distributed as dist

from socket import gethostname
from torch.distributed.elastic.multiprocessing.errors import record

from jointformer.configs.dataset import DatasetConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.configs.trainer import TrainerConfig
from jointformer.configs.logger import LoggerConfig

from jointformer.utils.datasets.auto import AutoDataset
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from jointformer.utils.loggers.auto import AutoLogger

from jointformer.trainers.trainer_fixed import Trainer

from jointformer.utils.runtime import set_seed, create_output_dir, set_to_dev_mode, log_args, dump_configs
from jointformer.utils.ddp import init_ddp, end_ddp
from torch.distributed import init_process_group, destroy_process_group
from jointformer.utils.data import write_dict_to_file

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.captureWarnings(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default='./results')
    parser.add_argument("--data_dir", type=str, default='./data')
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--path_to_dataset_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_trainer_config", type=str, required=True)
    parser.add_argument("--path_to_logger_config", type=str, nargs='?')
    parser.add_argument("--path_to_model_ckpt", type=str, nargs='?')
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--freeze_weights", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    log_args(args)
    return args


@record
def main(args):    

    # Create output directory
    if not os.path.exists(args.out_dir):
        raise ValueError(f"Output directory {args.out_dir} does not exist.")
    path_to_model_ckpt = os.path.join(args.out_dir, 'ckpt.pt')
        
    # Load configurations
    dataset_config = DatasetConfig.from_config_file(args.path_to_dataset_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    trainer_config = TrainerConfig.from_config_file(args.path_to_trainer_config)
    logger_config = LoggerConfig.from_config_file(args.path_to_logger_config) if args.path_to_logger_config else None
    if args.out_dir is not None:
        dump_configs(args.out_dir, dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the out_dir

    # Initialize
    test_dataset = AutoDataset.from_config(dataset_config, split='test', root=args.data_dir)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)    
    model = AutoModel.from_config(model_config, downstream_task=dataset_config.task_type, num_tasks=dataset_config.num_tasks)
    logger = AutoLogger.from_config(logger_config) if logger_config else None
    if logger is not None:
        logger.store_configs(dataset_config, tokenizer_config, model_config, trainer_config, logger_config) # Store configs, within the logger object

    # Run training
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    trainer = Trainer(
        out_dir=args.out_dir,
        seed=args.seed,
        device=device,
        config=trainer_config,
        model=model,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        logger=logger,
        test_metric=dataset_config.task_metric
        )

    trainer._init_data_loaders()
    print(f"Loading model from {path_to_model_ckpt}")
    trainer.model.load_state_dict(torch.load(path_to_model_ckpt, map_location=device)['model'], strict=True)

    test_metric = dataset_config.task_metric
    objective_metric = trainer.test(metric=test_metric)
    print(f"Test {test_metric}: {objective_metric}")
    if args.destroy_ckpt and os.path.exists(path_to_model_ckpt):
        os.remove(path_to_model_ckpt)
    
    write_dict_to_file({f'{test_metric}': str(objective_metric)}, os.path.join(args.out_dir, 'test_loss.json'))
    

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    args = parse_args()
    set_seed(args.seed)
    main(args)
