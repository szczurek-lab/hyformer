""" Evaluate the Guacamol distribution learning task. """

import os, logging, argparse, sys
import torch

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig

from hyformer.utils.tokenizers.auto import AutoTokenizer
from hyformer.models.auto import AutoModel

from hyformer.utils.experiments import log_args
from hyformer.utils.reproducibility import set_seed

from guacamol.assess_distribution_learning import assess_distribution_learning

console = logging.getLogger(__file__)
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format='%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
# logging.captureWarnings(False)


def main(args):    
    
    # assert that the reference file exists
    assert os.path.exists(args.chembl_training_file), f"Reference file {args.chembl_training_file} does not exist"

    # Load configurations
    tokenizer_config = TokenizerConfig.from_config_filepath(args.tokenizer_config_path)
    model_config = ModelConfig.from_config_filepath(args.model_config_path)
    
    # Initialize
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model = AutoModel.from_config(model_config)
    
    # Load model checkpoint
    model.load_pretrained(filepath=args.model_ckpt_path, discard_prediction_head=True)
    model = model.to_generator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_sequence_length=args.max_sequence_length,
        device=args.device
    )
    
    assess_distribution_learning(model, args.chembl_training_file, args.output_filepath)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-filepath", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--experiment-seed", type=int, default=0, help="Seed for the experiment")
    parser.add_argument("--tokenizer-config-path", type=str, required=True, help="Path to the tokenizer config file")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--model-ckpt-path", type=str, nargs='?', help="Path to the model checkpoint file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")
    parser.add_argument("--top-k", type=int, nargs='?', help="Top-k for the model")
    parser.add_argument("--top-p", type=float, nargs='?', help="Top-p for the model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for the model")
    parser.add_argument("--max-sequence-length", type=int, default=100, help="Maximum sequence length for the model")
    parser.add_argument("--chembl-training-file", type=str, required=True, help="Path to the ChEMBL training file")
    args = parser.parse_args()
    log_args(args)
    return args

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    args = parse_args()
    set_seed(args.experiment_seed)
    main(args)
    