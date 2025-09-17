""" Evaluate the Guacamol distribution learning task. """

import os, logging, argparse, sys
import torch
import numpy as np

from hyformer.configs.tokenizer import TokenizerConfig
from hyformer.configs.model import ModelConfig

from hyformer.tokenizers.auto import AutoTokenizer
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

    # Load configurations
    tokenizer_config = TokenizerConfig.from_config_filepath(args.tokenizer_config_path) if args.tokenizer_config_path is not None else None
    model_config = ModelConfig.from_config_filepath(args.model_config_path)
    
    # Initialize
    tokenizer = AutoTokenizer.from_config(tokenizer_config) if tokenizer_config is not None else None
    model = AutoModel.from_config(model_config)
    
    # Load model checkpoint
    model = model.to_generator(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_sequence_length=args.max_sequence_length,
        device=args.device,
        use_cache=args.use_cache
    )
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    record_iterations = 100
    times = []
    for idx in range(record_iterations):
        start_time.record()
        samples = model.generate(args.batch_size)
        end_time.record()
        torch.cuda.synchronize()
        times.append(start_time.elapsed_time(end_time)/args.batch_size)

    print(f"Number of samples generated: {len(samples)}")
    print(samples[0])
    print(samples[1])
    print(samples[2])
    print(samples[3])
    print(samples[4])
    print(samples[5])
    times = np.array(times[10:-10])
    print(f"Average time per sample: {np.mean(times)} ms")
    print(f"Std time per sample: {np.std(times)} ms")
    print(f"Min time per sample: {np.min(times)} ms")
    print(f"Max time per sample: {np.max(times)} ms")
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-config-path", type=str, nargs='?', help="Path to the tokenizer config file")
    parser.add_argument("--model-config-path", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the model")
    parser.add_argument("--top-k", type=int, nargs='?', help="Top-k for the model")
    parser.add_argument("--top-p", type=float, nargs='?', help="Top-p for the model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device for the model")
    parser.add_argument("--max-sequence-length", type=int, default=100, help="Maximum sequence length for the model")
    args = parser.parse_args()
    log_args(args)
    return args

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == torch.cuda.device_count(), "Number of GPUs on node does not match SLURM_GPUS_ON_NODE"
    print("Torch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    args = parse_args()
    main(args)
    