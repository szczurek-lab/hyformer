from argparse import ArgumentParser
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig
from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from guacamol.assess_distribution_learning import assess_distribution_learning
import logging
import os
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--chembl_training_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=25)
    return parser

def main(args):
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    model = AutoModel.from_config(model_config)

    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    
    model.load_pretrained(args.path_to_model_ckpt)
    model = model.to_guacamole_generator(tokenizer, args.batch_size, args.temperature, args.top_k, args.device)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    assess_distribution_learning(model, args.chembl_training_file, args.output)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)