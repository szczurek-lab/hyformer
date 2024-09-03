from argparse import ArgumentParser
from jointformer.configs.task import TaskConfig
from jointformer.configs.tokenizer import TokenizerConfig
from jointformer.configs.model import ModelConfig

from jointformer.utils.tokenizers.auto import AutoTokenizer
from jointformer.models.auto import AutoModel
from guacamol.assess_distribution_learning import assess_distribution_learning
import logging
import sys

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--path_to_model_ckpt", type=str, required=True)
    parser.add_argument("--path_to_model_config", type=str, required=True)
    parser.add_argument("--path_to_tokenizer_config", type=str, required=True)
    parser.add_argument("--chembl_training_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top_k", type=int, required=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser


def main(args):

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    model_config = ModelConfig.from_config_file(args.path_to_model_config)
    tokenizer_config = TokenizerConfig.from_config_file(args.path_to_tokenizer_config)
    model = AutoModel.from_config(model_config)
    tokenizer = AutoTokenizer.from_config(tokenizer_config)
    model.load_pretrained(args.path_to_model_ckpt)
    model = model.to_guacamole_generator(tokenizer, args.batch_size, args.temperature, args.top_k, args.device)
    assess_distribution_learning(model, args.chembl_training_file, args.output)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)