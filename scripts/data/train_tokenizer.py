import argparse
import os 

import sentencepiece as spm

from jointformer.utils.data_utils.utils import SMILESTokenizer, create_vocabulary, load_strings_from_txt


def parse_args():
    parser = argparse.ArgumentParser(description="Make tokenizer")
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--vocab_file_path", type=str, default=None, help="Path to the data file of rare substructures")
    parser.add_argument("--output_dir", type=str, help="Path to the tokenizer output directory")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "char"], default="bpe", help="Encoding type")
    parser.add_argument("--tokenizer_prefix", type=str, default="smiles_tokenizer", help="Tokenizer prefix")
    return parser.parse_args()


def main(args):
    
    # Create the output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Extract tokens from the SMILES strings
    if args.vocab_file_path is not None:
        smiles = load_strings_from_txt(args.vocab_file_path)
        tokenizer = SMILESTokenizer()
        vocabulary = create_vocabulary(smiles, tokenizer)
        tokens = vocabulary.tokens()    
    else:
        tokens = None    

    # Create the tokenizer
    spm.SentencePieceTrainer.train(
        input=args.data_path,
        model_prefix=os.path.join(args.output_dir, args.tokenizer_prefix),
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        model_type=args.tokenizer_type,
        user_defined_symbols=tokens,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
    ) 
    
    return None

if __name__ == "__main__":
    args = parse_args()
    main(args)
