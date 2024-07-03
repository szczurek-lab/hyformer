import os
import argparse

from jointformer.utils.runtime import set_seed
from experiments.joint_training.train import DEFAULT_SEED_ARRAY


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["guacamol", "moses"], required=True, help="Benchmark suite name")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--results_file_name", nargs='?', type=str, help="Results file name")
    parser.add_argument("--generated_file_path", type=str, required=True, help="Path to the generated data file")
    parser.add_argument("--reference_file_path", type=str, nargs='?', help="Path to the train data file")
    parser.add_argument("--seed", type=int, nargs='*', default=DEFAULT_SEED_ARRAY, help="Random seed")
    args = parser.parse_args()
    return args


def main(args):
    if args.benchmark == "guacamol":
        from jointformer.utils.evaluators.guacamol import GuacamolEvaluator
        evaluator = GuacamolEvaluator(
            generated_file_path=args.generated_file_path,
            reference_file_path=args.reference_file_path,
            out_dir=args.out_dir,
            device='cuda',
            seed=args.seed
        )
    elif args.benchmark == "moses":
        from jointformer.utils.evaluators.moses import MosesEvaluator
        evaluator = MosesEvaluator(
            generated_file_path=args.generated_file_path,
            out_dir=args.out_dir,
            device='cuda')
    else:
        raise ValueError(f"Invalid benchmark: {args.benchmark}")

    evaluator.evaluate()
    evaluator.save()


if __name__ == "__main__":
    args = parse_args()
    for seed in args.seed:
        tmp_args = args
        tmp_args.seed = seed
        tmp_args.out_dir = os.path.join(args.out_dir, f"seed_{seed}")
        set_seed(seed)
        main(tmp_args)
