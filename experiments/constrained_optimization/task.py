import argparse
import os

from experiments.constrained_optimization.utils import get_experiment_dir_path


def parse_args():
    parser = argparse.ArgumentParser(description='Constrained optimization experiment')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output directory')
    parser.add_argument('--objective', type=str, nargs='+', help='name of the objective')
    parser.add_argument('--backbone', type=str, nargs='+', help='name of the backbone model')
    parser.add_argument('--method', type=str, nargs='+', help='name of the method')
    parser.add_argument('--similarity_constraint', type=float, nargs='+', help='similarity constraint threshold')
    parser.add_argument('--reference_file_dir', type=str, required=True, help='path to initial data')
    parser.add_argument('--debug', type=bool, required=True, help='debug mode')
    return parser.parse_args()


def run_constrained_optimization_task(
        path: str, objective: str, backbone: str, method: str,
        similarity_constraint: float, reference_file_dir: str, debug: bool) -> None:

    # load reference file
    reference_file = os.path.join(reference_file_dir, f'{objective}.csv')
    # load data
    data = load_data(reference_file)
    # run optimization
    run_optimization(data, path, objective, backbone, method, similarity_constraint, debug)


def main(args: argparse.Namespace) -> None:
    for objective in args.objective:
        for backbone in args.backbone:
            for method in args.method:
                for similarity_constraint in args.similarity_constraint:
                    path = get_experiment_dir_path(
                        args.out_dir, objective, backbone, method, similarity_constraint)
                    path = os.path.join(path, 'debug') if args.debug else path
                    os.makedirs(path, exist_ok=False)
                    run_constrained_optimization_task(
                        path, objective, backbone, method, similarity_constraint, args.reference_file_dir, args.debug)


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
