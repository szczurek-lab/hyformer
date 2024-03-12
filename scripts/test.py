import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', nargs='?', type=float)
    args = parser.parse_args()
    print(args.number)
    if args.number is None:
        print("AMAZING!")


if __name__ == "__main__":
    main()
