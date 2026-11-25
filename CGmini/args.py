import argparse


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="gpu number or cpu")
    parser.add_argument("--path", type=str, default="./data/", help="Dataset path")
    parser.add_argument("--dataset", type=str, default="yelp", help="Dataset name")
    parser.add_argument("--seed", type=int, default=76870, help="Random seed")
    parser.add_argument("--fix_seed", action='store_true', default=True, help="xx")
    parser.add_argument("--normal", action='store_true', default=True, help="xx")

    parser.add_argument("--n_repeated", type=int, default=1, help="Repeated times")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")

    parser.add_argument("--ratio", type=float, default=0.2, help="Label ratio")
    parser.add_argument("--num_epoch", type=int, default=300, help="Training epochs")

    parser.add_argument("--edge_rate", type=float, default=0.6, help="distance mask")
    parser.add_argument("--alpha", type=float, default=1, help="att_z alpha")
    parser.add_argument("--beta", type=float, default=1, help="att_z beta")

    parser.add_argument("--fai", type=float, default=0.6, help="loss")
    parser.add_argument("--tau", type=float, default=0.1, help="soft label")
    parser.add_argument("--hidden_dims", type=int, default=64, help="hidden dimensions")

    args = parser.parse_args()

    return args
