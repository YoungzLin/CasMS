#encoding:utf-8
import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run CasSeqGCN.")

    parser.add_argument('--graph-folder',
                        nargs='?',
                        default='',
                        help='Folder with graph pair jsons.')
    parser.add_argument('--result-log',
                        type=str,
                        default='',
                        help='save results file')
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='batch-size')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.001,
                        help='')
    parser.add_argument('--number-of-nodes',
                        type=int,
                        default=200,
                        help='')
    parser.add_argument('--sub_size',
                        type=int,
                        default=3,
                        help='')
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.001,
                        help="Adam weight decay. Default is 0.001.")
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help="Number of training epochs. Default is 50.")
    parser.add_argument('--number-of-hand-features',
                        type=int,
                        default=2,
                        help="")
    parser.add_argument('--train-ratio',
                        type=float,
                        default=0.7,
                        help="")
    parser.add_argument('--valid-ratio',
                        type=float,
                        default=0.15,
                        help="")
    parser.add_argument('--test-ratio',
                        type=float,
                        default=0.15,
                        help="")
    parser.add_argument('--check-point',
                        type=int,
                        default=5,
                        help="")
    parser.add_argument('--gcn-out-channel',
                        type=int,
                        default=32,
                        help="gcn out nodes feature size")
    parser.add_argument('--gcn-filters-1',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--gcn-filters-2',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--capsule-out-features',
                        type=int,
                        default=4,
                        help="")
    parser.add_argument('--capsule-out-dim',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--tcn-hiddensize',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--dens-hiddensize',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--tcn-layers',
                        type=int,
                        default=2,
                        help="")
    parser.add_argument('--tcn-dropout',
                        type=float,
                        default=0.4,
                        help="")
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.1,
                        help="")
    parser.add_argument('--dens-dropout',
                        type=float,
                        default=0.5,
                        help="")
    parser.add_argument('--dens-outsize',
                        type=int,
                        default=1,
                        help="")

    return parser.parse_args()
