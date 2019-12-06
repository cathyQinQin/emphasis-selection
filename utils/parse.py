import argparse
def trainParser():
    parser = argparse.ArgumentParser(description='Process train parameters.')
    parser.add_argument('dataset', metavar='trainFile', type=str, nargs=1,
                        help='an dataset file for training')
    parser.add_argument('model', metavar='modelType', type=str, nargs=1,
                        help='an model for training [topk|freq]')
    parser.add_argument('-l', metavar='l', type=str, nargs='?',
                        help='a l')
    parser.add_argument('-s', metavar='s', type=str, nargs='?',
                        help='a s')
    args = parser.parse_args()
    print(args.dataset)
trainParser()