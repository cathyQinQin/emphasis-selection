from utils.io import read_data
from models import model_factory
import argparse

def parse_args():
    parser =  argparse.ArgumentParser(description="Train the emphasis selection model")
    parser.add_argument("-m","--model",action="store",choices=["base","topk"],default="topk", help="Type of model to be trained, default is TopK model")
    parser.add_argument("-i","--input",default="input/ref/train.txt",help="Input dataset for train model, default is input/ref/train.txt")
    parser.add_argument("-l","--lemmatizer",action="store_true",help="Apply lemmatizer to terms")
    parser.add_argument("-s","--stemmer",action="store_true",help="Apply stem to terms")
    return parser.parse_args()
def main():
    args = parse_args()
    model = model_factory(args)
    _, post_lsts, _, _, e_freq_lsts, pos_lsts = read_data(args.input)
    model.train(post_lsts,pos_lsts,e_freq_lsts)
    model.save()
    print("The model has been trained successfully")
if __name__ == '__main__':
    main()