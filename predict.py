from utils.io import read_data,write_results
from models import model_factory
import argparse
import os
from pathlib import Path

def parse_args():
    parser =  argparse.ArgumentParser(description="Predict emphasis selectijon with trained model")
    parser.add_argument("-m","--model",action="store",choices=["base","topk"],default="topk", help="Type of model to be trained, default is TopK model")
    parser.add_argument("-i","--input",default="input/ref/gold.txt",help="Input dataset for train model, default is input/ref/gold.txt")
    parser.add_argument("-o","--output",default="input/res/submission.txt",help="Input dataset for train model, default is input/res/submission.txt")
    parser.add_argument("-l","--lemmatizer",action="store_true",help="Apply lemmatizer to terms")
    parser.add_argument("-s","--stemmer",action="store_true",help="Apply stemmer to terms")
    parser.add_argument("-k",type=int,default=10,help="K value used by TopK model, default is 10")
    return parser.parse_args()
def main():
    args = parse_args()
    model = model_factory(args)
    word_id_lst, post_lsts, _, _, _, pos_lsts = read_data(args.input)
    freqs_lst = []

    odir = Path(args.output).parent
    if not odir.is_dir():
        os.mkdir(odir)

    for post,pos_tags  in zip(post_lsts,pos_lsts):
        freqs_lst.append(model.predict(post,pos_tags))
    write_results(word_id_lst,post_lsts,freqs_lst,args.output)
    print("Output file created successfully")
if __name__ == '__main__':
    main()