from models.base_model import BaseModel
from models.topk_model import TopKModel
from preprocessors import Lemmatizer,Stemmer

def model_factory(args):
    p = []
    if args.lemmatizer:
        print("Lemmatizer applied")
        p.append(Lemmatizer())
    if args.stemmer:
        print("Stemmer applied")
        p.append(Stemmer())
    if args.model == "topk":
        print("TopK model selected")
        return TopKModel(p)
    elif args.model == "base":
        print("Base model selected")
        return BaseModel(p)