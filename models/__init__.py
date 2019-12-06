from models.base_model import BaseModel
from models.topk_model import TopKModel
from preprocessors import Lemmatizer,Stemmer

def model_factory(args):
    p = []
    if args.lemmatizer:
        p.append(Lemmatizer())
    if args.stemmer:
        p.append(Stemmer())

    if args.model == "topk":
        return TopKModel(p)
    elif args.model == "base":
        return BaseModel(p)