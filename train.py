# train.py dataset --model topk|freq... [-l -s] --help

from utils.io import read_data
from models.TopKModel import TopKModel

word_id_lsts, post_lsts, bio_lsts, freq_lsts, e_freq_lsts, pos_lsts = read_data("input/res/train.txt")
model = TopKModel()
model.train(post_lsts,pos_lsts,e_freq_lsts)