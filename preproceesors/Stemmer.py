from nltk.stem import SnowballStemmer
from preproceesors.PreProcessor import PreProcessor

class Stemmer(PreProcessor):
    def __init__(self,):
        self.stm = SnowballStemmer("english")

    def process(self,word,pos_tag=None):
        return self.stm.stem(word)