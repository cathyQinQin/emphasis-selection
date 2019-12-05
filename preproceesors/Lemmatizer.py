from nltk.stem import WordNetLemmatizer
from preproceesors.PreProcessor import PreProcessor

class Lemmatizer(PreProcessor):
    def __init__(self,):
        self._name = "lemmatizer"
        self.wnl = WordNetLemmatizer()

    def process(self,word,pos_tag):
        return self.wnl.lemmatize(word,pos_tag)