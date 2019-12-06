from nltk.stem import WordNetLemmatizer
from preprocessors.preprocessor import PreProcessor

class Lemmatizer(PreProcessor):
    def __init__(self,):
        self._name = "lemmatizer"
        self.wnl = WordNetLemmatizer()

    def process(self,word,pos_tag):
        try:
            return self.wnl.lemmatize(word,pos_tag.lower()[0])
        except KeyError:
            return word