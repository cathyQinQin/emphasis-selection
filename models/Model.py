from abc import ABC,abstractmethod
class Model(ABC):
    def __init__(self,preprocessors=[]):
        self.preprocessors = preprocessors
    
    def word(self,word,pos_tag):
        for p in self.preprocessors:
            word = p.process(p,pos_tag)
        return word

    @abstractmethod
    def train(self,post_lsts, pos_lsts, e_freq_lsts):
        pass

    @abstractmethod
    def predict(self,post_lsts, pos_lsts):
        pass
    
    @abstractmethod
    def save(self,path):
        pass

    @abstractmethod
    def load(self,path):
        pass