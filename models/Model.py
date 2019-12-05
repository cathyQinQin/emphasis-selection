from abc import ABC,abstractmethod,property
class Model(ABC):
    def __init__(self,preprocessors=[]):
        self._name = None
        self.preprocessors = preprocessors
    
    def word(self,word,pos_tag):
        for p in self.preprocessors:
            word = p.process(p,pos_tag)
        return word

    def save(self):
        name = self.name
        for p in self.preprocessors:
            name += '.' + p.name[0]
        self._save("/saved/" + name)

    @property
    def name(self):
        return self._name

    @abstractmethod
    def train(self,post_lsts, pos_lsts, e_freq_lsts):
        pass

    @abstractmethod
    def predict(self,post_lsts, pos_lsts):
        pass
    
    @abstractmethod
    def _save(self,name):
        pass

    @abstractmethod
    def load(self):
        pass