from abc import ABC,abstractmethod,property
class PreProcessor(ABC):
    def __init__(self):
        self._name = None

    @abstractmethod
    def process(self,word,pos_tag):
        pass

    @property
    def name(self):
        return self._name