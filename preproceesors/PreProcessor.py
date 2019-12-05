from abc import ABC,abstractmethod
class PreProcessor(ABC):
    @abstractmethod
    def process(self,word,pos_tag):
        pass