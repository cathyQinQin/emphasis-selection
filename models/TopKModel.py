from models.BaseModel import BaseModel
from gensim.models import word2vec


class TopKModel(BaseModel):
    def __init__(self, preprocessors=[]):
        super().__init__(preprocessors=preprocessors)
        self._name = "topk"
        self.model = None

    def _train(self,corpus,pos_lsts,e_freq_lsts):
        self.w2v = word2vec.Word2Vec(corpus, min_count=1)

    def _predict(self,words,pos_tags,freqs,unseen):
        if len(unseen) == 0:
            return freqs
        if not self.model:
            return super()._predict(words,pos_tags,freqs,unseen)
        self.model.train([words])
        for i in unseen:
            sum_similarity = 0
            sum_freq = 0 
            for neighbor,similarity in self.model.most_similar(words[i]):
                freq = self.freq(neighbor)
                if not freq < 0:
                    sum_similarity += similarity
                    sum_freq += freq * similarity
            freqs[i] = sum_freq / sum_similarity if sum_similarity > 0 else 0
        self.train_sentence(words,pos_tags,freqs)
        return freqs

    def _save(self,path):
        self.model.save(path + ".w2v")

    def _load(self,path):
        self.model = word2vec.Word2Vec.load(path + ".w2v")