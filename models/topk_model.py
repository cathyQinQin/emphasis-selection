from models import BaseModel
from gensim.models import word2vec


class TopKModel(BaseModel):
    def __init__(self, preprocessors=[],k=10):
        super().__init__(preprocessors=preprocessors)
        self._name = "topk"
        self.k = k
        self.wv_model = None
        self.warning = True

    def _train(self,corpus,pos_lsts,e_freq_lsts):
        self.wv_model = word2vec.Word2Vec(corpus, min_count=1)
    

    def _predict(self,words,pos_tags,freqs,unseen):
        if not self.wv_model:
            if self.warning:
                self.warning = False
                print("No word vectors, fallback to use base model")
            return super()._predict(words,pos_tags,freqs,unseen)
        self.wv_model.build_vocab([words], update=True)
        self.wv_model.train([words], total_examples=self.wv_model.corpus_count, epochs = self.wv_model.epochs)
        for i in range(len(words)):
            sum_similarity = 0
            sum_freq = 0 
            for neighbor,similarity in self.wv_model.wv.most_similar(words[i],topn=self.k):
                freq = self.freq(neighbor)
                if not freq < 0:
                    sum_similarity += similarity
                    sum_freq += freq * similarity
            freq = sum_freq / sum_similarity if sum_similarity > 0 else 0
            if i in unseen:
                freqs[i] = freq
            else:
                freqs[i] = 0.8 * freqs[i] + 0.2 * freq
        self.train_sentence(words,pos_tags,freqs)
        return freqs

    def _save(self,path):
        self.wv_model.save(path + ".w2v")

    def _load(self,path):
        self.wv_model = word2vec.Word2Vec.load(path + ".w2v")