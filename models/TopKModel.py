from models.Model import Model
from gensim.models import word2vec
from collections import Counter 

class TopKModel(Model):
    def __init__(self, preprocessors=[]):
        super().__init__(preprocessors=preprocessors)
        self._name = "tokp"

    def train(self,post_lsts, pos_lsts, e_freq_lsts):
        self.counter = Counter()
        self.freqs = {}
        posts = []
        for posts,pos_tags,freqs in zip(post_lsts, pos_lsts, e_freq_lsts):
            words = []
            for word,pos_tag,freq in zip(posts,pos_tags,freqs):
                word = self.word(word,pos_tag)
                words.append(word)

                self.counter[word] += 1
                self.freqs[word] = self.freqs.get(word,0) + float(freq)
            posts.append(words)
        self.model = word2vec.Word2Vec(posts, min_count=1)

    def predict(self, post_lsts, pos_lsts):
        pass

    def _save(self,path):
        name = "topk"
        self.model.save(path + "/w2v.model")


    def load(self,path):
        pass