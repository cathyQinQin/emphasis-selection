from collections import Counter 
import pickle
from pathlib import Path
import os

class BaseModel:
    def __init__(self,preprocessors=[]):
        self._name = "base"
        self.preprocessors = preprocessors
        self.word_counter = Counter()
        self.word_freqs = {}
        self.pos_counter = Counter()
        self.pos_freqs = {}

    @property
    def name(self):
        name = self._name
        for p in self.preprocessors:
            name += '.' + p.name[0]
        return name
       
    def word(self,word,pos_tag):
        word = word.lower()
        for p in self.preprocessors:
            word = p.process(word,pos_tag)
        return word
    
    def freq(self,word,pos_tag = ""):
        if word not in self.word_counter:
            return -1
        wf = self.word_freqs[word]/ self.word_counter[word]
        pf = self.pos_freqs.get(pos_tag,0) / self.pos_counter.get(pos_tag,1)
        return wf * ( pf * abs(wf - pf) + 1)

    def path(self):
        path = Path(__file__).parent.parent.joinpath("saved/")
        if not path.exists():
            os.mkdir(path)
        return path

    def save(self):
        path = str(self.path().joinpath(self.name))
        
        with open(path,"wb+") as f:
            pickle.dump({
                "wc" : self.word_counter,
                "wf" : self.word_freqs,
                "pc" : self.pos_counter,
                "pf" : self.pos_freqs
            },f)
        self._save(path)

    def _save(self,path):
        pass

    def load(self):
        path = self.path().joinpath(self.name)

        if not path.is_file():
            print("Trained model not found! Make sure to provide same preprocessors(stemmer/lemmatizer/both/none) as you run train.py")
            exit()

        with open(path,"rb") as f:
            data = pickle.load(f)
            self.word_counter = data["wc"]
            self.word_freqs = data["wf"]
            self.pos_counter = data["pc"] 
            self.pos_freqs = data["pf"]    
        self._load(str(path))

    def _load(self,path):
        pass


    def train(self,corpus, pos_lsts, e_freq_lsts):
        for words,pos_tags,freqs in zip(corpus, pos_lsts, e_freq_lsts):
            for i,(word,pos_tag,freq) in enumerate(zip(words,pos_tags,freqs)):
                words[i] = self.word(word,pos_tag)
                freqs[i] = float(freq)
            self.train_sentence(words,pos_tags,freqs)
        self._train(corpus, pos_lsts, e_freq_lsts)

    def train_sentence(self,words,pos_tags,freqs):
        for word,pos_tag,freq in zip(words,pos_tags,freqs):
            self.word_counter[word] += 1
            self.word_freqs[word] = self.word_freqs.get(word,0) + freq
            self.pos_counter[pos_tag] += 1
            self.pos_freqs[pos_tag] = self.pos_freqs.get(pos_tag,0) + freq

    def _train(self,corpus,pos_lsts,e_freq_lsts):
        pass

    def predict(self,post,pos_tags):
        words = []
        freqs = []
        unseen = []
        for i,(word,pos_tag) in enumerate(zip(post,pos_tags)):
            word = self.word(word,pos_tag)
            words.append(word)
            freq = self.freq(word,pos_tag)
            freqs.append(freq)
            if freq < 0:
                unseen.append(i)
        return self._predict(words,pos_tags,freqs,unseen)

    def _predict(self,words,pos_tags,freqs,unseen):
        for i in unseen:
            freqs[i] = self.pos_freqs.get(pos_tags[i],0) / self.pos_counter.get(pos_tags[i],1)
        return freqs