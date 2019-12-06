import unittest
from models import TopKModel
from models import BaseModel
from utils.io import read_data

class TestBaseModel(unittest.TestCase):
    def setUp(self):
        self.model = BaseModel
        self.name = "base"
        self.trained = None
    
    def train(self):
        if self.trained:
            return self.trained
        _, post_lsts, _, _, e_freq_lsts, pos_lsts = read_data("input/ref/dev.txt")
        model = self.model()
        model.train(post_lsts,pos_lsts,e_freq_lsts)
        self.trained = model
        return model

    def test_name(self):
        model = self.model()
        self.assertEqual(model.name,self.name)

    def test_train(self):
        model = self.train()
        self.assertTrue("mocktails" in model.word_counter)
        self.assertAlmostEqual(model.word_freqs["mocktails"],0.7777777777777778)

    def test_freq(self):
        model = self.train()   
        self.assertTrue(model.freq("mocktails","NNPS") > 0.8)
        self.assertTrue(model.freq("is","VBZ") < 0.2)
        self.assertEqual(model.freq("unittest","NN"),-1)


    def test_predict(self):
        model = self.train()
        post = ["What","is","once","well","done","is","done","forever"]
        pos_tags = ["WP","VBZ","RB","RB","VBN","VBZ","VBN","RB"]
        freqs = model.predict(post,pos_tags)
        self.assertTrue(max(freqs) == freqs[-1])

        post = ["What","is","once","well","done","is","done","forever"]
        pos_tags = ["WP","VBZ","RB","RB","VBN","VBZ","VBN","RB"]
        freqs = model.predict(post,pos_tags)
        self.assertTrue(max(freqs) == freqs[-1])

        post = ["All","about","unittest"]
        pos_tags = ["DT","IN","NNPS"]
        freqs = model.predict(post,pos_tags)
        self.assertTrue(max(freqs) == freqs[-1])
    
    def test_saveload(self):
        model1 = self.train()
        model1.save()
        model2 = self.model()
        model2.load()
        self.assertDictEqual(model1.word_freqs,model2.word_freqs)

class TestTopkModel(TestBaseModel):
    def setUp(self):
        self.model = TopKModel
        self.name = "topk"
        self.trained = None