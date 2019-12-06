import unittest
from preproceesors.Stemmer import Stemmer

class TestLemmatizer(unittest.TestCase):
    def setUp(self):
        self.p = Stemmer()

    def test_name(self):
        self.assertEqual(self.p.name,"stemmer")

    def test_words(self):
        self.assertEqual(self.p.process("having"),"have")
        self.assertEqual(self.p.process("generously"),"generous")
        self.assertEqual(self.p.process("running"),"run")