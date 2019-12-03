import unittest
from preproceesors.Stemmer import Stemmer

class TestLemmatizer(unittest.TestCase):
    def setUp(self):
        self.p = Stemmer()
        return super().setUp()

    def test_words(self):
        self.assertEqual(self.p.process("having"),"have")
        self.assertEqual(self.p.process("generously"),"generous")
        self.assertEqual(self.p.process("running"),"run")