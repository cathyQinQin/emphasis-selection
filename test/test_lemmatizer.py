import unittest
from preprocessors import Lemmatizer

class TestLemmatizer(unittest.TestCase):
    def setUp(self):
        self.p = Lemmatizer()

    def test_name(self):
        self.assertEqual(self.p.name,"lemmatizer")

    def test_nouns(self):
        self.assertEqual(self.p.process("cars","n"),"car")
        self.assertEqual(self.p.process("women","n"),"woman")

    def test_verbs(self):
        self.assertEqual(self.p.process("running","v"),"run")
        self.assertEqual(self.p.process("ate","v"),"eat")

    def test_adjectives(self):
        self.assertEqual(self.p.process("saddest","a"),"sad")
        self.assertEqual(self.p.process("fancier","a"),"fancy")