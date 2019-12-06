from test.test_basemodel import TestBaseModel
from models.TopKModel import TopKModel

class TestTopkModel(TestBaseModel):
    def setUp(self):
        self.model = TopKModel
        self.name = "topk"
        self.trained = None