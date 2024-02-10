import tensorflow as tf
import tensorflow_hub as hub


class STSPredictor:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer(model_path)