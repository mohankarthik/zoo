from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self,
                 input_shape: tuple=(32, 32, 3),
                 output_shape: tuple=10):
        # Initialize the variables
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        super().__init__()

    @abstractmethod
    def compile(self, optimizer: str, loss: str, metrics: list):
        pass

    @abstractmethod
    def train(self, x, y, x_val, y_val, batch_size, epochs):
        assert x.shape[1:] == self.input_shape
        assert x_val.shape[1:] == self.input_shape
