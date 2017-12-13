from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from Model import Model


class LeNetClassifier(Model):
    def __init__(self, input_shape: tuple=(32, 32, 3), output_shape: tuple=10):
        """
        Creates a LeNet model
        :param input_shape:
        :param output_shape:
        """
        # Initialize the base class
        super().__init__(input_shape, output_shape)

        # Form the model
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                              input_shape=self.input_shape))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.model.add(MaxPool2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(self.output_shape, activation='softmax'))
        self.model.summary()

    def compile(self, optimizer: str, loss: str, metrics: list):
        """
        Compile the model
        :return:
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)

    def train(self, x, y, x_val, y_val, batch_size, epochs):
        # Call the base class
        super().train(x, y, x_val, y_val, batch_size, epochs)

        # Train
        self.model.fit(x, y,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_val, y_val),
                       shuffle=True,
                       verbose=2)


if __name__ == "__main__":
    # Constants
    batch_size = 32
    num_classes = 10
    epochs = 100

    # Get the data
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices.
    import keras
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    lenet = LeNet()
    lenet.compile('adam', 'categorical_crossentropy', ['acc'])
    lenet.train(x_train, y_train, x_test, y_test, batch_size, epochs)
