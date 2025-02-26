'''
    Tensor Neural Network (Python Class Format)
    Copyright (C) 2025  Quira Walker

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random, os
from tensorflow.keras.layers import Dense, Dropout   # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

class NeuralNetwork:
    def __init__(self):
        self.X = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
        self.Y = np.array(([1],[0],[0],[0],[0],[0],[0],[1]), dtype=float)
        self.model = tf.keras.Sequential()

# Changing the seed for the Model allows you to either create different
# results every time or if the seed is the same your results will be the same.
    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

# Starting Neurons determines the number of neurons in the first layer
# Usually the more complex the problem the more neurons you need to start with
# This current set up is layer divisible by 2, so each subsequent layer will half half the neurons
# Starting Dropout is the percentage of neurons that will be dropped out if they
# Are not giving valid results per epoch. This maintains a clean progression
# Layers is the number of layers you want in your model
# The more layers you need for more complex issues but this comes at a cost of training speed
    def create_model(self, startingNeurons, startingDropout, layers=2):
        self.model.add(Dense(startingNeurons, input_dim=3, activation='relu', use_bias=True)) # Layer: 1
        self.model.add(Dropout(startingDropout))
        if layers > 2:
            for x in range(layers-2):
                if startingNeurons//2 != 2:
                    self.model.add(Dense(startingNeurons//2, activation='relu', use_bias=True))
                elif startingNeurons//2 == 2:
                    self.model.add(Dense(2, activation='relu', use_bias=True))
                self.model.add(Dropout(startingDropout))
        self.model.add(Dense(1, activation='sigmoid', use_bias=True))
        return layers

# Rate is the learning rate, the lower the rate the slower the model learns but the more accurate
# Faster rates will train faster but at the expense of accuracy in certain cases
# Epochs is the number of times the model will run through the data
# Patience is the number of epochs to wait before stopping the model
# This is usefull if your model reaches the desired level of acccuracy earlier then set epochs
    def train_model(self, rate=0.01, epochs=2000, patience=50):
        optimizer = tf.keras.optimizers.Adam(learning_rate=rate)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
        early_stopping = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
        print(self.model.get_weights())
        self.history = self.model.fit(self.X, self.Y, epochs=epochs, validation_split=0.2, callbacks=[early_stopping], validation_data=(self.X, self.Y), verbose=1)
        self.model.summary()
    
    def save_history(self, path):
        self.numpy_loss_history        = np.array(self.history.history['loss'])
        self.numpy_binary_accuracy     = np.array(self.history.history['binary_accuracy'])
        self.numpy_val_binary_accuracy = np.array(self.history.history['val_binary_accuracy'])
        np.savetxt(path+"loss_history.txt",    self.numpy_loss_history,        delimiter="\n")
        np.savetxt(path+"binary_accuracy.txt", self.numpy_binary_accuracy,     delimiter="\n")
        np.savetxt(path+"val_binary_accuracy.txt", self.numpy_val_binary_accuracy, delimiter="\n")
        print(np.mean(self.numpy_binary_accuracy))
        print(self.model.predict(self.X).round())

    def plot_history(self, layers):
        x = range(1, len(self.numpy_loss_history) + 1)
        y1 = self.numpy_loss_history
        y2 = self.numpy_binary_accuracy
        y3 = self.numpy_val_binary_accuracy

        plt.plot(x, y1,     label='Loss History',                 color='red')
        plt.plot(x, y2,     label='Binary Accuracy',              color='green')
        plt.plot(x, y3+0.5, label='Value Binary Accuracy (+0.5)', color='blue')
        plt.title(f'{layers} Layer Net')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    def run_model(self):
        self.set_seed(int(input("SEED >>> ")))
        layers = self.create_model(64, 0.4, 8) # 64 Neurons, 0.4 Dropout, 8 Layers
        self.train_model(0.01, 2000, 50)       # Learning Rate, Epochs, Patience
        ### Change Path as required to your 'Neural_Network/History_Data' folder location ###
        self.save_history('Neural_Network/History_Data')
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
        self.plot_history(layers)

if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.run_model()