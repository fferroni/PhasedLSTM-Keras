from keras.models import Sequential
from keras.layers import LSTM
from PhasedLSTM import PhasedLSTM
import numpy as np


def main():
	X = np.random.random((32,100,2))
	Y = np.random.random((32,10))

	model_lstm = Sequential()
	model_lstm.add(LSTM(10, input_shape=(100,2)))
	model_lstm.summary()
	model_lstm.compile('rmsprop','mse')

	model_plstm = Sequential()
	model_plstm.add(PhasedLSTM(10, input_shape=(100,2)))
	model_plstm.summary()
	model_plstm.compile('rmsprop','mse')

	model_lstm.fit(X,Y)
	model_plstm.fit(X,Y)


if __name__ == "__main__":
	main()