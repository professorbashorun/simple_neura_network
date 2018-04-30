from keras.models import Sequential;
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD



# globals
CATEGORICAL_CROSSENTROPY_LOSS = 'categorical_crossentropy';
STOCHASTIC_GRADIENT_DESCENT = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True);
RELU_ACTIVATION = 'relu';
SOFTMAX_ACTIVATION = 'softmax';
ACCURACY_METRIC = 'metric';




# Declare model architecture type
model = Sequential();







# train your model with this simple architecture
def train(x_train, y_train, loss=CATEGORICAL_CROSSENTROPY_LOSS, optimizer=STOCHASTIC_GRADIENT_DESCENT, batch_size=32, epochs=10):
	n_ins = x_train.shape[1];
	n_outs = y_train.shape[1]
	model.add(Convolution2D(32, (3,3), activation=RELU_ACTIVATION, input_dim=n_ins ));
	model.add(MaxPooling2D(pool_size=(2,2)));
	model.add(Dropout(0.25));
	model.add(Flatten());
	model.add(Dense(256, activation=RELU_ACTIVATION));
	model.add(Dropout(0.5));
	model.add(Dense(n_outs, activation=SOFTMAX_ACTIVATION));
	model.compile(loss=loss, optimizer=optimizer, metrics=[ACCURACY_METRIC]);
	model.fit(x_train, y_train);
	return model;








# check the accuracy of your model
def test(x_test, y_test, batch_size=32):
	return model.evaluate(x_test, y_test, batch_size);








# user model to make predictions
def predict(x):
	return model.predict(x);