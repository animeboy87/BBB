# Practical 4
# 4A
# Using deep feed forward network with two hidden layers for performing classification and predicting the class.
from tensorflow import keras  # Using tensorflow.keras instead of keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=500)
Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)
# Use predict followed by argmax for class prediction
Ynew = np.argmax(model.predict(Xnew), axis=-1)  # Import numpy as np
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s, Desired=%s" % (Xnew[i], Ynew[i], Yreal[i]))


# 4B
# Using deep feed forward network with two hidden layers for performing classification and predicting the probability of class.
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import numpy as np
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scalar = MinMaxScaler()
scalar.fit(X)
X = scalar.transform(X)
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, Y, epochs=500)
Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scalar.transform(Xnew)  # Corrected variable assignment here
Ynew_probabilities = model.predict(Xnew)
Ynew_classes = np.round(Ynew_probabilities).astype(int)  # Convert probabilities to classes
for i in range(len(Xnew)):
    print("X=%s, Predicted_probability=%s, Predicted_class=%s" % (Xnew[i], Ynew_probabilities[i], Ynew_classes[i]))



