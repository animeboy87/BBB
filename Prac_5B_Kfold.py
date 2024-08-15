# Practical 5B
# Evaluating feed forward deep network for multiclass classification using KFold cross-validation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
iris = load_iris()
X = iris.data
y = iris.target
label_encoder=LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y_encoded = onehot_encoder.fit_transform(integer_encoded)
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
def create_model():
  model = Sequential()
  model.add(Dense(10, input_dim=4, activation= 'relu'))
  model.add(Dense(8, activation='relu'))
  model.add(Dense(3, activation='softmax'))
  model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
fold = 0
accuracies = []
conf_matrices = []
for train_index, test_index in kf.split(X):
  fold += 1
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y_encoded[train_index], y_encoded[test_index]
  model = create_model()
  model.fit(X_train, y_train, epochs=100, batch_size=5, verbose=0)
  y_pred = model.predict(X_test)
  y_pred_classes = np.argmax(y_pred, axis=1)
  y_test_classes = np.argmax(y_test, axis=1)
  accuracy = accuracy_score(y_test_classes, y_pred_classes)
  accuracies.append(accuracy)
  conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
  conf_matrices.append(conf_matrix)
  print(f"Fold {fold} - Accuracy: {accuracy}")
  print("Confusion Matrix: ")
  print(conf_matrix)
  print()
avg_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {avg_accuracy}')
plt.figure(figsize=(8, 6))
sns.boxplot(y=accuracies)
plt.title('Distribution of Accuracies Across Folds')
plt.xlabel('Accuracy')
plt.show()
