{"nbformat":4,"nbformat_minor":0,"metadata":{"colab":{"provenance":[],"authorship_tag":"ABX9TyMNPLpm9XcFxlwhOHMogY24"},"kernelspec":{"name":"python3","display_name":"Python 3"},"language_info":{"name":"python"}},"cells":[{"cell_type":"code","execution_count":1,"metadata":{"colab":{"base_uri":"https://localhost:8080/","height":484},"id":"jqb4cz0iW_S1","executionInfo":{"status":"ok","timestamp":1723708106745,"user_tz":-330,"elapsed":15938,"user":{"displayName":"HARSH JETHWA","userId":"12315150990572674590"}},"outputId":"99a40bea-3694-4a27-c51f-4bd8917736c6"},"outputs":[{"output_type":"stream","name":"stdout","text":["Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz\n","\u001b[1m11490434/11490434\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\n"]},{"output_type":"display_data","data":{"text/plain":["<Figure size 640x480 with 1 Axes>"],"image/png":"iVBORw0KGgoAAAANSUhEUgAAAaAAAAGdCAYAAABU0qcqAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAcTUlEQVR4nO3df3DU9b3v8dcCyQqaLI0hv0rAgD+wAvEWJWZAxJJLSOc4gIwHf3QGvF4cMXiKaPXGUZHWM2nxjrV6qd7TqURnxB+cEaiO5Y4GE441oQNKGW7blNBY4iEJFSe7IUgIyef+wXXrQgJ+1l3eSXg+Zr4zZPf75vvx69Znv9nNNwHnnBMAAOfYMOsFAADOTwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYGGG9gFP19vbq4MGDSktLUyAQsF4OAMCTc04dHR3Ky8vTsGH9X+cMuAAdPHhQ+fn51ssAAHxDzc3NGjt2bL/PD7gApaWlSZJm6vsaoRTj1QAAfJ1Qtz7QO9H/nvcnaQFat26dnnrqKbW2tqqwsFDPPfecpk+ffta5L7/tNkIpGhEgQAAw6Pz/O4ye7W2UpHwI4fXXX9eqVau0evVqffTRRyosLFRpaakOHTqUjMMBAAahpATo6aef1rJly3TnnXfqO9/5jl544QWNGjVKL774YjIOBwAYhBIeoOPHj2vXrl0qKSn5x0GGDVNJSYnq6upO27+rq0uRSCRmAwAMfQkP0Geffaaenh5lZ2fHPJ6dna3W1tbT9q+srFQoFIpufAIOAM4P5j+IWlFRoXA4HN2am5utlwQAOAcS/im4zMxMDR8+XG1tbTGPt7W1KScn57T9g8GggsFgopcBABjgEn4FlJqaqmnTpqm6ujr6WG9vr6qrq1VcXJzowwEABqmk/BzQqlWrtGTJEl1zzTWaPn26nnnmGXV2durOO+9MxuEAAINQUgK0ePFi/f3vf9fjjz+u1tZWXX311dq6detpH0wAAJy/As45Z72Ir4pEIgqFQpqt+dwJAQAGoROuWzXaonA4rPT09H73M/8UHADg/ESAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYGGG9AGAgCYzw/5/E8DGZSVhJYjQ8eElccz2jer1nxk885D0z6t6A90zr06neMx9d87r3jCR91tPpPVO08QHvmUtX1XvPDAVcAQEATBAgAICJhAfoiSeeUCAQiNkmTZqU6MMAAAa5pLwHdNVVV+m99977x0Hi+L46AGBoS0oZRowYoZycnGT81QCAISIp7wHt27dPeXl5mjBhgu644w4dOHCg3327uroUiURiNgDA0JfwABUVFamqqkpbt27V888/r6amJl1//fXq6Ojoc//KykqFQqHolp+fn+glAQAGoIQHqKysTLfccoumTp2q0tJSvfPOO2pvb9cbb7zR5/4VFRUKh8PRrbm5OdFLAgAMQEn/dMDo0aN1+eWXq7Gxsc/ng8GggsFgspcBABhgkv5zQEeOHNH+/fuVm5ub7EMBAAaRhAfowQcfVG1trT755BN9+OGHWrhwoYYPH67bbrst0YcCAAxiCf8W3KeffqrbbrtNhw8f1pgxYzRz5kzV19drzJgxiT4UAGAQS3iAXnvttUT/lRighl95mfeMC6Z4zxy8YbT3zBfX+d9EUpIyQv5z/1EY340uh5rfHk3znvnZ/5rnPbNjygbvmabuL7xnJOmnbf/VeybvP1xcxzofcS84AIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMBE0n8hHQa+ntnfjWvu6ap13jOXp6TGdSycW92ux3vm8eeWes+M6PS/cWfxxhXeM2n/ecJ7RpKCn/nfxHTUzh1xHet8xBUQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATHA3bCjYcDCuuV3H8r1nLk9pi+tYQ80DLdd5z/z1SKb3TNXEf/eekaRwr/9dqrOf/TCuYw1k/mcBPrgCAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMcDNS6ERLa1xzz/3sFu+Zf53X6T0zfM9F3jN/uPc575l4PfnZVO+ZxpJR3jM97S3eM7cX3+s9I0mf/Iv/TIH+ENexcP7iCggAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMMHNSBG3jPV13jNj3rrYe6bn8OfeM1dN/m/eM5L0f2e96D3zm3+7wXsmq/1D75l4BOriu0Fogf+/WsAbV0AAABMECABgwjtA27dv10033aS8vDwFAgFt3rw55nnnnB5//HHl5uZq5MiRKikp0b59+xK1XgDAEOEdoM7OThUWFmrdunV9Pr927Vo9++yzeuGFF7Rjxw5deOGFKi0t1bFjx77xYgEAQ4f3hxDKyspUVlbW53POOT3zzDN69NFHNX/+fEnSyy+/rOzsbG3evFm33nrrN1stAGDISOh7QE1NTWptbVVJSUn0sVAopKKiItXV9f2xmq6uLkUikZgNADD0JTRAra2tkqTs7OyYx7Ozs6PPnaqyslKhUCi65efnJ3JJAIAByvxTcBUVFQqHw9GtubnZekkAgHMgoQHKycmRJLW1tcU83tbWFn3uVMFgUOnp6TEbAGDoS2iACgoKlJOTo+rq6uhjkUhEO3bsUHFxcSIPBQAY5Lw/BXfkyBE1NjZGv25qatLu3buVkZGhcePGaeXKlXryySd12WWXqaCgQI899pjy8vK0YMGCRK4bADDIeQdo586duvHGG6Nfr1q1SpK0ZMkSVVVV6aGHHlJnZ6fuvvtutbe3a+bMmdq6dasuuOCCxK0aADDoBZxzznoRXxWJRBQKhTRb8zUikGK9HAxSf/nf18Y3908veM/c+bc53jN/n9nhPaPeHv8ZwMAJ160abVE4HD7j+/rmn4IDAJyfCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYML71zEAg8GVD/8lrrk7p/jf2Xr9+Oqz73SKG24p955Je73eewYYyLgCAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMcDNSDEk97eG45g4vv9J75sBvvvCe+R9Pvuw9U/HPC71n3Mch7xlJyv/XOv8h5+I6Fs5fXAEBAEwQIACACQIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACa4GSnwFb1/+JP3zK1rfuQ988rq/+k9s/s6/xuY6jr/EUm66sIV3jOX/arFe+bEXz/xnsHQwRUQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGAi4Jxz1ov4qkgkolAopNmarxGBFOvlAEnhZlztPZP+00+9Z16d8H+8Z+I16f3/7j1zxZqw90zPvr96z+DcOuG6VaMtCofDSk9P73c/roAAACYIEADAhHeAtm/frptuukl5eXkKBALavHlzzPNLly5VIBCI2ebNm5eo9QIAhgjvAHV2dqqwsFDr1q3rd5958+appaUlur366qvfaJEAgKHH+zeilpWVqays7Iz7BINB5eTkxL0oAMDQl5T3gGpqapSVlaUrrrhCy5cv1+HDh/vdt6urS5FIJGYDAAx9CQ/QvHnz9PLLL6u6ulo/+9nPVFtbq7KyMvX09PS5f2VlpUKhUHTLz89P9JIAAAOQ97fgzubWW2+N/nnKlCmaOnWqJk6cqJqaGs2ZM+e0/SsqKrRq1aro15FIhAgBwHkg6R/DnjBhgjIzM9XY2Njn88FgUOnp6TEbAGDoS3qAPv30Ux0+fFi5ubnJPhQAYBDx/hbckSNHYq5mmpqatHv3bmVkZCgjI0Nr1qzRokWLlJOTo/379+uhhx7SpZdeqtLS0oQuHAAwuHkHaOfOnbrxxhujX3/5/s2SJUv0/PPPa8+ePXrppZfU3t6uvLw8zZ07Vz/5yU8UDAYTt2oAwKDHzUiBQWJ4dpb3zMHFl8Z1rB0P/8J7Zlgc39G/o2mu90x4Zv8/1oGBgZuRAgAGNAIEADBBgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJhI+K/kBpAcPW2HvGeyn/WfkaRjD53wnhkVSPWe+dUlb3vP/NPCld4zozbt8J5B8nEFBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCY4GakgIHemVd7z+y/5QLvmclXf+I9I8V3Y9F4PPf5f/GeGbVlZxJWAgtcAQEATBAgAIAJAgQAMEGAAAAmCBAAwAQBAgCYIEAAABMECABgggABAEwQIACACQIEADBBgAAAJrgZKfAVgWsme8/85V/8b9z5qxkvec/MuuC498y51OW6vWfqPy/wP1Bvi/8MBiSugAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAE9yMFAPeiILx3jP778yL61hPLH7Ne2bRRZ/FdayB7JG2a7xnan9xnffMt16q857B0MEVEADABAECAJjwClBlZaWuvfZapaWlKSsrSwsWLFBDQ0PMPseOHVN5ebkuvvhiXXTRRVq0aJHa2toSumgAwODnFaDa2lqVl5ervr5e7777rrq7uzV37lx1dnZG97n//vv11ltvaePGjaqtrdXBgwd18803J3zhAIDBzetDCFu3bo35uqqqSllZWdq1a5dmzZqlcDisX//619qwYYO+973vSZLWr1+vK6+8UvX19bruOv83KQEAQ9M3eg8oHA5LkjIyMiRJu3btUnd3t0pKSqL7TJo0SePGjVNdXd+fdunq6lIkEonZAABDX9wB6u3t1cqVKzVjxgxNnjxZktTa2qrU1FSNHj06Zt/s7Gy1trb2+fdUVlYqFApFt/z8/HiXBAAYROIOUHl5ufbu3avXXvP/uYmvqqioUDgcjm7Nzc3f6O8DAAwOcf0g6ooVK/T2229r+/btGjt2bPTxnJwcHT9+XO3t7TFXQW1tbcrJyenz7woGgwoGg/EsAwAwiHldATnntGLFCm3atEnbtm1TQUFBzPPTpk1TSkqKqquro481NDTowIEDKi4uTsyKAQBDgtcVUHl5uTZs2KAtW7YoLS0t+r5OKBTSyJEjFQqFdNddd2nVqlXKyMhQenq67rvvPhUXF/MJOABADK8APf/885Kk2bNnxzy+fv16LV26VJL085//XMOGDdOiRYvU1dWl0tJS/fKXv0zIYgEAQ0fAOeesF/FVkUhEoVBIszVfIwIp1svBGYy4ZJz3THharvfM4h9vPftOp7hn9F+9Zwa6B1r8v4tQ90v/m4pKUkbV7/2HenviOhaGnhOuWzXaonA4rPT09H73415wAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJggQAAAEwQIAGCCAAEATBAgAIAJAgQAMBHXb0TFwDUit+/fPHsmn794YVzHWl5Q6z1z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6yUlVTgcliRlZGRIknbt2qXu7u6Y18OkSZM0bty4If16OPU8fOmVV15RZmamJk+erIqKCh09etRief0acDcjPdVnn32mnp4eZWdnxzyenZ2tP//5z0arslFUVKSqqipdccUVamlp0Zo1a3T99ddr7969SktLs16eidbWVknq8/Xx5XPni3nz5unmm29WQUGB9u/fr0ceeURlZWWqq6vT8OHDrZeXcL29vVq5cqVmzJihyZMnSzr5ekhNTdXo0aNj9h3Kr4e+zoMk3X777Ro/frzy8vK0Z88ePfzww2poaNCbb75puNpYAz5A+IeysrLon6dOnaqioiKNHz9eb7zxhu666y7DlWEguPXWW6N/njJliqZOnaqJEyeqpqZGc+bMMVxZcpSXl2vv3r3nxfugZ9Lfebj77rujf54yZYpyc3M1Z84c7d+/XxMnTjzXy+zTgP8WXGZmpoYPH37ap1ja2tqUk5NjtKqBYfTo0br88svV2NhovRQzX74GeH2cbsKECcrMzBySr48VK1bo7bff1vvvvx/z61tycnJ0/Phxtbe3x+w/VF8P/Z2HvhQVFUnSgHo9DPgApaamatq0aaquro4+1tvbq+rqahUXFxuuzN6RI0e0f/9+5ebmWi/FTEFBgXJycmJeH5FIRDt27DjvXx+ffvqpDh8+PKReH845rVixQps2bdK2bdtUUFAQ8/y0adOUkpIS83poaGjQgQMHhtTr4WznoS+7d++WpIH1erD+FMTX8dprr7lgMOiqqqrcH//4R3f33Xe70aNHu9bWVuulnVMPPPCAq6mpcU1NTe53v/udKykpcZmZme7QoUPWS0uqjo4O9/HHH7uPP/7YSXJPP/20+/jjj93f/vY355xzP/3pT93o0aPdli1b3J49e9z8+fNdQUGB++KLL4xXnlhnOg8dHR3uwQcfdHV1da6pqcm999577rvf/a677LLL3LFjx6yXnjDLly93oVDI1dTUuJaWluh29OjR6D733HOPGzdunNu2bZvbuXOnKy4udsXFxYarTryznYfGxkb34x//2O3cudM1NTW5LVu2uAkTJrhZs2YZrzzWoAiQc84999xzbty4cS41NdVNnz7d1dfXWy/pnFu8eLHLzc11qamp7tvf/rZbvHixa2xstF5W0r3//vtO0mnbkiVLnHMnP4r92GOPuezsbBcMBt2cOXNcQ0OD7aKT4Ezn4ejRo27u3LluzJgxLiUlxY0fP94tW7ZsyP2ftL7++SW59evXR/f54osv3L333uu+9a1vuVGjRrmFCxe6lpYWu0UnwdnOw4EDB9ysWbNcRkaGCwaD7tJLL3U/+tGPXDgctl34Kfh1DAAAEwP+PSAAwNBEgAAAJggQAMAEAQIAmCBAAAATBAgAYIIAAQBMECAAgAkCBAAwQYAAACYIEADABAECAJj4f4W4/AnknuSPAAAAAElFTkSuQmCC\n"},"metadata":{}},{"output_type":"stream","name":"stdout","text":["(28, 28)\n"]}],"source":["# Practical 9\n","# Implementation of convolutional neural network to predict numbers from number images.\n","from keras.datasets import mnist\n","from keras.utils import to_categorical\n","from keras.models import Sequential\n","from keras.layers import Dense,Conv2D,Flatten\n","import matplotlib.pyplot as plt\n","#download mnist data and split into train and test sets\n","(X_train,Y_train),(X_test,Y_test)=mnist.load_data()\n","#plot the first image in the dataset\n","plt.imshow(X_train[0])\n","plt.show()\n","print(X_train[0].shape)\n"]},{"cell_type":"code","source":["X_train=X_train.reshape(60000,28,28,1)\n","X_test=X_test.reshape(10000,28,28,1)\n","Y_train=to_categorical(Y_train)\n","Y_test=to_categorical(Y_test)\n","Y_train[0]\n","print(Y_train[0])\n","model=Sequential()\n","#add model layers\n","#learn image features\n","model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(28,28,1)))\n","model.add(Conv2D(32,kernel_size=3,activation='relu'))\n","model.add(Flatten())\n","model.add(Dense(10,activation='softmax'))\n","model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])\n","#train\n","model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=3)\n"],"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"BsSFsXkeAmAg","executionInfo":{"status":"ok","timestamp":1723708720066,"user_tz":-330,"elapsed":577836,"user":{"displayName":"HARSH JETHWA","userId":"12315150990572674590"}},"outputId":"f58463f3-79fd-457d-b645-c5cf5f72d70f"},"execution_count":2,"outputs":[{"output_type":"stream","name":"stdout","text":["[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]\n"]},{"output_type":"stream","name":"stderr","text":["/usr/local/lib/python3.10/dist-packages/keras/src/layers/convolutional/base_conv.py:107: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n","  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"]},{"output_type":"stream","name":"stdout","text":["Epoch 1/3\n","\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m178s\u001b[0m 94ms/step - accuracy: 0.9176 - loss: 0.7098 - val_accuracy: 0.9695 - val_loss: 0.0945\n","Epoch 2/3\n","\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m201s\u001b[0m 94ms/step - accuracy: 0.9807 - loss: 0.0618 - val_accuracy: 0.9745 - val_loss: 0.0796\n","Epoch 3/3\n","\u001b[1m1875/1875\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m198s\u001b[0m 92ms/step - accuracy: 0.9894 - loss: 0.0354 - val_accuracy: 0.9729 - val_loss: 0.1065\n"]},{"output_type":"execute_result","data":{"text/plain":["<keras.src.callbacks.history.History at 0x783ee37a9870>"]},"metadata":{},"execution_count":2}]},{"cell_type":"code","source":["print(model.predict(X_test[:4]))\n","#actual results for 1st 4 images in the test set\n","print(Y_test[:4])\n"],"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"NlYz5XRJAmDT","executionInfo":{"status":"ok","timestamp":1723708729337,"user_tz":-330,"elapsed":436,"user":{"displayName":"HARSH JETHWA","userId":"12315150990572674590"}},"outputId":"dd2fec20-b630-4d96-a056-2b71dc419cd7"},"execution_count":3,"outputs":[{"output_type":"stream","name":"stdout","text":["\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 82ms/step\n","[[2.17051169e-10 1.02870141e-17 1.12878794e-07 2.76721153e-06\n","  2.05811523e-16 3.61337804e-13 2.28831410e-18 9.99997020e-01\n","  4.05557010e-09 7.10084151e-08]\n"," [7.84439980e-09 3.04237364e-11 9.99999404e-01 1.64115228e-08\n","  3.67084386e-16 4.39920073e-13 6.30336217e-07 1.07863234e-16\n","  8.05282438e-11 1.71869604e-12]\n"," [2.00671657e-05 9.96867001e-01 1.49983622e-04 1.54276927e-06\n","  1.38927787e-03 1.61139295e-04 1.87082587e-05 5.77298124e-06\n","  1.38640963e-03 6.27018153e-08]\n"," [9.99998331e-01 9.62229862e-13 1.18627295e-08 4.56635729e-10\n","  2.70722941e-12 2.57116516e-11 1.45321792e-06 5.66733032e-12\n","  2.99604265e-11 2.40928784e-07]]\n","[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]\n"," [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]\n"," [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]\n"," [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]\n"]}]},{"cell_type":"code","source":[],"metadata":{"id":"x72jqDAMAmFs"},"execution_count":null,"outputs":[]},{"cell_type":"code","source":[],"metadata":{"id":"i-OnocjtAmJK"},"execution_count":null,"outputs":[]}]}