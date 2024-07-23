import torch
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

from model import Cnn, getClassifier
from dataset import handwriting

# %% Load model

torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')

# %% Load MINST dataset
mnist = handwriting('mnist_784', normalize=True)

#XCnn_train, XCnn_test, y_train, y_test = train_test_split(XCnn, y, test_size=0.005, random_state=42)
#XCnn_train.shape, y_train.shape

# %% Test the model
testnum = 50

Images = mnist.XCnn[:testnum]
Xlabel_pred = cnn.predict(Images)

# print(Xsample.shape)
# print(type(mnist), type(mnist.X), type(mnist.y), type(mnist.XCnn), type(Xsample), type(Xlabel_pred))
# print(Xlabel-mnist.y[:testnum])

# Pick up a image
pick = 1
Xtoy = Images[pick,0]
toylabel_pred = Xlabel_pred[pick]

print(toylabel_pred)

plt.imshow(Xtoy)
plt.show()
