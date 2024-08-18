import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetClassifier

# Train a CNN to classify MNIST
class Cnn(nn.Module):
    def __init__(self, dropout=0.5):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(p=dropout)
        self.fc1 = nn.Linear(1600, 100) # 1600 = number channels * width * height
        self.fc2 = nn.Linear(100, 10)
        self.fc1_drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flatten over channel, height and width = 1600
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        x = torch.relu(self.fc1_drop(self.fc1(x)))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
    
def getClassifier(Cnn, device, f_params='./MINST.pkl'):
    classifier = NeuralNetClassifier(
                    Cnn,
                    max_epochs=20,
                    lr=0.002,
                    optimizer=torch.optim.Adam,
                    device=device,
                )
    ### Load pretrained parameters
    classifier.initialize()
    classifier.load_params(f_params)

    return classifier

if __name__ == '__main__':
    # %% Load model

    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')

    # %% Load MINST dataset
    import sys
    from dataset import handwriting
    mnist = handwriting('mnist_784', normalize=True)

    # %% Test the model
    testnum = 50

    Xsample = mnist.XCnn[:testnum]
    Xlabel = cnn.predict_proba(Xsample)
    print(Xsample.shape, Xlabel.shape)
    print(type(mnist), type(mnist.X), type(mnist.y), type(mnist.XCnn), type(Xsample), type(Xlabel))
    # print(Xlabel-mnist.y[:testnum])