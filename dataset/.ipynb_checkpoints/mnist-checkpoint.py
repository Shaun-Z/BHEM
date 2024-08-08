from sklearn.datasets import fetch_openml
from torch.utils.data import Dataset

class handwriting(Dataset):
    def __init__(self, name:str = 'mnist_784', normalize:bool = True):
        mnist = fetch_openml(name, as_frame = False, cache = True)
        self.y = mnist.target.astype('int64')
        if normalize:
            self.X = mnist.data.astype('float32')/255.0
        else:
            self.X = mnist.data.astype('float32')
        self.XCnn = self.X.reshape(-1, 1, 28, 28)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return {
            "data": self.XCnn[idx],
            "label": self.y[idx]
        }
    
    # def get(self):
    #     return self.XCnn, self.y

if __name__ == '__main__':
    mnist = handwriting('mnist_784', normalize=True)
    print(type(mnist), type(mnist.X), type(mnist.y), type(mnist.XCnn))
    print(mnist.y)
    import matplotlib.pyplot as plt

    plt.imshow(mnist.XCnn[0].reshape(28, 28), cmap='gray')
    plt.show()