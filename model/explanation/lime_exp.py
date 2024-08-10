import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries
from lime.lime_image import LimeImageExplainer


def lime_predict(z):
    """In `explainer.explain_instance()` method, the 2D image is converted to 3D by adding a channel dimension"""
    z = z[:, :, :, 0]
    return cnn.predict_proba(z[:,np.newaxis,:,:])

class LimeExp:
    def __init__(self, image: torch.Tensor, seg_fn, num_features=196, num_samples=5000):
        explainer = LimeImageExplainer()
        self.explanation = explainer.explain_instance(np.array(image.squeeze(0).cpu()), 
                                        lime_predict, # 分类预测函数
                                        top_labels=5,
                                        hide_color=0,
                                        num_features=num_features, # 特征数
                                        num_samples=num_samples, # LIME生成的邻域图像个数
                                        segmentation_fn=seg_fn
                                        )
        self.image_marked, self.mask = self.explanation.get_image_and_mask(self.explanation.top_labels[0], positive_only=False, num_features=num_features//11, hide_rest=False)
        self.img_boundry = mark_boundaries(self.image_marked/255.0, self.mask)
    
    def get_image(self):
        return  self.mask, self.image_marked, self.img_boundry
    
    def plot_mask(self, save=False, show=True):
        plt.figure()
        plt.imshow(self.mask)
        plt.title("Mask")
        if show:
            plt.show()

    def plot_image(self, save=False, show=True):
        plt.figure()
        plt.imshow(self.image_marked)
        plt.title("Positive/Negative")
        if show:
            plt.show()
    
    def plot_boundry(self, save=False, show=True):
        plt.figure()
        plt.imshow(self.img_boundry)
        plt.title("Boundary")
        if show:
            plt.show()

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import sys
    sys.path.append('E:/Projects/XAI/BHEM')
    sys.path.append('/umich/Library/Mobile Documents/com~apple~CloudDocs/BHEM')
    sys.path.append('/run/media/xiangyu/Data/Projects/XAI/BHEM')
    from model import Cnn, getClassifier
    from dataset import handwriting
    from utils import reconstruct_mask, basic_segment, quickshift, slic
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')
    # %% Load MINST dataset
    mnist = handwriting('mnist_784', normalize=True)
    testnum = 50
    Images = torch.from_numpy(mnist.XCnn[:testnum]).to(device)
    # y_pred = torch.from_numpy(cnn.predict(Images))
    y_pred_logits = torch.from_numpy(cnn.predict_proba(Images))
    pred_softmax = F.softmax(y_pred_logits, dim=1)
    top_n = pred_softmax.topk(1)
    plt.figure(figsize=(10, 8))
    for i in range(testnum):
        plt.subplot(5, 10, i+1)
        plt.imshow(Images[i].cpu().reshape(28, 28), cmap='gray')
        plt.title(f"Pred: {top_n.indices[i].item()}")
        plt.axis('off')
    plt.show()

    basic_seg = basic_segment(Images)
    test_ID = 2
    lime_exp = LimeExp(Images[test_ID], basic_seg.get_mask, num_features=784, num_samples=5000)
    # lime_exp = LimeExp(Images[test_ID], lambda img: slic(img, n_segments=100, compactness=0.1, sigma=0.5), num_features=784, num_samples=5000)
    # lime_exp = LimeExp(Images[test_ID], lambda img: quickshift(img, kernel_size=4, max_dist=10, ratio=0.2), num_features=196, num_samples=5000)
    mask, image_marked, img_boundary = lime_exp.get_image()
    lime_exp.plot_boundry()
    lime_exp.plot_image()
    lime_exp.plot_mask()

    print(image_marked)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(Images[test_ID].cpu().reshape(28, 28), cmap='gray')
    plt.title("Original Image")
    plt.subplot(1, 3, 2)
    plt.imshow(image_marked)
    plt.title("Positive/Negative")
    plt.subplot(1, 3, 3)
    plt.imshow(mask)
    plt.title("Mask")
    plt.show()



