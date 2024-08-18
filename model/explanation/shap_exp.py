import shap
import numpy as np
import torch

import matplotlib.pyplot as plt

import sys

from utils import red_transparent_blue

class ShapExp:
    def __init__(self, model, image: torch.Tensor, masker=None):
        self.batch_size = image.shape[0]
        self.H = image.shape[-2]
        self.W = image.shape[-1]

        self.mask = masker

        if masker is None:
            masker = shap.maskers.Image(f"blur({self.H},{self.W})", (1,self.H,self.W))
        else:
            masker = shap.maskers.Image(np.expand_dims(masker, axis=0), (1,self.H,self.W))
        
        self.y_pred = model(image)
        self.y = np.argmax(self.y_pred[0])
        
        self.explainer = shap.Explainer(model, masker)

        shap_values = self.explainer(image, max_evals=20000, batch_size=500)

        self.shap_values = np.moveaxis(shap_values.values,-1,0)

        self.shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in self.shap_values]
        self.test_numpy = np.swapaxes(np.swapaxes(image.numpy(), 1, -1), 1, 2)

    def plot_cmp(self):
        shap.image_plot(self.shap_numpy, -self.test_numpy)
    
    def plot_mask(self):
        plt.figure()
        plt.imshow(self.mask)
        plt.show()

    def plot_shap(self):
        plt.figure()
        plt.imshow(-self.test_numpy[0], cmap='gray', alpha=0.3)
        plt.imshow(self.shap_values[self.y,0].reshape(self.H,self.W), cmap=red_transparent_blue, vmin=-np.nanpercentile(self.shap_values[self.y,0], 99.9),vmax=np.nanpercentile(self.shap_values[self.y,0], 99.9))
        plt.show()

    def plot_image(self):
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(self.test_numpy[0], cmap='gray')
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.imshow(-self.test_numpy[0], cmap='gray', alpha=0.3)
        plt.imshow(self.shap_values[self.y,0].reshape(self.H,self.W), cmap=red_transparent_blue, vmin=-np.nanpercentile(self.shap_values[self.y,0], 99.9),vmax=np.nanpercentile(self.shap_values[self.y,0], 99.9))
        plt.title("Shap Image")
        plt.subplot(1, 3, 3)
        plt.imshow(self.mask)
        plt.title("Mask")
        plt.show()

if __name__ == "__main__":
    import sys
    from model import Cnn, getClassifier
    from dataset import handwriting
    from utils import reconstruct_mask, basic_segment, quickshift, slic
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Cnn().to(device)
    cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')
    # %% Load MINST dataset
    mnist = handwriting('mnist_784', normalize=True)
    testID = 5
    Image = torch.from_numpy(mnist.XCnn[testID]) # C H W

    # %%
    from skimage.color import gray2rgb
    image = Image[0]
    if len(image.shape) == 2:
        image = gray2rgb(image)
    segments_qs = quickshift(image, kernel_size=3, max_dist=10, ratio=0.2)
    segments_slic = slic(image, n_segments=100, compactness=0.1, sigma=0.5)
    segments_basic = basic_segment(Image)
    print(segments_basic.get_mask().shape)
    # %% Shap
    shap_exp = ShapExp(cnn.predict_proba, Image.unsqueeze(0), masker=segments_basic.get_mask())
    # shap_exp = ShapExp(cnn.predict_proba, Image.unsqueeze(0), masker=segments_qs)
    print(shap_exp.y_pred, np.argmax(shap_exp.y_pred))
    shap_exp.plot_cmp()
    shap_exp.plot_mask()
    shap_exp.plot_shap()
    shap_exp.plot_image()

    import pandas as pd
    import seaborn as sns
    print(segments_qs)
    shap_values = shap_exp.shap_values[:,0].reshape(-1,shap_exp.H,shap_exp.W)
    # df = pd.DataFrame(shap_values)
    # df.to_csv('shap_values.csv', index=False)
    # print(len(np.unique(shap_values)))
    # sns.heatmap(shap_values, cmap='coolwarm')
    # plt.show()

    np.save(f'./result/shap/result_array_{testID}.npy', shap_values)