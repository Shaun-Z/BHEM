import numpy as np
import torch
import sys
import tqdm

sys.path.append('/run/media/xiangyu/Data/Projects/XAI/BHEM')

from model.explanation.shap_exp import ShapExp
from matplotlib import pyplot as plt
from model import Cnn, getClassifier
from dataset import handwriting
from utils import basic_segment

def delet_top_k_feature(k, img, value):
    total_sum = np.sum(value)

    # Sort the values in descending order and get the corresponding indices
    sorted_indices = np.argsort(-img.flatten())
    sorted_values = value.flatten()[sorted_indices]

    # Calculate the cumulative sum
    cumulative_sum = np.cumsum(sorted_values)

    # Find the index where the cumulative sum exceeds 50% of the total sum
    index = np.argmax(cumulative_sum > k * total_sum)

    # Set the values after the index to 0
    result = img.flatten()
    result[sorted_indices[:index]] = 0
    result = result.reshape(img.shape)

    return result

class aopc:
    def __init__(self, model, percents=[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]):
        # self.exp_values = exp_values
        self.model = model
        # self.testnum = testnum
        self.percents = np.array(percents)

        self.LIME_AOPC = np.array([0.0]*len(self.percents))
        self.SHAP_AOPC = np.array([0.0]*len(self.percents))
        self.BHEM_AOPC = np.array([0.0]*len(self.percents))
        self.ACD_AOPC = np.array([0.0]*len(self.percents))

    '''
    This method could be used to calculate the AOPC value for a single image. Any explanation method fits as long as the explanation values are provided.
    '''
    def get_single_aopc_value(self, image, y, exp_values):
        X_pred = self.model(image)
        base_value = X_pred.flatten()[y].item()

        AOPC = np.array([0.0]*len(self.percents))
        for i in range(len(self.percents)):
            res = delet_top_k_feature(self.percents[i], image.numpy(), exp_values)
            AOPC[i] = base_value - self.model(res).flatten()[y]
        return AOPC
    
    def get_average_shap_aopc_value(self, testnum, dataset):
        self.SHAP_AOPC = np.array([0.0]*len(self.percents))
        for img_ID in tqdm.tqdm(range(testnum), desc=f"\033[92m{testnum}\033[0m images"):
            Image = torch.from_numpy(dataset.XCnn[img_ID]).unsqueeze(0)
            y = dataset.y[img_ID]

            basic_seg = basic_segment(Image)
            shap_exp = ShapExp(self.model, Image, masker=basic_seg.get_mask())
            exp_values = shap_exp.shap_values[y, 0, 0, :, :]

            self.SHAP_AOPC += self.get_single_aopc_value(Image, y, exp_values)

        self.SHAP_AOPC = self.SHAP_AOPC/testnum

        return self.SHAP_AOPC
    
    def plot_aopc(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.percents*100, self.BHEM_AOPC, 's-', markersize = 4, color = 'y', label="BHEM")
        plt.plot(self.percents*100, self.SHAP_AOPC, '^-', markersize = 4, color = 'g', label="SHAP")
        plt.plot(self.percents*100, self.LIME_AOPC, 'o-', markersize = 4, color = 'c', label="LIME")
        plt.plot(self.percents*100, self.ACD_AOPC, '*-', markersize = 4, color = 'b', label="ACD")
        plt.ylabel("AOPC (MINST)",fontsize = 20)
        plt.xlabel("k%",fontsize = 20)
        plt.legend(prop={'size':20})
        plt.show()

if __name__ == '__main__':
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')
    # %% Load MINST dataset
    mnist = handwriting('mnist_784', normalize=True)

    AOPCs = aopc(cnn.predict_proba)

    # %% Test the AOPC on single image
    img_ID = 0
    Image = torch.from_numpy(mnist.XCnn[img_ID]).unsqueeze(0)
    y = mnist.y[img_ID]

    basic_seg = basic_segment(Image)
    shap_exp = ShapExp(cnn.predict_proba, Image, masker=basic_seg.get_mask())
    shap_exp_values = shap_exp.shap_values[y, 0, 0, :, :]

    print(f"SHAP AOPC on image {img_ID} is {AOPCs.get_single_aopc_value(Image, y, shap_exp_values)}")

    # %% Test the AOPC on batched images
    print(AOPCs.get_average_shap_aopc_value(testnum=50, dataset = mnist.XCnn))
    AOPCs.plot_aopc()