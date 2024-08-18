import numpy as np
import torch
import sys
import tqdm

from model.explanation.shap_exp import ShapExp
from model.explanation.lime_exp import LimeExp
from model.explanation.acd_exp import AcdExp
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
        self.ACD_AOPC0 = np.array([0.0]*len(self.percents))
        self.ACD_AOPC1 = np.array([0.0]*len(self.percents))
        self.ACD_AOPC2 = np.array([0.0]*len(self.percents))

    '''
    This method could be used to calculate the AOPC value for a single image. Any explanation method fits as long as the explanation values are provided.
    '''
    def get_single_aopc_value(self, model, image, y, exp_values):
        X_pred = model(image)
        base_value = X_pred.flatten()[y].item()

        AOPC = np.array([0.0]*len(self.percents))
        for i in range(len(self.percents)):
            res = delet_top_k_feature(self.percents[i], image.numpy(), exp_values)
            AOPC[i] = base_value - model(res).flatten()[y]
        return AOPC
    
    def get_average_shap_aopc_value(self, testnum, dataset):
        self.SHAP_AOPC = np.array([0.0]*len(self.percents))
        for img_ID in tqdm.tqdm(range(testnum), desc=f"SHAP: \033[92m{testnum}\033[0m images"):
            Image = torch.from_numpy(dataset.XCnn[img_ID]).unsqueeze(0)
            y = dataset.y[img_ID]

            basic_seg = basic_segment(Image)
            shap_exp = ShapExp(self.model, Image, masker=basic_seg.get_mask())
            exp_values = shap_exp.shap_values[y, 0, 0, :, :]

            self.SHAP_AOPC += self.get_single_aopc_value(self.model, Image, y, exp_values)

        self.SHAP_AOPC = self.SHAP_AOPC/testnum

        return self.SHAP_AOPC

    def get_average_lime_aopc_value(self, testnum, dataset):
        self.LIME_AOPC = np.array([0.0]*len(self.percents))
        for img_ID in tqdm.tqdm(range(testnum), desc=f"LIME: \033[92m{testnum}\033[0m images"):
            Image = torch.from_numpy(dataset.XCnn[img_ID])
            y = dataset.y[img_ID]
            basic_seg = basic_segment(Image)
            lime_exp = LimeExp(Image, basic_seg.get_mask, num_features=784, num_samples=2000)
            exp_values = lime_exp.get_exp_values()

            self.LIME_AOPC += self.get_single_aopc_value(self.model, Image.unsqueeze(0), y, exp_values)

        self.LIME_AOPC = self.LIME_AOPC/testnum

        return self.LIME_AOPC
    
    def get_average_bhem_aopc_value(self, testnum, dataset):
        pass

    def get_average_acd_aopc_value(self, testnum, dataset):
        model = Cnn()
        checkpoint = torch.load('./MINST.pkl', map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        self.ACD_AOPC0 = np.array([0.0]*len(self.percents))
        self.ACD_AOPC1 = np.array([0.0]*len(self.percents))
        self.ACD_AOPC2 = np.array([0.0]*len(self.percents))

        for img_ID in tqdm.tqdm(range(testnum), desc=f"ACD: \033[92m{testnum}\033[0m images"):
            Image = torch.tensor(dataset.XCnn[img_ID].reshape(-1, 1, 28, 28)).to(device)
            X_pred = model(Image)
            y = dataset.y[img_ID]

            base_value = X_pred.flatten()[y].item()
            
            ACDexp = AcdExp(Image, sweep_dim=1)
            scores = ACDexp.get_explanation(model, y)
            acd_exp_values0 = scores[0][:, y].reshape(28,28)
            acd_exp_values1 = scores[1][:, y].reshape(28,28)
            acd_exp_values2 = scores[2][:, y].reshape(28,28)

            for i in range(len(self.percents)):
                res0 = torch.tensor(delet_top_k_feature(self.percents[i], Image.cpu().numpy(), acd_exp_values0)).to(device)
                res1 = torch.tensor(delet_top_k_feature(self.percents[i], Image.cpu().numpy(), acd_exp_values1)).to(device)
                res2 = torch.tensor(delet_top_k_feature(self.percents[i], Image.cpu().numpy(), acd_exp_values2)).to(device)
                # plt.imshow(res.cpu().numpy().reshape(28, 28), cmap='gray')
                # plt.show()
                self.ACD_AOPC0[i] += base_value - model(res0).flatten()[y]
                self.ACD_AOPC1[i] += base_value - model(res1).flatten()[y]
                self.ACD_AOPC2[i] += base_value - model(res2).flatten()[y]

        self.ACD_AOPC0 = self.ACD_AOPC0/testnum
        self.ACD_AOPC1 = self.ACD_AOPC1/testnum
        self.ACD_AOPC2 = self.ACD_AOPC2/testnum

        return self.ACD_AOPC0, self.ACD_AOPC1, self.ACD_AOPC2

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


    AOPCs = aopc(cnn.predict_proba, percents=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    # [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    # [0.2,0.4,0.6,0.8,1.0]

    # %% Test the AOPC on single image
    img_ID = 0
    Image = torch.from_numpy(mnist.XCnn[img_ID]).unsqueeze(0)
    y = mnist.y[img_ID]

    basic_seg = basic_segment(Image)
    # %% Test on SHAP
    # shap_exp = ShapExp(cnn.predict_proba, Image, masker=basic_seg.get_mask())
    # shap_exp_values = shap_exp.shap_values[y, 0, 0, :, :]
    # print(f"SHAP AOPC on image {img_ID} is {AOPCs.get_single_aopc_value(Image, y, shap_exp_values)}")
    # %% Test on LIME
    # lime_exp = LimeExp(Image.squeeze(0), basic_seg.get_mask, num_features=784, num_samples=2000)
    # lime_exp_values = lime_exp.get_exp_values()
    # print(f"SHAP AOPC on image {img_ID} is {AOPCs.get_single_aopc_value(Image, y, lime_exp_values)}")
    # %% Test on ACD

    # %% Test the AOPC on batched images
    print(AOPCs.get_average_shap_aopc_value(testnum=50, dataset = mnist))
    print(AOPCs.get_average_lime_aopc_value(testnum=50, dataset = mnist))
    print(AOPCs.get_average_acd_aopc_value(testnum=50, dataset = mnist))
    AOPCs.plot_aopc()