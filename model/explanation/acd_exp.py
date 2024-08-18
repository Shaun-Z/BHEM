import numpy as np
import matplotlib.pyplot as plt
import foolbox
import torch
import random
import sys
from tqdm import tqdm

import acd

sys.path.append('/umich/Library/Mobile Documents/com~apple~CloudDocs/BHEM')
sys.path.append('/run/media/xiangyu/Data/Projects/XAI/BHEM')

from model.explanation import dset

# %%
import torch.nn as nn
import torch.nn.functional as F
from utils import red_transparent_blue

class AcdExp:
    def __init__(self, image = None, sweep_dim=1, device = 'cuda'):
       # image: tensor(1,1,28,28)
       self.image = image
       self.sweep_dim = sweep_dim
       self.device = device

    def get_diff_scores(self, im_torch, im_orig, label_num, model, preds, sweep_dim):
        '''Computes different attribution scores
        '''
        scores = []

        # cd
        method = 'cd'
        model_type = 'mnist'
        tiles = acd.tiling_2d.gen_tiles(im_orig, fill=0, method=method, sweep_dim=sweep_dim)
        scores_cd = acd.get_scores_2d(model, method=method, ims=tiles, 
                                    im_torch=im_torch, model_type=model_type, device=self.device)
        scores.append(scores_cd)

        for method in ['occlusion', 'build_up']: # 'build_up'
            tiles_break = acd.tiling_2d.gen_tiles(im_orig, fill=0, method=method, sweep_dim=sweep_dim)
            preds_break = acd.get_scores_2d(model, method=method, ims=tiles_break, 
                                                im_torch=im_torch, pred_ims=dset.pred_ims)
            if method == 'occlusion':
                preds_break += preds
            scores.append(np.copy(preds_break))
        
        # get integrated gradients scores
        scores.append(acd.ig_scores_2d(model, im_torch, num_classes=10, 
                                            im_size=28, sweep_dim=sweep_dim, ind=[label_num], device=self.device))
        return scores

    def get_explanation(self, model, label):
        # model is cnn object
        # self.image is (1,1,28,28) tensor
        X_orig = self.image.view(self.image.shape[-2], self.image.shape[-1]).cpu().numpy()
        # pred.size = (10,)
        preds = model(self.image).flatten().cpu().detach().numpy()
        scores = self.get_diff_scores(self.image, X_orig, label, model, preds, self.sweep_dim)
        return scores
    
    def plot_raw_scores(self, scores):
        # raw image
        raw_image = self.image.view(self.image.shape[-2], self.image.shape[-1]).cpu().numpy()
        titles = ['CD', 'Occlusion', 'Build-Up', 'IG']
        
        plt.figure(figsize=(40, 10))
        for i in range(len(scores)):
            score = scores[i]
            score = score.reshape(28,28,10)
            score = np.moveaxis(score, -1, 0)

            for j in range(11):
                plt.subplot(len(scores), 11, (i)*11+j+1)
                if j == 0:
                    plt.imshow(raw_image, cmap='gray_r', alpha=0.3)
                    plt.xticks([])
                    plt.yticks([])
                    plt.ylabel(titles[i], fontsize = 25)
                else:
                    plt.imshow(raw_image, cmap='gray_r', alpha=0.3)
                    plt.imshow(score[j-1], cmap=red_transparent_blue, vmin=-np.nanpercentile(score, 99.9),vmax=np.nanpercentile(score, 99.9))
                    plt.axis('off')
        plt.show()

if __name__ == '__main__':

    from model import Cnn, getClassifier
    from dataset import handwriting
    from acd_exp import AcdExp

    model_type = 'mnist'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist = handwriting('mnist_784', normalize=True)
    print(type(mnist), type(mnist.X), type(mnist.y), type(mnist.XCnn))
    print(mnist.y)

    model = Cnn()
    checkpoint = torch.load('./MINST.pkl', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.to(device)

    Xsample = torch.tensor(mnist.XCnn[0:1]).to(device)
    X_preds = model(Xsample)
    Xsample.shape, X_preds.shape

    img_ID = 5

    image = torch.tensor(mnist.XCnn[img_ID].reshape(-1, 1, 28, 28)).to(device)

    ACDexp = AcdExp(image, sweep_dim=1)

    scores = ACDexp.get_explanation(model, mnist.y[img_ID])

    ACDexp.plot_raw_scores(scores)
