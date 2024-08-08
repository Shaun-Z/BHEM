import numpy as np
import matplotlib.pyplot as plt
import foolbox
import torch
import random
import sys
from tqdm import tqdm

import viz_2d as viz
import acd
import dset

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
        # method = 'cd'
        # tiles = acd.tiling_2d.gen_tiles(im_orig, fill=0, method=method, sweep_dim=sweep_dim)
        # scores_cd = acd.get_scores_2d(model, method=method, ims=tiles, 
        #                             im_torch=im_torch, model_type=model_type, device=device)
        # scores.append(scores_cd)

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
        titles = ['Occlusion', 'Build-Up', 'IG']
        
        for i in range(len(scores)):
            score = scores[i]
            score = score.reshape(28,28,10)
            score = np.moveaxis(score, -1, 0)

            plt.figure(figsize=(40, 10))
            for j in range(11):
                plt.subplot(i+1, 11, j+1)
                if j == 0:
                    plt.imshow(raw_image, cmap='gray')
                    plt.xticks([])
                    plt.yticks([])
                    plt.ylabel(titles[i], fontsize = 30)
                else:
                    # plt.imshow(-bhem_exp.image, cmap='gray', alpha=0.3)
                    plt.imshow(score[j-1], cmap=red_transparent_blue, vmin=-np.nanpercentile(score, 99.9),vmax=np.nanpercentile(score, 99.9))
                    plt.axis('off')
            plt.show()
