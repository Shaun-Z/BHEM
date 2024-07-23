import numpy as np
import torch

# LIME explanation
from lime import lime_image
import shap

from IPython.display import clear_output

# %% LIME explanation
def lime_predict(z):
    """In `explainer.explain_instance()` method, the 2D image is converted to 3D by adding a channel dimension"""
    z = z[:, :, :, 0]
    return cnn.predict_proba(z[:,np.newaxis,:,:])

def lmask(contribution, SEG_PIXELS:dict):
    mask = np.zeros((28,28),dtype = np.float32)
    for data in contribution:
        index, score = data
        mask[SEG_PIXELS[index]] = score
    return mask

def get_lime_explaination(images, exlabel, seg_fn):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(images, lime_predict, labels=(exlabel,), top_labels=None, num_features=1000, segmentation_fn = seg_fn, num_samples=5000)
    #explanation = explainer.explain_instance(images, lime_predict, hide_color=0, labels=(exlabel,), top_labels=None, num_features=1000, segmentation_fn = lime_segments, num_samples=500) # number of images that will be sent to classification function
    #explanation = explainer.explain_instance(Xsample[1,0], lime_predict, hide_color=0, top_labels=10, num_features=1000, num_samples=500) # number of images that will be sent to classification function
    clear_output(wait=True)
    lime_mask = explanation.local_exp
    temp = lime_mask[exlabel]
    powerlimedict = {}
    for pair in temp:
        powerlimedict[pair[0]] = pair[1]
    #return lmask(temp,BASIC_SEG_PIXELS)
    return powerlimedict

# %% Shap explanation
def shapPf(z):
    return cnn.predict_proba(z)

def get_shap_explaination(images, exlabel):
    masker = shap.maskers.Image("blur(28,28)", (1,28,28))
    Pexplainer = shap.Explainer(shapPf, masker)
    shap_Pvalues = Pexplainer(images.reshape(1,1,28,28), max_evals=20000, batch_size=500)
    temp = shap_Pvalues.values[0,0,:,:,exlabel] #(28,28)
    mask = np.zeros((28,28),dtype = np.float32)
    powershapdict = {}
    for k in range(0,14*14):
        row = k//14
        column = k%14
        s = np.mean(temp[row*2:(row+1)*2,column*2:(column+1)*2])
        mask[row*2:(row+1)*2,column*2:(column+1)*2] = s
        powershapdict[k] = s
    #return mask
    return powershapdict

# %%

# %% Test
if __name__ == '__main__':
    import sys
    sys.path.append('E:/Projects/XAI/BHEM')
    from model import Cnn, getClassifier
    from dataset import handwriting
    from utils import reconstruct_mask, basic_segment
    
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cnn = getClassifier(Cnn, device, f_params='./MINST.pkl')

    # %% Load MINST dataset
    mnist = handwriting('mnist_784', normalize=True)

    testnum = 50
    Xsample = mnist.XCnn[:testnum]
    Xlabel_pred = cnn.predict(Xsample)

    pick = 2
    Xtoy = Xsample[pick,0]
    toylabel_pred = Xlabel_pred[pick]

    print(f"Size of Xtoy: {Xtoy.shape}")

    print(f"Pick: {pick}, Label_pred: {toylabel_pred}")
    # %% Basic segmentation: Filling 28*28 with 2*2 segments
    basic_seg = basic_segment(Xtoy)
    feature = basic_seg.get_mask(4)
    basic_seg.plot_segments(4)

    BASIC_SEG_CLASSES = np.unique(feature)
    BASIC_SEG_PIXELS = {}
    for i in BASIC_SEG_CLASSES:
        BASIC_SEG_PIXELS[i] = np.where(feature==i)

    import matplotlib.pyplot as plt
    plt.imshow(Xtoy)
    plt.show()

    print(type(Xtoy), Xtoy.shape)

    limedict = get_lime_explaination(Xtoy,toylabel_pred,seg_fn=basic_seg.get_mask)
    shapdict = get_shap_explaination(Xtoy,toylabel_pred)
    # bhemdict = get_bhem_explaination(Xtoy,toylabel_pred)
    limefeatures = sorted(limedict, key=lambda k: limedict[k], reverse=True)
    shapfeatures = sorted(shapdict, key=lambda k: shapdict[k], reverse=True)
    # bhemfeatures = sorted(bhemdict, key=lambda k: bhemdict[k], reverse=True)
    
    percent = 50
    num_select = int(14*14*percent/100)
    
    lime_fill = reconstruct_mask(limefeatures[:num_select],Xtoy,BASIC_SEG_PIXELS)
    shap_fill = reconstruct_mask(shapfeatures[:num_select],Xtoy,BASIC_SEG_PIXELS)
    # bhem_fill = reconstruct_mask(bhemfeatures[:num_select],Xtoy,BASIC_SEG_PIXELS)

    fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)
    ax[0].imshow(Xtoy, cmap=plt.get_cmap('gray'))
    ax[0].axis('off')
    #ax[0].set_title('{}'.format(label))
    ax[1].imshow(lime_fill, cmap=plt.get_cmap('gray'))
    ax[1].axis('off')
    ax[2].imshow(shap_fill, cmap=plt.get_cmap('gray'))
    ax[2].axis('off')
    #ax[1].set_title('Level-1')
    # ax[3].imshow(bhem_fill, cmap=plt.get_cmap('gray'))
    # ax[3].axis('off')
    
    ax[0].set_title('Original Image')
    ax[1].set_title('LIME')
    ax[2].set_title('SHAP')
    # ax[3].set_title('BHEM')

    fig.suptitle(r'Image reconstructed eliminating top $k\%$ features. $k=50\%$', fontsize=16)
    plt.show()