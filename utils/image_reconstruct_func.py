import numpy as np
# %% Reconstruction
def reconstruct_mask(features:list, img:np.ndarray, SEG_PIXELS:dict):
    # out = np.ones_like(img) * 0.0
    out = np.zeros_like(img)
    if len(features) == 0:
        return out
    for i in features:
        print(i)
        out[SEG_PIXELS[i]] = img[SEG_PIXELS[i]]
    return out
# %% Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    sys.path.append('E:/Projects/XAI/BHEM')
    from segmentation import basic_segmentation
    from dataset import handwriting
    # Load MINST dataset
    mnist = handwriting('mnist_784', normalize=True)
    testnum = 2
    img = mnist.XCnn[testnum,0]
    
    # Basic segmentation: Filling 28*28 with 2*2 segments
    basic_seg = basic_segmentation(0)
    
    BASIC_SEG_CLASSES = np.unique(basic_seg)
    BASIC_SEG_PIXELS = {}
    for i in BASIC_SEG_CLASSES:
        BASIC_SEG_PIXELS[i] = np.where(basic_seg==i)

    features = list(range(89, 196))
    lime_recon = reconstruct_mask(features,img,BASIC_SEG_PIXELS)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].imshow(img, cmap=plt.get_cmap('gray'))
    ax[0].axis('on')
    ax[1].imshow(lime_recon, cmap=plt.get_cmap('gray'))
    ax[1].axis('on')
    plt.show()
