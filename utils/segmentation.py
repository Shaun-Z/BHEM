import numpy as np
import matplotlib.pyplot as plt
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import quickshift, slic

# %% Basic segmentation: Filling 28*28 with 2*2 segments
def basic_segmentation(img):
    seg = np.zeros((28,28),dtype = int)
    for i in range(14):
        for j in range(14):
            block_number = i * 14 + j
            seg[i*2:(i+1)*2,j*2:(j+1)*2] = block_number
    return seg

class basic_segment:
    def __init__(self, img):
        H = img.shape[0]
        W = img.shape[1]
        self.W = W
        self.H = H
        self.img = img

        Feature_0 = np.zeros((H,W),dtype = int)

        Feature_1 = np.zeros((H,W),dtype = int)
        Feature_1[:,:W//3-1], Feature_1[:,W//3-1:W-W//3+1], Feature_1[:,W-W//3+1:]= 0,1,2
        
        Feature_2 = np.zeros((H,W),dtype = int)
        H1, H2, H3= H//7*2, H//7*3, H//7*2
        W1, W2, W3= W//7*2, W//7*3, W//7*2
        Feature_2[:H1,:W1], Feature_2[H1:H1+H2,:W1], Feature_2[H1+H2:H1+H2+H3,:W1] = 0, 1, 2
        Feature_2[:H1,W1:W1+W2], Feature_2[H1:H1+H2,W1:W1+W2], Feature_2[H1+H2:H1+H2+H3,W1:W1+W2] = 3, 4, 5 
        Feature_2[:H1,W1+W2:W1+W2+W3], Feature_2[H1:H1+H2,W1+W2:W1+W2+W3], Feature_2[H1+H2:H1+H2+H3,W1+W2:W1+W2+W3] = 6, 7, 8

        Feature_3 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, 4):
            for j in range(0, W, 4):
                Feature_3[i:i+4,j:j+4] = num
                num += 1

        Feature_4 = np.zeros((H,W),dtype = int)
        num = 0
        for i in range(0, H, 2):
            for j in range(0, W, 2):
                Feature_4[i:i+2,j:j+2] = num
                num += 1

        Feature_5 = np.zeros((H,W),dtype = int)
        for i in range(H):
            for j in range(W):
                Feature_5[i,j] = i*W+j

        self.features_list = [Feature_0, Feature_1, Feature_2, Feature_3, Feature_4, Feature_5]

    def get_mask(self, img=None, feature_ID=4):
        return self.features_list[feature_ID]

    def plot_segments(self, feature_ID):
        feature = self.get_mask(feature_ID=feature_ID)
        print(feature)
        # Display heatmap of basic_seg
        plt.figure(figsize=(18,18))
        plt.imshow(feature, cmap='cool')
        plt.colorbar().outline.set_linewidth(1)
        plt.gca().xaxis.tick_top()
        plt.gca().yaxis.tick_left()
        for i in range(self.W):
            for j in range(self.H):
                plt.text(j, i, int(feature[i, j]), ha='center', va='center', color='black')

        plt.xticks(np.arange(0, self.W, 1))
        plt.yticks(np.arange(0, self.H, 1))
        plt.title(f'Basic Segmentation: Layer {feature_ID}', fontsize=25, y=1.05)
        plt.show()

    # def __call__(self):
    #     return self.seg

if __name__ == '__main__':
    import torch
    from matplotlib import pyplot as plt
    from skimage.color import gray2rgb
    import sys
    sys.path.append('E:/Projects/XAI/BHEM')
    # basic_seg = basic_segment(np.zeros((28,28)))
    # print(basic_seg.H, basic_seg.W)
    # for i in range(6):
    #     basic_seg.plot_segments(feature_ID=i)
    from dataset import handwriting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mnist = handwriting('mnist_784', normalize=True)

    testnum = 50
    Images = mnist.XCnn[:testnum]
    test_ID = 17
    image = Images[test_ID][0]
    if len(image.shape) == 2:
        image = gray2rgb(image)


    segments_qs = quickshift(image, kernel_size=3, max_dist=10, ratio=0.2)

    segments_slic = slic(image, n_segments=100, compactness=0.1, sigma=0.5)

    print(f"Number of segments: {len(np.unique(segments_qs))}, {len(np.unique(segments_slic))}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Original Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(segments_qs)
    plt.title(f"Quickshift Segmentation")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(segments_slic)
    plt.title(f"SLIC Segmentation")
    plt.axis('off')
    plt.show()

    print(segments_qs)
