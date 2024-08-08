from lime.wrappers.scikit_image import SegmentationAlgorithm
from sklearn.utils import check_random_state
from skimage.color import gray2rgb
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

sys.path.append('/run/media/xiangyu/Data/Projects/XAI/BHEM')
from utils import basic_segment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from itertools import combinations
import tqdm

# RC Viz Code

from matplotlib.colors import LinearSegmentedColormap
colors = []
for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,j))
for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,j))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

def all_subsets(lst):
    for r in range(len(lst) + 1):
        for subset in combinations(lst, r):
            yield list(subset)

class layer:
    def __init__(self, image, layer_ID, seg_func=None, random_state=None, random_seed=None):
        self.image = image
        self.layer_ID = layer_ID
        self.random_state = check_random_state(random_state)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)

        if seg_func is None:
            # seg_func = SegmentationAlgorithm(
            #             'quickshift', kernel_size=image.shape[0]//7,
            #             max_dist=image.shape[0]//2, ratio=0.2,
            #             random_seed=random_seed)
            seg_func = SegmentationAlgorithm(
                        'quickshift', kernel_size=4,
                        max_dist=5, ratio=0.2,
                        random_seed=random_seed)
        elif seg_func == 'basic':
            basic_seg = basic_segment(image)
            seg_func = lambda img: basic_seg.get_mask(img, feature_ID=layer_ID)
            
        self.seg_func = seg_func    # Segmentation function

        if layer_ID == 0:
            self.segment = np.zeros_like(image)
            self.segment_num = 1
            self.masked_image = image
        else:
            if len(image.shape) == 2:
                img = gray2rgb(image) # H W C
            else:
                img = image
            try:
                self.segment = self.seg_func(img)
            except ValueError as e:
                raise e
            self.segment_num = np.unique(self.segment).shape[0]
            self.masked_image = None
        
        self.segment_mapping = {}
        for key in np.unique(self.segment):
            self.segment_mapping[key] = np.where(self.segment == key)

        self.seg_active = None
        
    def mask_image(self, seg_keys: list):
        masks = list(map(self.segment_mapping.get, seg_keys))
        img = np.zeros_like(self.image)
        if len(masks) != 0:
            for mask in masks:
                img[mask] = self.image[mask]
        self.masked_image = img
        self.seg_active = np.zeros(self.segment_num)
        self.seg_active = seg_keys

    def print_info(self, draw=False):
        assert self.segment is not None
        logging.info(f'''Layer {self.layer_ID}
            layer_ID:           {self.layer_ID}
            segment_num:        {self.segment_num}
            seg_keys:           {self.segment_mapping.keys()}
            segment:            {self.plot_segment() if draw else "Not Draw"}
            seg_active:         {self.seg_active}
            masked_image:       {self.plot_masked_image() if draw else "Not Draw"}
        ''')

    def plot_segment(self):
        assert self.segment is not None
        plt.imshow(self.segment)
        plt.colorbar()
        plt.title("Segment")
        plt.show()
        return "Plot Segment"

    def plot_masked_image(self):
        if self.masked_image is None:
            return None
        else:
            plt.imshow(self.masked_image)
            plt.colorbar()
            plt.title("Masked Image")
            plt.show()
            return "Plot Masked Image"

class BhemExp:
    def __init__(self, image, layer_num, random_state=None, random_seed=None):
        self.random_state = check_random_state(random_state)
        self.image = image
        self.random_seed = random_seed
        self.layer_num = layer_num+1
        # Create layers
        self.layers = [layer(image, layer_ID=0, seg_func=None, random_state=random_state, random_seed=random_seed)]
        self.layers += [layer(image, layer_ID=i, seg_func='basic', random_state=random_state, random_seed=random_seed) for i in range(1, layer_num+1)]
    
        self.mappings = {}
    
        # Initialize mapping between layers
        for i in range(1, self.layer_num-1):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            temp_dict = {}
            for key in current_layer.segment_mapping.keys():
                indexes = current_layer.segment_mapping[key]   # Get indexes
                temp_dict[key] = np.unique(next_layer.segment[indexes])
            self.mappings[f'{i}{i+1}'] = temp_dict
        
        for i in range(2, self.layer_num):
            current_layer = self.layers[i]
            prev_layer = self.layers[i-1]
            temp_dict = {}
            for key in current_layer.segment_mapping.keys():
                indexes = current_layer.segment_mapping[key]   # Get indexes
                temp_dict[key] = np.unique(prev_layer.segment[indexes])
            self.mappings[f'{i}{i-1}'] = temp_dict
    
    def get_current_masked_image(self, seg_keys: list):
        assert len(seg_keys) == self.layer_num-1
        img = np.zeros_like(self.image)
        for i in range(1, self.layer_num):
            self.layers[i].mask_image(seg_keys[i-1])
            img += self.layers[i].masked_image
        return img

    def get_explanation(self, model, label):
        indexes = [list(self.layers[i].segment_mapping.keys()) for i in range(1, self.layer_num)]
        scores = np.zeros((1, 10, int(self.image.size/4)))

        for f1 in indexes[0]:
            f1s = indexes[0].copy() # copy
            f1s.remove(f1)
            f2_idx = self.mappings['12'][f1]

            for f2 in f2_idx:
                f2s =list(f2_idx.copy()) # copy
                f2s.remove(f2)
                f3_idx = self.mappings['23'][f2]

                for f3 in tqdm.tqdm(f3_idx, desc="Layer 3"):
                    f3s = list(f3_idx.copy()) # copy
                    f3s.remove(f3)
                    f4_idx = self.mappings['34'][f3]

                    for f4 in tqdm.tqdm(f4_idx, desc="Layer 4"):
                        f4s = list(f4_idx.copy())
                        f4s.remove(f4)
                            
                        s1,s2,s3,s4 = 0,0,0,0
                        
                        for subset1 in all_subsets(f1s):
                            for subset2 in all_subsets(f2s):
                                for subset3 in all_subsets(f3s):
                                    for subset4 in all_subsets(f4s):
                                        # print(f"Feature 4: {subset4}")

                                        img1 = self.get_current_masked_image([subset1, subset2, subset3, subset4])
                                        subset4.append(f4)

                                        img = self.get_current_masked_image([subset1, subset2, subset3, subset4])

                                        # plt.figure(figsize=(20, 10))
                                        # plt.subplot(1, 2, 1)
                                        # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                                        # plt.title("Include f4")
                                        # plt.colorbar()
                                        # plt.subplot(1, 2, 2)
                                        # plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
                                        # plt.colorbar()
                                        # plt.title("Exclude f4")
                                        # plt.savefig(f'./img_res/img({f1}_{s1})({f2}_{s2})({f3}_{s3})({f4}_{s4}).png')
                                        # plt.close()

                                        P1 = model(img.reshape(1,1,28,28))
                                        P2 = model(img1.reshape(1,1,28,28))

                                        # P1.shape, P2.shape: (1,10)

                                        scores[:,:,f4] += (P1-P2)/(2**(len(subset1)+len(subset2)+len(subset3)+len(subset4)))

                                        s4 +=1
                                    s3 +=1
                                s2 +=1
                            s1 +=1
                        # print(s1,s2,s3,s4)
        return scores

        '''for f1 in indexes[0]:
            f1s = indexes[0].copy() # copy
            f1s.remove(f1)
            f2_idx = self.mappings['12'][f1]

            for f2 in f2_idx:
                f2s =list(f2_idx.copy()) # copy
                f2s.remove(f2)
                f3_idx = self.mappings['23'][f2]

                for f3 in f3_idx:
                    f3s = list(f3_idx.copy()) # copy
                    f3s.remove(f3)
                    f4_idx = self.mappings['34'][f3]

                    for f4 in f4_idx:
                        f4s = list(f4_idx.copy())
                        f4s.remove(f4)
                        f5_idx = self.mappings['45'][f4]

                        for f5 in f5_idx:
                            f5s = list(f5_idx.copy())
                            f5s.remove(f5)
                            print(f5, f5s)

                            s1,s2,s3,s4,s5 = 0,0,0,0,0
                            for subset1 in all_subsets(f1s):
                                for subset2 in all_subsets(f2s):
                                    for subset3 in all_subsets(f3s):
                                        for subset4 in all_subsets(f4s):
                                            # print(f"Feature 4: {subset4}\tLayer 5: {f5s}")
                                            for subset5 in all_subsets(f5s):

                                                img1 = self.get_current_masked_image([subset1, subset2, subset3, subset4, subset5])
                                                subset5.append(f5)

                                                img = self.get_current_masked_image([subset1, subset2, subset3, subset4, subset5])

                                                plt.figure(figsize=(20, 10))
                                                plt.subplot(1, 2, 1)
                                                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                                                plt.title("labelInclude f5")
                                                plt.colorbar()
                                                plt.subplot(1, 2, 2)
                                                plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
                                                plt.colorbar()
                                                plt.title("Exclude f5")
                                                plt.savefig(f'./img_res/img({f1}_{s1})({f2}_{s2})({f3}_{s3})({f4}_{s4})({f5}_{s5}).png')
                                                plt.close()

                                                P1 = model(img.reshape(1,1,28,28))
                                                P2 = model(img1.reshape(1,1,28,28))

                                                # P1.shape, P2.shape: (1,10)

                                                scores[:,:,f5] += (P1-P2)/(2**(len(subset1)+len(subset2)+len(subset3)+len(subset4)+len(subset5)))

                                                s5 +=1
                                            s4 +=1
                                        s3 +=1
                                    s2 +=1
                                s1 +=1
                            print(s1,s2,s3,s4,s5)
        return scores'''

        '''for f1 in indexes[0]:
            f1s = indexes[0].copy() # copy
            f1s.remove(f1)
            f2_idx = self.mappings['12'][f1]

            for f2 in f2_idx:
                f2s =list(f2_idx.copy()) # copy
                f2s.remove(f2)
                f3_idx = self.mappings['23'][f2]

                for f3 in f3_idx:
                    f3s = list(f3_idx.copy()) # copy
                    f3s.remove(f3)
                    f4_idx = self.mappings['34'][f3]

                    for f4 in f4_idx:
                        f4s = list(f4_idx.copy())
                        f4s.remove(f4)
                        f5_idx = self.mappings['45'][f4]

                        for f5 in f5_idx:
                            f5s = list(f5_idx.copy())
                            f5s.remove(f5)
                            print(f5, f5s)

                            s1,s2,s3,s4,s5 = 0,0,0,0,0
                            for subset1 in all_subsets(f1s):
                                for subset2 in all_subsets(f2s):
                                    for subset3 in all_subsets(f3s):
                                        for subset4 in all_subsets(f4s):
                                            # print(f"Feature 4: {subset4}\tLayer 5: {f5s}")
                                            for subset5 in all_subsets(f5s):
                                                img1 = self.get_current_masked_image([subset1, subset2, subset3, subset4, subset5])
                                                subset5.append(f5)

                                                img = self.get_current_masked_image([subset1, subset2, subset3, subset4, subset5])

                                                plt.figure(figsize=(20, 10))
                                                plt.subplot(1, 2, 1)
                                                plt.imshow(img, cmap='gray', vmin=0)
                                                plt.title("Include f5")
                                                plt.colorbar()
                                                plt.subplot(1, 2, 2)
                                                plt.imshow(img1, cmap='gray', vmin=0)
                                                plt.colorbar()
                                                plt.title("Exclude f5")
                                                plt.savefig(f'./img_res/img({f1}_{s1})({f2}_{s2})({f3}_{s3})({f4}_{s4})({f5}_{s5}).png')
                                                plt.close()

                                                s5 +=1
                                            s4 +=1
                                        s3 +=1
                                    s2 +=1
                                s1 +=1
                            print(s1,s2,s3,s4,s5)'''
                

        # for f2 in indexes[1]:
        #     f2s = indexes[1].copy() # copy
        #     f2s.remove(f2)
        #     subsets2 = all_subsets(f2s)
        #     for subset2 in subsets2:
        #         print(f"Layer 2: {f2}: {subset2}")
                
    def plot_viz(self, result, resized_images, savename=None):
        # result = scores.reshape(1,10, 14, 14)

        fig, axes = plt.subplots(nrows=1, ncols=11, figsize=(40,10), squeeze=False)

        axes[0, 0].imshow(self.image, cmap=plt.get_cmap("gray_r"))
        axes[0][0].axis('off')
        max_val = np.nanpercentile(result[0], 99.9)
        for i in range(10):
            axes[0][i+1].imshow(-self.image, cmap='gray', alpha=0.3)
            axes[0][i+1].imshow(resized_images[0][i], cmap=red_transparent_blue, vmin=-np.nanpercentile(result[0], 99.9),vmax=np.nanpercentile(result[0], 99.9))
            axes[0][i+1].axis('off')
            im = axes[0, i+1].imshow(resized_images[0][i], cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)

        plt.colorbar( im, ax=np.ravel(axes).tolist(), label="BHEM value", orientation="horizontal", aspect=40 / 0.2)

        if savename is not None:
            plt.savefig(savename)
        
        plt.show()

    def print_explanation_info(self):
        logging.info(f'''Explanaion Info:
            layer_num:          {self.layer_num-1}
            layers_mapping:     {self.mappings.keys()}
        ''')
        for layer_ID in range(1, self.layer_num):
            self.layers[layer_ID].print_info(draw=False)


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import sys
    import torch
    sys.path.append('E:/Projects/XAI/BHEM')
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
    Images = mnist.XCnn[:testnum]
    Xlabel = cnn.predict(Images)
    Images.shape, Xlabel.shape

    img_ID = 19
    img = Images[img_ID].reshape(28, 28)
    label = Xlabel[img_ID]
    # img = image_array.astype(np.float32)
    bhem_exp = BhemExp(img, layer_num=4, random_state=None, random_seed=None)

    plt.figure(figsize=(10, 5))
    for i in range(testnum):
        plt.subplot(5, 10, i+1)
        plt.imshow(Images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')

    bhem_exp.print_explanation_info()

    scores = bhem_exp.get_explanation(cnn.predict_proba,label)

    result = scores.reshape(1,10, 14, 14)

    np.save(f'result_array_{img_ID}.npy', result)

    import torch.nn.functional as F
    resized_images = F.interpolate(torch.tensor(result), size=(28, 28), mode='bilinear', align_corners=False).numpy()

    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for j in np.linspace(1, 0, 100):
        colors.append((30./255, 136./255, 229./255,j))
    for j in np.linspace(0, 1, 100):
        colors.append((255./255, 13./255, 87./255,j))
    red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
    plt.figure(figsize=(40, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(result[0][i], cmap=red_transparent_blue, vmin=-np.nanpercentile(result[0], 99.9),vmax=np.nanpercentile(result[0], 99.9))
        plt.axis('off')
    plt.show()

    bhem_exp.plot_viz(result, resized_images, savename=f'./images/bhem_{img_ID}.png')