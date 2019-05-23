import argparse
import os
from os.path import join
import sys
import numpy as np
import cv2
from copy import deepcopy as copy
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
import random
from numpy.random import randint
import importlib
from ipdb import set_trace
import time
import imgaug as ia
from imgaug import augmenters as iaa
plt.ion()

ia.seed(1)
## EXAMPLE USAGE ####
# python create_synthetic_images.py -i /home/msieb/projects/gps-lfd/demo_data/four_objects -e four_objects -m train
# SET EXPNAME IN CONFIG.PY

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Name of the dataset we`re collecting.')
parser.add_argument('--expdir', required=True,
                       help='dir to write experimental data to.')

parser.add_argument('--maskdir', required=True,
                       help='Base directory to write masks.')
parser.add_argument('--backgrounddir', required=True,
                       help='Base directory to write backgrounds.')       
parser.add_argument('--mode', type=str, default='train', help='train, valid or test')
args = parser.parse_args()


MAX_SHIFT_COL = 90
MAX_SHIFT_ROW = 180
N_ADDED_MIN = 3
N_ADDED_MAX = 4

def main(args):
    gen = SyntheticImageGenerator(args)
    gen.create_synthetic_images()

class SyntheticImageGenerator(object):

    def __init__(self, args):
        self.mask_root_path = join(args.expdir, args.dataset, args.maskdir)
        self.bg_path = join(args.expdir, args.dataset, args.backgrounddir)
        self.save_path = join(args.expdir, args.dataset, 'synthetic_data', args.mode)
        print("write to: ",self.save_path)
        time.sleep(3)

    def create_synthetic_images(self, n_iter=500):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # input_list = os.listdir(self.mask_root_path)
        # input_list = sorted(input_list, key=lambda x: x.split('.')[0])
        bg_list = os.listdir(self.bg_path)
        all_images_to_add = []
        for folder in os.listdir(self.mask_root_path):
            mask_path = join(self.mask_root_path, folder)
            all_images_to_add.extend([file for file in os.listdir(mask_path) if (file.endswith('.jpg') or file.endswith('.png')) and not 'masked' in file  and not 'hand' in file and not 'robot' in file])
        random.shuffle(all_images_to_add)
        
        for itr in range(n_iter):
            for file_base in bg_list:

                print("Base image: ", file_base)
                n_added_images = randint(N_ADDED_MIN, N_ADDED_MAX)
                added_images = [all_images_to_add[i] for i in np.random.choice(np.arange(len(all_images_to_add)), n_added_images)]
                # Overlay images to get synthetic image
                img_overlayed, mask_overlayed, mask_labels, save_name = self.make_synthetic_image(file_base, added_images)

                # Save to file
                print("saved as ", save_name )
                print("="*20)

                cv2.imwrite(join(self.save_path, save_name + '.png'), img_overlayed)
                # np.save(join(self.save_path, save_name + '.npy'), mask_overlayed)
                # np.save(join(self.save_path, save_name + '_labels.npy'), mask_labels)
                cv2.imwrite(join(self.save_path, save_name + '_maskvis.png'), mask_overlayed*255)

                # cv2.imshow('img_overlayed',img_overlayed)

                k = cv2.waitKey(50)

    def make_synthetic_image(self, file_base, added_images):

        img_base = cv2.imread(join(self.bg_path, file_base))  

        # Store mask labels for later training, i.e. stores the corresponding object label for every mask channel
        save_name = ''
        mask_labels = []
        mask_overlayed = np.zeros((img_base.shape[0], img_base.shape[1]), dtype=np.uint8)[:, :, None]

        # Get rid of placeholder channel entry
        mask_overlayed = mask_overlayed[:, :, 1:]

        if len(mask_overlayed.shape) < 3:
            mask_overlayed = mask_overlayed[:, :, np.newaxis]
        img_overlayed = copy(img_base)

        # Perturb background
        scale = np.random.uniform(0.6,1.1)
        img_perturbed = copy(img_overlayed)
        img_perturbed = (img_perturbed * scale).astype(np.uint8)
        img_perturbed[np.where(img_perturbed > 255)] = 255
        img_perturbed[np.where(img_perturbed < 0)] = 0
        img_overlayed = img_perturbed

        for i, file_added in enumerate(added_images):
             # Read image to be added on top

            print("Added image: ", file_added)

            img_added = cv2.imread(join(self.mask_root_path, file_added.split('_')[0], file_added))
            
            if file_base.endswith('.jpg'):
                mask_added = np.load(join(self.mask_root_path, file_added.split('_')[0], file_added.split('.jpg')[0] + '.npy'))
            else:
                mask_added = np.load(join(self.mask_root_path, file_added.split('_')[0], file_added.split('.png')[0] + '.npy'))

            mask_labels.append(self.mask_root_path.split('/')[-1])

            # Mask image
            # img_added_masked = img_added * mask_added[:,:,np.newaxis]
            
            aff = iaa.Sequential(
                [
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    iaa.Flipud(0.2), # vertically flip 20% of all images

                    
                    iaa.Affine(
                    scale={"x": (0.7, 1.3), "y": (0.7, 1.3)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-90, 90),
                    shear=(-15, 15)
                )
                ])
            # Affine transform
            aff_det = aff.to_deterministic()
            img_added = aff_det.augment_image(img_added)
            mask_added = aff_det.augment_image(mask_added)

            # Static Transform
            st = iaa.SomeOf((0, 3),
                [
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.02*255), per_channel=0.5
                ),


                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.8, 1.2), per_channel=0.0),

                # Improve or worsen the contrast of images.
                iaa.ContrastNormalization((0.8, 1.2), per_channel=0.5),
                iaa.OneOf([
                    iaa.Dropout((0.005, 0.01), per_channel=0.0),
                    iaa.CoarseDropout(
                        (0.01, 0.05), size_percent=(0.01,0.05),
                        per_channel=0.0
                    ),
                ]),
              # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 30), per_channel=0.0),
                ]
            )
            img_added = st.augment_image(img_added)
            img_added_masked = img_added * mask_added[:,:,np.newaxis]

            # # Augment masks
            # img_added_masked, mask_added = self.translate_mask(img_added_masked, mask_added, \
            #                                                 row_shift=randint(-MAX_SHIFT_ROW, MAX_SHIFT_ROW), \
            #                                                 col_shift=randint(-MAX_SHIFT_COL, MAX_SHIFT_COL))
            # img_added_masked, mask_added = self.rotate_mask(img_added_masked, mask_added, \
            #                                                 angle=randint(-180,180,1), center=None, \
            #                                                 scale=np.random.uniform(0.9, 1.1))
            # img_added_masked, mask_added = self.perturb_intensity(img_added_masked, mask_added, scale=np.random.uniform(0.9,1.1))

            # Apply masks
            img_overlayed[np.where(mask_added == 1)] = img_added_masked[np.where(mask_added == 1)]
            for j in range(mask_overlayed.shape[-1]):
                mask_overlayed[:, :, j] *= np.logical_not(mask_added)
            mask_overlayed = np.concatenate([mask_overlayed, \
                                    mask_added[:, :, np.newaxis]], axis=2)  
            # Save image and mask
            if i > 0: connector = '_' 
            else: connector = ''

            if file_base.endswith('.jpg'):
                save_name += connector + file_added.split('.jpg')[0] 
            else:
                save_name += connector + file_added.split('.png')[0] 
        
        # if same overlay combo exists, assign unique suffix
        save_name += '-0' 
        while os.path.exists(join(self.save_path, save_name + '.jpg')):
            index = int(save_name.split('-')[-1][0])
            save_name = save_name.split('-')[0] + '-' + str(index + 1)

        return img_overlayed, mask_overlayed, mask_labels, save_name


    def overlay_img(self, img_base, mask_base, img_added_masked, mask_added):
        img_overlayed = copy(img_base)
        img_overlayed[np.where(mask_added == 1)] = img_added_masked[np.where(mask_added == 1)]

        mask_overlayed = copy(mask_base)
        mask_overlayed *= np.logical_not(mask_added)
        mask_overlayed = np.concatenate([mask_overlayed[:, :, np.newaxis], \
                                        mask_added[:, :, np.newaxis]], axis=2)
        return img_overlayed, mask_overlayed

    def perturb_intensity(self, img_masked, mask, scale=0):
        img_perturbed = copy(img_masked)
        img_perturbed = (img_perturbed * scale).astype(np.uint8)
        img_perturbed[np.where(img_perturbed > 255)] = 255
        img_perturbed[np.where(img_perturbed < 0)] = 0
        return img_perturbed, mask

    def translate_mask(self, img_masked, mask, row_shift=0, col_shift=0):
        mask_shifted = shift(mask, [row_shift, col_shift, ])
        img_masked_shifted = shift(img_masked, [row_shift, col_shift, 0])
        return img_masked_shifted, mask_shifted

    def rotate_mask(self, img_masked, mask, angle=0, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = img_masked.shape[:2]

        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)

        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated_img = cv2.warpAffine(img_masked, M, (w, h))
        rotated_mask = cv2.warpAffine(mask, M, (w, h))

        # return the rotated image
        return rotated_img, rotated_mask


    # def translate(image, x, y):
    #     # define the translation matrix and perform the translation
    #     M = np.float32([[1, 0, x], [0, 1, y]])
    #     shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    #     # return the translated image
    #     return shifted




    def get_masked_img(self, img, mask):
        img_masked = img*mask[:,:,np.newaxis]
        return img_masked


if __name__ == '__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        # do nothing here
        cv2.destroyAllWindows()

