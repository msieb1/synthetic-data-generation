import argparse
import os
from os.path import join as join
from PIL import Image
from copy import deepcopy as copy
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops
from scipy import ndimage as ndi
import time
# plt.ion()
from ipdb import set_trace as st


DEPTH_TH = 5 # Minimum depth difference after BS
DEPTH_MIN = 0 # General minimum depth (to exclude erroneous depth values or out of bounds (zeroed out))
DEPTH_MAX = 80
RGB_TH = 30 # Minimum rgb difference after BS

H = 480 
W = 640
CLIPPING_DISTANCE = 1.5

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Name of the dataset we`re collecting.')
parser.add_argument('--expdir', default='/home/zhouxian/projects/experiments',
                       help='dir to write experimental data to.')
parser.add_argument('--imagedir', default='/tmp/tcn/videos',
                       help='Base directory to write videos.')
parser.add_argument('--depthdir', default='/tmp/tcn/depth',
                       help='Base directory to write depth.')
parser.add_argument('--target', type=str, required=True, help='the target object name')
args = parser.parse_args()



def main(args):
    # process_images(args.filepath)
    # depth_subtraction(args.filepath)
    print('make sure that ..00000_viewX.png is the BACKGROUND image!')

    save_path = join(args.expdir, args.dataset, 'masks', args.target)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # for file in sorted(os.listdir(join(args.expdir, args.dataset, 'images'))):
    image_path = join(args.expdir, args.dataset, args.imagedir, args.target)
    grabcut(image_path, save_path, args)


def grabcut(image_path, save_path, args):

    print("input path: ", image_path)
    for file in sorted(os.listdir(image_path)):
        view = file.split('view')[-1][0]
        if int(file.split('_')[1]) == 0:
            continue    
        try:
            rgb, depth, valid_rgb, valid_depth = get_preprocessed_data(image_path, file)
        except:
            break
        
        if os.path.exists(join(save_path, file)):
            continue
        print("file: %s" % file)

        img = rgb
        mask = np.zeros(img.shape[:2],np.uint8)
        # Make mask generation easier> 3 means possibly foreground, 1 means definitely foreground
        mask[np.where(valid_rgb)] = 3 
        # mask[np.where(valid_depth)] = 3 
        # mask[np.where(valid_depth * valid_rgb)] = 1 

        # mask[np.where(valid == False)] = 2
        mask[:int(0.3*H), 0:int(0.2*W)] = 0
        mask[-int(0.1*H):, 0:int(0.2*W)] = 0
        # mask[:200, 0:50] = 0
        mask[-int(0.1*H):, -int(0.2*W):] = 0
        mask[:int(0.3*H), -int(0.2*W):] = 0
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (int(0.1*W), int(0.1*H), int(0.9*W), int(0.9*H) ) # (x, y, w, h)
        mask, bgdModel, fgdModel = cv2.grabCut(img,mask,None,bgdModel,fgdModel,13,cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        rps = sorted((regionprops(label(mask > 0.5, background=0))), key=lambda x: x.area, reverse=True)
        mask_clean = np.zeros(mask.shape)
        mask_clean[rps[0].coords[:, 0], rps[0].coords[:,1]] = 1
        # mask_clean = ndi.binary_fill_holes(mask_clean, structure=np.ones((2,2))).astype(np.uint8)
        mask_clean_ = copy(mask_clean)
        mask_clean = ndi.binary_fill_holes(mask_clean_).astype(np.uint8)
        img_masked = img*mask_clean[:,:,np.newaxis]
        np.save(join(save_path, '{}'.format(file[:-4])), mask_clean)
        cv2.imwrite(join(save_path, '{}.png'.format(file[:-4])), img)

        cv2.imwrite(join(save_path, '{}_masked.png'.format(file[:-4])), img_masked.astype(np.uint8)[:, :, :])
        cv2.imshow('frame',rgb.astype(np.uint8))
        vis_mask = ndi.binary_erosion(mask)
        cv2.imshow('mask', mask_clean*255)
        cv2.imshow('img',img_masked.astype(np.uint8)[:, :, :])
        
        k = cv2.waitKey(150)
    print("="*20)
    cv2.destroyAllWindows()

def get_preprocessed_data(image_path, file):
    # depth_path = join(data_path, 'depth', seqname, '{0:06d}.png'.format(fr))
    object_name = file.split('_')[0]
    rgb_path = join(image_path, file)
    rgb = cv2.imread(rgb_path)
    # depth_img = cv2.imread(depth_path)
    # depth_img = ndi.filters.gaussian_filter(depth_img, (7, 7, 0), order=0)
    # depth = depth_img / 255.0 * CLIPPING_DISTANCE        
    # depth *= 0.0010000000474974513
    
    view = file.split('view')[-1][0]
    background_rgb = cv2.imread(join(image_path, '{}_000000_view{}.png'.format(object_name, view)))
    rgb_fg = rgb.astype(np.float) - background_rgb.astype(np.float)
    # depth_fg = depth_img.astype(np.float) - background_depth.astype(np.float)
    # plt.imshow(rgb_fg)
    valid_rgb = get_mask(np.max(np.abs(rgb_fg), axis=2) > RGB_TH).astype(np.uint8)
    # valid_depth = get_mask((np.abs(depth_fg[:, :, 0]) > DEPTH_TH) * \
    #             (np.abs(depth_img[:, :, 0]) > DEPTH_MIN) * \
    #             (np.abs(depth_img[:, :, 0]) < DEPTH_MAX)).astype(np.uint8)
    # valid = ndimage.binary_erosion(valid).astype(np.float32)
    # rgb = rgb.astype(np.float32)
    # rgb = rgb/255.0 - 0.5
    # rgb = np.reshape(rgb, [1, H, W, 3])

    # depth = np.expand_dims(depth, axis=0)
    # depth = depth.astype(np.float32)
    # depth = np.reshape(depth, [1, H, W, 1])
    # plt.show()
    return rgb, None, valid_rgb, None

def process_images(filepath):

    fgbg = cv2.createBackgroundSubtractorMOG2()  
    filepath = join(filepath, 'images')
    for file in os.listdir(filepath):
        if file[-4:] != '.png':
            continue
        print("Processing current image {}".format(file))
        file_path = join(ROOT_PATH, 'images', file)
        try:
            frame = cv2.imread(file_path)
        except:
            print("Image could not be opened - check path")

        fgmask = fgbg.apply(frame)
        cv2.imshow('frame',frame)
        cv2.imshow('mask',fgmask)
        k = cv2.waitKey(400)

    cv2.destroyAllWindows()

def get_mask(cond):
    return np.where(cond, np.ones([H, W]), np.zeros([H, W])).astype(np.float32)





def depth_subtraction(seqname):
    nFrames = 100
    for fr in range(nFrames):
        rgb, depth, mask = get_preprocessed_data(seqname, fr)
        mask = np.stack((mask, mask, mask), axis=2)
        masked = np.multiply(rgb, mask).astype(np.uint8)
        cv2.imshow('frame',rgb.astype(np.uint8))
        cv2.imshow('mask',masked.astype(np.uint8)[:, :, :])
        k = cv2.waitKey(20)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        main(args)
    except KeyboardInterrupt:
        # do nothing here
        cv2.destroyAllWindows()
