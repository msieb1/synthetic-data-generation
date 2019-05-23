# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import multiprocessing
from multiprocessing import Process
import imageio
import os
import subprocess
import sys
import time
import cv2
import string
import matplotlib
matplotlib.use('TkAgg')

from matplotlib import animation  # pylint: disable=g-import-not-at-top
import matplotlib.pyplot as plt
import numpy as np
from six.moves import input
import wave
from scipy import misc
import pyrealsense2 as rs
import io
import rospy
import tf
from geometry_msgs.msg import PointStamped

sys.path.append('/'.join(os.path.realpath(__file__).split('/')[:-1]))
import ipdb; ipdb.set_trace()
from utils.subscribers import img_subscriber, depth_subscriber
from utils.utils import query_yes_no
from utils.data_collection_utils import ImageQueue, timer, setup_paths, setup_paths_w_depth, get_view_dirs, get_view_dirs_depth


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='training_2', help='Name of the dataset we`re collecting.')
parser.add_argument('--mode', default='train',help='What type of data we`re collecting. E.g.:'
                       '`train`,`valid`,`test`, or `demo`')
parser.add_argument('--seqname', default='',help='Name of this sequence. If empty, the script will use'
                       'the name seq_N+1 where seq_N is the latest'
                       'integer-named sequence in the videos directory.')
parser.add_argument('--num_views', type=int, default=1,help='Number of webcams.')
parser.add_argument('--expdir', default='/home/zhouxian/projects/experiments',
                       help='dir to write experimental data to.')
parser.add_argument('--tmp_imagedir', default='/tmp/tcn/data',
                       help='Temporary outdir to write images.')
parser.add_argument('--viddir', default='/tmp/tcn/videos',
                       help='Base directory to write videos.')
parser.add_argument('--depthdir', default='/tmp/tcn/depth',
                       help='Base directory to write depth.')

parser.add_argument('--debug_vids', default=False,
                        help='Whether to generate debug vids with multiple concatenated views.')
parser.add_argument('--debug_lhs_view', default='1',
                       help='Which viewpoint to use for the lhs video.')
parser.add_argument('--debug_rhs_view', default='2',
                       help='Which viewpoint to use for the rhs video.')
parser.add_argument('--height', default=1080, help='Raw input height.')
parser.add_argument('--width', default=1920, help='Raw input width.')
parser.add_argument('--time_in_between', type=float, default=0.3, help='time between pictures')
parser.add_argument('--webcam_ports', default='2,5,8',help='Comma-separated list of each webcam usb port.')
args = parser.parse_args()


### Global Constants ###
INTRIN = np.array([615.3900146484375, 0.0, 326.35467529296875, 
		              0.0, 615.323974609375, 240.33250427246094, 
		              0.0, 0.0, 1.0]).reshape(3, 3) # D435 intrinsics matrix
FPS = 25.0
DEPTH_SCALE = 0.001 # not precisely, but up to e-8
CLIPPING_DISTANCE_IN_METERS = 1.5 #1.5m

### Initialize buffers ###
GLOBAL_IMAGE_BUFFER = []
GLOBAL_DEPTH_BUFFER = []


def collect_n_pictures_parallel(device_ids):

  topic_img_list = ['/camera' + device_id + '/color/image_raw' for device_id in device_ids]
  topic_depth_list = ['/camera' + device_id + '/aligned_depth_to_color/image_raw' for device_id in device_ids]
  img_subs_list = [img_subscriber(topic=topic_img) for topic_img in topic_img_list]
  depth_subs_list = [depth_subscriber(topic=topic_depth) for topic_depth in topic_depth_list]
  print( "Depth Scale is: " , DEPTH_SCALE)
  rospy.sleep(1.0)
  # We will be removing the background of objects more than
  #  clipping_distance_in_meters meters away
  clipping_distance = CLIPPING_DISTANCE_IN_METERS / DEPTH_SCALE

  # Take some ramp images to allow cams to adjust for brightness etc.
  for i in range(50):
    # Get frameset of color and depth
    idx = 0
    # set_trace()
    for img_subs, depth_subs in zip(img_subs_list, depth_subs_list):
      try:
        color_image = img_subs.img
        depth_image = depth_subs.img
      except:
        print("Check connection of cameras; replug camera USB C connection (You can also run realsense-viewer and see if all cameras are visible to verify that connection is broken)")
      idx += 1
    # Warm camera up (picture quality is bad in beginning frames)
    print('Taking ramp image %d.' % i)
          
  frame_count = 0
  # Streaming loop
  start_time = time.time()
  depth_stacked = []


  try:
    cv2.namedWindow('image')
    print("Start collecting images...")
    i = 0
    while True:
      curr_time = time.time()
      color_view_buffer = []
      depth_view_buffer = []
      pixel_view_buffer = []
      if i == 0:
        input("Press Enter to take BACKGROUND image (or automatically multiple images if multiple views are enabled)")
        take_image = True
      else:
        take_image = query_yes_no("Take image? (Otherwise, stop data collection", default='yes')
      if take_image:
        view_idx = 0
        arr = np.empty((color_image.shape[0], 0, 3), color_image.dtype)
        for img_subs, depth_subs in zip(img_subs_list, depth_subs_list):

          color_image = img_subs.img
          depth_image = depth_subs.img
          depth_image[np.where(depth_image > clipping_distance)] = 0
          depth_rescaled = ((depth_image  - 0) / (clipping_distance - 0)) * (255 - 0) + 0
          depth_image_3d = depth_rescaled
          color_view_buffer.append(color_image)
          depth_view_buffer.append(depth_rescaled)
          arr = np.hstack([arr, color_image])
          view_idx += 1

        cv2.imshow('image', arr)
        cv2.waitKey(1)
        GLOBAL_IMAGE_BUFFER.append(color_view_buffer)
        GLOBAL_DEPTH_BUFFER.append(depth_view_buffer)

        i += 1
        print('took image nr {} (including background)'.format(i))
        print('------')
      else:
        break

  finally:
    pass

def main():
  # Initialize the camera capture objects.
  # Get one output directory per view.
  rospy.init_node("data_collection", disable_signals=True)
  rospy.sleep(1)
  ctx = rs.context()
  ds5_dev = rs.device()
  devices = ctx.query_devices()
  # device_ids = ['817612071456', '819112072363', '801212070655']  # Manuela's lab USB IDs
  device_ids = ['831612072676', '826212070528', '826212070219']
  # device_indices = ['1', '2', '3']
  for device in devices:
    print(device)
  device_indices = ['2'] # cross-check with IDs assigned in /home/zhouxian/ros_ws/src/realsense-2.1.0/realsense2_camera/launch/rs_multiple_devices.launch
  collect_n_pictures_parallel(device_indices)

  print("Save images to file..")
  assert len(GLOBAL_DEPTH_BUFFER) == len(GLOBAL_IMAGE_BUFFER)

  view_dirs, vid_paths, debug_path, seqname, view_dirs_depth, depth_paths, debug_path_depth = setup_paths_w_depth(args)

  offsets = []
  if not os.path.exists(vid_paths[0].split('_view')[0]):
    os.makedirs(vid_paths[0].split('_view')[0])
  if not os.path.exists(depth_paths[0].split('_view')[0]):
    os.makedirs(depth_paths[0].split('_view')[0])   

  for t in range(0, len(GLOBAL_DEPTH_BUFFER)):
    stacked_images = GLOBAL_IMAGE_BUFFER[t][0][:,:,::-1]
    for view_idx in range(len(device_indices)):
      object_name = vid_paths[view_idx].strip('.mp4').split('/')[-1].split('_view')[0]
      # if t == 0:
      #   cv2.imwrite(os.path.join('/'.join(vid_paths[0].strip('.mp4').split('/')[:-2]), 'backgrounds', '{0}_{1:06d}_view{2}.png'.format(object_name, t, view_idx)), GLOBAL_IMAGE_BUFFER[t][view_idx][:,:,::-1])
      cv2.imwrite(os.path.join(vid_paths[0].strip('.mp4').split('_view')[0], '{0}_{1:06d}_view{2}.png'.format(object_name, t, view_idx)), GLOBAL_IMAGE_BUFFER[t][view_idx][:,:,::-1])
      cv2.imwrite(os.path.join(depth_paths[0].strip('.mp4').split('_view')[0], '{0}_{1:06d}_view{2}.png'.format(object_name, t, view_idx)), GLOBAL_DEPTH_BUFFER[t][view_idx].astype(np.uint8))

  for p, q in zip(vid_paths, depth_paths):
    print('Writing final color picture to: %s' % p.strip('.mp4'))
    print('Writing final depth picture to: %s' % q.strip('.mp4'))

  try:
    sys.exit(0)
  except SystemExit:
    os._exit(0)  # pylint: disable=protected-access



if __name__ == '__main__':
  main()
