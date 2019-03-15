### README

## TL;DR

Collect images of different objects and create synthetic data.

## (optional) 1. Collect data

### 1.1 Setup
```
# Launch ROS environment
cd ~/ros_ws && ./baxter.sh

# in separate window:
## If using one camera launch
roslaunch realsense2_camera rs_aligned_depth.launch

## If using more than one camera launch
roslaunch realsense2_camera rs_multiple_devices.launch
```

### 1.2 Collect image data

Collect the dataset by taking the first image as a background and subsequent images with ONE object added to the scene,
namely the object of choice (file will guide you through the collection).

Don't forget to set the root path in the script where the images will be saved under.

Make sure object is not too close to border of image (background subtraction assumes object is
at least somewhat centered in the middle of the image)

```
. collect_data.sh folder_name object_name

# E.g.
. collect_data.sh iccv2019 hexagon 

```


## 2. Background Subtraction
Within the newly created data folder, for example ../../iccv2019/original_data/hexagon, you should see a bunch of pictures now, 
where the 00000 indexed one should be the background view for each view (if multiple views were used, there should be multiple 00000 index images) which gets automatically saved into a backgrounds folder.

The following values are camera dependent and should be verified:

RGB_TH = 30 # Minimum rgb difference after BS to be labelled as likely object
H = 480 
W = 640
CLIPPING_DISTANCE = 1.5

The most important value for grabcut is the RGB_TH value. So if masks are not crisp enough, increase this value; if their too sharp (i.e. take away part of the object), decrease the value.

```
. create_masks.sh folder_name object_name

#e.g.:
. create_masks.sh iccv2019 hexagon
```


## 3. Synthetic Data Generation

Creates synthetic data using all available objects with the masks folders, overlaying them on all available images in the backgrounds folder.

It will take a random subset of objects (number can be specified) and put them on top of the current background image. It will iterate through all available background images, creating a certain number of synthetic images per background image.

```
. create_masks.sh folder_name output_mode

# e.g.:
. create_synthetic_images.sh iccv2019 train
```


### Useful commands
  GNU nano 2.5.3                    File: README_misc.md                                              


### Useful commands

* ls -ltrh /dev/video*  :       shows all plugged in webcame devices

* v4l2-ctl -d /dev/video<index> --list-formats  :       shows the input format of selected input chan$

* get frame count of video : ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=$

