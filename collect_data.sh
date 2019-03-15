# kill -9 $(lsof -t /dev/video0) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video1) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video2) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video3) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video4) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video5) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video6) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video7) # Make sure port is freed up, somehow D435 port stays busy after terminating run
# kill -9 $(lsof -t /dev/video8) # Make sure port is freed up, somehow D435 port stays busy after terminating run


##### Sorted by audio, depth, video #####


##### Sorted by experiments and depth, audio, video are subfolders #####
dataset=$1  # Name of the dataset.
seqname="$2" # Name of Object!
num_views=2 # Number of webcams.
expdir=/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/synthetic_data_generation
viddir=images # Output directory for the videos.
depthdir=depth
tmp_imagedir=/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/synthetic_data_generation/tmp
# Temp directory to hold images.
debugvids=0 # Whether or not to generate side-by-side debug videos.

export DISPLAY=:0.0  # This allows real time matplotlib display.



python src/data_collection.py \
--dataset $dataset \
--num_views $num_views \
--tmp_imagedir $tmp_imagedir \
--viddir $viddir \
--depthdir $depthdir \
--debug_vids $debugvids \
--seqname $seqname \
--expdir $expdir \
# python test.py


