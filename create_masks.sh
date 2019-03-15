dataset=$1  # Name of the dataset.
target=$2 # Name of Object!
expdir=/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/synthetic_data_generation
imagedir="images" # Output directory for the videos.
depthdir="depth"

export DISPLAY=:0.0  # This allows real time matplotlib display.



python src/create_masks.py \
--dataset $dataset \
--target $target \
--imagedir $viddir \
--depthdir $depthdir \
--expdir $expdir \
