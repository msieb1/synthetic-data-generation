dataset=$1  # Name of the dataset.
mode=$2 # output mode: train, val, test
expdir=/media/data/iccv2019/synthetic_data_generation
maskdir="masks" #assumes a copy of each original image to be in the masks folder!
backgrounddir="backgrounds"
export DISPLAY=:0.0  # This allows real time matplotlib display.



python src/create_synthetic_images_aug.py \
--dataset $dataset \
--expdir $expdir \
--maskdir $maskdir \
--backgrounddir $backgrounddir \
--mode $mode \