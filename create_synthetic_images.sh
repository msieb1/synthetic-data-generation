dataset=$1  # Name of the dataset.
mode=$2 # output mode: train, val, test
expdir=/media/zhouxian/ed854110-6801-4dcd-9acf-c4f904955d71/iccv2019/synthetic_data_generation
maskdir="masks" #assumes a copy of each original image to be in the masks folder!
backgrounddir="backgrounds"
export DISPLAY=:0.0  # This allows real time matplotlib display.



python src/create_synthetic_images.py \
--dataset $dataset \
--expdir $expdir \
--maskdir $maskdir \
--backgrounddir $backgrounddir \
--mode $mode \