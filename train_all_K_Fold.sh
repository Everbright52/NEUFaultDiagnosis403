!/usr/bin/env bash

echo "开始测试......"

# 定义默认值
DEFAULT_EPOCH=3
DEFAULT_BATCH_SIZE=32
DEFAULT_LEARNING_RATE=0.0001
DEFAULT_DATA_DIR="image_sst"
DEFAULT_NUM_FOLDS=14
DEFAULT_NUM_RUNS=10

# 使用命令行参数或默认值
epoch=${1:-$DEFAULT_EPOCH}
batch_size=${2:-$DEFAULT_BATCH_SIZE}
learning_rate=${3:-$DEFAULT_LEARNING_RATE}
data_dir=${4:-$DEFAULT_DATA_DIR}
num_folds=${5:-$DEFAULT_NUM_FOLDS}
num_runs=${6:-$DEFAULT_NUM_RUNS}


#python main_K_Fold.py --model cnn --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model cacnn1 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model cacnn2 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model resnet18 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model mobilenetv2 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model mobilenetv3 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model vit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait
#python main_K_Fold.py --model deepvit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
#wait

python main_K_Fold.py --model mobilenetv2 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
wait
python main_K_Fold.py --model mobilenetv3 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
wait
python main_K_Fold.py --model vit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
wait
python main_K_Fold.py --model deepvit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir --fold $num_folds --run $num_runs
wait



echo "结束测试......"

#命令输入格式:   ./train_all.sh