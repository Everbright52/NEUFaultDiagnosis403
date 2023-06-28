!/usr/bin/env bash

echo "开始测试......"

# 定义默认值
DEFAULT_EPOCH=50
DEFAULT_BATCH_SIZE=10
DEFAULT_LEARNING_RATE=0.0001
DEFAULT_DATA_DIR="image_sst"

# 使用命令行参数或默认值
epoch=${1:-$DEFAULT_EPOCH}
batch_size=${2:-$DEFAULT_BATCH_SIZE}
learning_rate=${3:-$DEFAULT_LEARNING_RATE}
data_dir=${4:-$DEFAULT_DATA_DIR}

python main.py --model cnn --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model cacnn1 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model cacnn2 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model resnet18 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model mobilenetv2 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model mobilenetv3 --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model vit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait
python main.py --model deepvit --epoch $epoch --bs $batch_size --lr $learning_rate --data $data_dir
wait

echo "结束测试......"

#命令输入格式:   ./train_all.sh 50 64 0.001 image_sst