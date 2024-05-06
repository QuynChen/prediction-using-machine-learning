#!/bin/bash

train_path=$1
dev_path=$2
model_dir=$3

python main.py train --train_path "$train_path" --dev_path "$dev_path" --model_dir "$model_dir"
