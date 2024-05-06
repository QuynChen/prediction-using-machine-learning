#!/bin/bash

train_path=$1
dev_path=$2

python main.py train_models --train_path "$train_path" --dev_path "$dev_path"
