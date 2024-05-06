#!/bin/bash

model_path=$1
test_path=$2
output_path=$3

python main.py predict --model_path "$model_path" --test_path "$test_path" --output_path "$output_path"
