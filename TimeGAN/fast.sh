#!/bin/bash

# First Python script with specified parameters
#python main_timegan.py --output_dir output/timegan24 --seq_len 24 --module gru --num_layer 4 --iteration 4000 --hidden_dim 12


# Second Python script with different parameters
python main_timegan.py --output_dir output/timegan --seq_len 24 --module gru --num_layer 4 --iteration 4000 --hidden_dim 12

#python main_timegan.py --output_dir output/cgan24 --seq_len 24 --module lstm --num_layer 3 --iteration 3000 --hidden_dim 10


# Second Python script with different parameters
python main_timegan.py --output_dir output/cgan --seq_len 24 --module lstm --num_layer 3 --iteration 3000 --hidden_dim 10
#python main_timegan.py --output_dir output/wgan24 --seq_len 24 --module lstm --num_layer 4 --iteration 3500 --hidden_dim 10


# Second Python script with different parameters
python main_timegan.py --output_dir output/wgan --seq_len 24 --module lstm --num_layer 4 --iteration 3500 --hidden_dim 10


# Wait for all background processes to finish
wait
echo "Both Python scripts have completed execution."
