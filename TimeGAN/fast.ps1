# Run First Python script with specified parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/timegan24 --seq_len 24 --module gru --num_layer 4 --iteration 4000 --hidden_dim 12" -NoNewWindow

# Run Second Python script with different parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/timegan48 --seq_len 48 --module gru --num_layer 4 --iteration 4000 --hidden_dim 12" -NoNewWindow

# Run Third Python script with different parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/cgan24 --seq_len 24 --module rnn --num_layer 3 --iteration 3000 --hidden_dim 10" -NoNewWindow

# Run Fourth Python script with different parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/cgan48 --seq_len 48 --module rnn --num_layer 3 --iteration 3000 --hidden_dim 10" -NoNewWindow

# Run Fifth Python script with different parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/wgan24 --seq_len 24 --module rnn --num_layer 4 --iteration 3500 --hidden_dim 10" -NoNewWindow

# Run Sixth Python script with different parameters
Start-Process python -ArgumentList "main_timegan.py --output_dir output/wgan48 --seq_len 48 --module rnn --num_layer 4 --iteration 3500 --hidden_dim 10" -NoNewWindow

# Wait for all background processes to finish
Write-Host "All Python scripts have completed execution."
