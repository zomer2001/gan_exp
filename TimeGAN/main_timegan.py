from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings
import os

# TimeGAN model and utilities
from timegan import timegan
from data_loading import real_data_loading2, sine_data_generation
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

warnings.filterwarnings("ignore")

def main(args):
    """Main function for TimeGAN experiments with sparsity rates.

    Args:
        - data_name: sine, stock, or energy
        - seq_len: sequence length
        - module: gru, lstm, or lstmLN
        - hidden_dim: hidden dimensions
        - num_layer: number of layers
        - iteration: training iterations
        - batch_size: number of samples in mini-batch
        - metric_iteration: iterations for metric computation
        - area: number of time steps to select
        - start: starting index for data selection
        - output_dir: directory to save generated data
    """
    # Define sparsity rates and data parameters
    #sparsity_rates = [2.0, 4.0, 8.0]
    sparsity_rates = [0.9,0.7,0.5,0.3,2.0, 4.0, 8.0]
    data_subfolders = ['720','2160','4320']

    data_root = 'train_data'

    for subfolder in data_subfolders:
        data_folder = os.path.join(data_root, subfolder)
        if not os.path.exists(data_folder):
            print(f"Data folder {data_folder} does not exist, skipping.")
            continue

        csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
        for csv_file in csv_files:
            print(f"\nProcessing file: {csv_file} in {data_folder}")
            for sparsity in sparsity_rates:
                sparsity_percentage = int(sparsity * 100)
                # Create output directory
                output_dir = os.path.join(args.output_dir, f'{sparsity_percentage}', 'train')
                os.makedirs(output_dir, exist_ok=True)
                print(f"Output directory for sparsity {sparsity_percentage}: {output_dir}")

                # Check if the generated file already exists
                base_name = os.path.splitext(csv_file)[0]
                output_path = os.path.join(output_dir, f"generated_{base_name}.npy")
                if os.path.exists(output_path):
                    print(
                        f"Generated file {output_path} already exists. Skipping training for {csv_file} at sparsity {sparsity_percentage}.")
                    continue  # Skip this csv_file if the generated data already exists

                # Load real data with sparsity-based selection
                try:
                    filepath = os.path.join(data_folder, csv_file)
                    ori_data = real_data_loading2(
                        filepath,
                        seq_len=args.seq_len,
                        proportion=sparsity
                    )
                    print(
                        f'{csv_file} dataset loaded successfully for sparsity {sparsity_percentage}: {len(ori_data)} samples')
                except Exception as e:
                    print(f"Error loading {csv_file} for sparsity {sparsity_percentage}: {str(e)}")
                    continue

                # Convert to numpy array for timegan
                ori_data = np.array(ori_data)
                if ori_data.shape[0] == 0:
                    print(f"No samples available for {csv_file} at sparsity {sparsity_percentage}, skipping.")
                    continue

                # Set network parameters for TimeGAN
                parameters = {
                    'module': args.module,
                    'hidden_dim': args.hidden_dim,
                    'num_layer': args.num_layer,
                    'iterations': args.iteration,
                    'batch_size': args.batch_size
                }

                # Generate synthetic data
                try:
                    generated_data = timegan(ori_data, parameters)
                    print(f'Synthetic data generation completed for {csv_file} at sparsity {sparsity_percentage}')
                except Exception as e:
                    print(f"Error generating data for {csv_file} at sparsity {sparsity_percentage}: {str(e)}")
                    continue

                # Save generated data
                try:
                    np.save(output_path, generated_data)
                    print(f"Data saved to {output_path}")
                except Exception as e:
                    print(f"Error saving data for {csv_file} at sparsity {sparsity_percentage}: {str(e)}")

                # Evaluate metrics
                try:
                    # Visualization
                    visualization(ori_data, generated_data, 'pca', output_dir,
                                  base_name + f'_sparsity_{sparsity_percentage}')
                    visualization(ori_data, generated_data, 'tsne', output_dir,
                                  base_name + f'_sparsity_{sparsity_percentage}')
                    # Discriminative score
                    discriminative_score = discriminative_score_metrics(ori_data, generated_data)
                    print(
                        f"Discriminative score for {csv_file} at sparsity {sparsity_percentage}: {discriminative_score}")
                    # Predictive score
                    predictive_score = predictive_score_metrics(ori_data, generated_data)
                    print(f"Predictive score for {csv_file} at sparsity {sparsity_percentage}: {predictive_score}")
                except Exception as e:
                    print(f"Error evaluating metrics for {csv_file} at sparsity {sparsity_percentage}: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy'],
        default='energy',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions',
        default=12,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations',
        default=3000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)
    parser.add_argument(
        '--area',
        help='number of time steps to select',
        default=720,
        type=int)
    parser.add_argument(
        '--start',
        help='starting index for data selection',
        default=10000,
        type=int)
    parser.add_argument(
        '--output_dir',
        help='Directory to save generated data',
        default='output/timegan',  # Default value if not provided
        type=str)

    args = parser.parse_args()

    # Validate parameters
    if args.area <= args.seq_len:
        raise ValueError("Area parameter must be greater than seq_len")

    main(args)
