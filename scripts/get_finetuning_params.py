import os
import argparse

import pandas as pd


# Generate name for finetuned model based on experiment
def generate_finetuned_model_name(experiment_dir='.'):
    experiment = os.path.basename(os.path.abspath(experiment_dir))
    experiment_name_parts = experiment.split('_')
    train_data_part = '_'.join([experiment_name_parts[0], experiment_name_parts[1], experiment_name_parts[2]])
    test_data_part = '_'.join([experiment_name_parts[3], experiment_name_parts[4], experiment_name_parts[5]])
    activation_part = experiment_name_parts[6]
    kernel_size_part = experiment_name_parts[7]
    lr_part = experiment_name_parts[8]
    n_epochs_part = experiment_name_parts[9]
    batch_size_part = experiment_name_parts[10]
    finetuned_model_name = '_'.join(
        [
            'finetuned',
            'aug',
            'estonia',
            activation_part,
            kernel_size_part,
            lr_part,
            n_epochs_part,
            batch_size_part,
            'from',
            train_data_part,
            test_data_part
        ]
    )
    return finetuned_model_name


def main():
    
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default='.')
    
    # Parse arguments
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    out_dir = args.out_dir
    
    # Get finetuning parameters
    finetuned_model_name = generate_finetuned_model_name(experiment_dir)
    best_results = pd.read_csv(f'{experiment_dir}/best_results.csv')
    best_weight_pkl = os.path.basename(best_results['best_weight_pkl_path'].values[0])
    experiment = os.path.basename(os.path.abspath(experiment_dir))
    config_weight_pkl_path = f'./experiments/{experiment}/weight/{best_weight_pkl}'
    config_template_path = f'{os.path.abspath(experiment_dir)}/config.yml'
    
    # Create DataFrame with finetuning parameters
    finetuning_params_df = pd.DataFrame(
        [
            {
                'finetuned_model_name': finetuned_model_name,
                'config_weight_pkl_path': config_weight_pkl_path,
                'config_template_path': config_template_path
            }
        ]
    )
    
    # Write to CSV
    finetuning_params_df.to_csv(f'{out_dir}/finetuning_params.csv', index=False)


if __name__ == '__main__':
    main()
