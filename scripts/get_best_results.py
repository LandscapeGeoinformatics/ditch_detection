import os
import argparse

import pandas as pd


# Convert seconds to HH:MM:SS
def convert_seconds(seconds):
    hours = int(round(seconds // 3600, 0))
    minutes = int(round((seconds % 3600) // 60, 0))
    remaining_seconds = int(round(seconds % 60, 0))
    return f'{hours}:{minutes}:{remaining_seconds}'


# Get path to best weights
def get_best_weight_pkl_path(best_epoch, experiment_dir='.'):
    
    # Path to best weights
    for file in os.listdir(f'{experiment_dir}/weight'):
        epoch = int(file.split('_')[3])
        if epoch == best_epoch:
            best_weight_pkl = file
    experiment = os.path.basename(os.path.abspath(experiment_dir))
    best_weight_pkl_path = f'{os.path.abspath(experiment_dir)}/weight/{best_weight_pkl}'
    
    return best_weight_pkl_path



# Create DataFrame with best results
def create_best_results_df(experiment_dir='.'):
    
    # Read results
    result_pkl_path = f'{experiment_dir}/result/result.pkl'
    results = pd.read_pickle(result_pkl_path)
    results['BinaryF1Score'] = results['BinaryF1Score'].astype('float')
    results['BinaryAccuracy'] = results['BinaryAccuracy'].astype('float')
    results['BinaryPrecision'] = results['BinaryPrecision'].astype('float')
    results['BinaryRecall'] = results['BinaryRecall'].astype('float')
    
    # Get temporal information
    train_time_seconds = results.loc[results['Train_Test'] == 'Train']['time(s)'].sum()
    train_time = convert_seconds(train_time_seconds)
    test_time_seconds = results.loc[results['Train_Test'] == 'Test']['time(s)'].sum()
    test_time = convert_seconds(test_time_seconds)
    total_time_seconds = results['time(s)'].sum()
    total_time = convert_seconds(total_time_seconds)
    
    # Extract test results and sort by loss and F1 score
    test_results = results[results['Train_Test'] == 'Test'].reset_index(drop=True)
    test_results['BinaryF1Score'] = test_results['BinaryF1Score'].astype(float)
    test_results = test_results.sort_values(by=['loss', 'BinaryF1Score'], ascending=[True, False])
    
    # Get best epoch
    best_epoch = test_results.head(1)['Epoch'].values[0]
    
    # Extract best results
    best_results = results[results['Epoch'] == best_epoch]
    
    # Path to best weights
    best_weight_pkl_path = get_best_weight_pkl_path(best_epoch, experiment_dir)
    
    # Output dictionary for results
    out_results = {
        'experiment': os.path.basename(os.path.abspath(experiment_dir)),
        'best_epoch': best_epoch,
        'best_weight_pkl_path': best_weight_pkl_path,
        'total_time_seconds': total_time_seconds,
        'total_time': total_time,
        'train_loss': best_results.loc[best_results['Train_Test'] == 'Train', 'loss'].values[0],
        'train_f1': best_results.loc[best_results['Train_Test'] == 'Train', 'BinaryF1Score'].values[0],
        'train_accuracy': best_results.loc[best_results['Train_Test'] == 'Train', 'BinaryAccuracy'].values[0],
        'train_precision': best_results.loc[best_results['Train_Test'] == 'Train', 'BinaryPrecision'].values[0],
        'train_recall': best_results.loc[best_results['Train_Test'] == 'Train', 'BinaryRecall'].values[0],
        'train_time_seconds': train_time_seconds,
        'train_time': train_time,
        'test_loss': best_results.loc[best_results['Train_Test'] == 'Test', 'loss'].values[0],
        'test_f1': best_results.loc[best_results['Train_Test'] == 'Test', 'BinaryF1Score'].values[0],
        'test_accuracy': best_results.loc[best_results['Train_Test'] == 'Test', 'BinaryAccuracy'].values[0],
        'test_precision': best_results.loc[best_results['Train_Test'] == 'Test', 'BinaryPrecision'].values[0],
        'test_recall': best_results.loc[best_results['Train_Test'] == 'Test', 'BinaryRecall'].values[0],
        'test_time_seconds': test_time_seconds,
        'test_time': test_time
    }
    
    # Convert dictionary to DataFrame
    out_df = pd.DataFrame([out_results])
    
    return out_df


def main():
    
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='.')
    parser.add_argument('--out_dir', type=str, default='.')
    
    # Parse arguments
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    out_dir = args.out_dir
    
    # Create DataFrame with best results
    best_results_df = create_best_results_df(experiment_dir)
    
    # Write to CSV
    best_results_df.to_csv(f'{out_dir}/best_results.csv', index=False)


if __name__ == '__main__':
    main()
