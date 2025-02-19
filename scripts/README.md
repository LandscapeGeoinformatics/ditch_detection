# Scripts

Subdirectory `data_prep` contains Jupyter notebooks for the following steps:
1. Downloading 1 m DTM data from the Estonian Land Board (`download_elevation_data.ipynb`)
2. Generating HPMF rasters from the DTM data with Whitebox (`generate_hpmf_rasters_wbt.ipynb`)
3. Normalizing the HPMF rasters (`normalize_hpmf_rasters.ipynb`)
4. Preparing data for pre-training (`prepare_pretraining_data.ipynb`)
5. Preparing data for fine-tuning (`prepare_finetuning_data.ipynb`)
6. Augmenting data for pre-training (`augment_pretraining_data.ipynb`)
7. Augmenting data for fine-tuning (`augment_finetuning_data.ipynb`)

Subdirectory `pytorch_unet` contains the following files:
1. Metrics used to measure model loss (`model/CustomLoss.py`)
2. Metrics used to measure model performance (`model/CustomMetric.py`)
3. U-Net model architecture (`model/unet.py`)
4. Procedure for loading data into the model (`utils/dataloader.py`)
5. Procedure for post-processing predictions (`utils/datapostprocessing.py`)
6. Script for training and testing the U-Net (`DLRunner.py`)

Other scripts:
- Calculating accuracy metrics per land use class (`calc_f1_per_land_use_class.ipynb`)
- Extracting training time information (`extract_training_time_stats.ipynb`)
- Extracting best results for a model configuration from the epoch with lowest loss (`get_best_results.py`)
- Extracting parameters for fine-tuning based on pre-training results (`get_finetuning_params.py`)
- Plotting examples of digitized tiles (`plot_examples_of_digitized_tiles.ipynb`)
- Plotting examples of predictions (`plot_examples_of_predictions.ipynb`)
- Plotting pre-training and fine-tuning results (`plot_experiment_results.ipynb`)
- Converting predictions to GeoTIFFs (`process_predictions.ipynb`)