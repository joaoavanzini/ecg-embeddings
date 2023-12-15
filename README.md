# ecg_downloader/
$ python main.py --start_case 1 --end_case 51 --blobs_per_patient 5
- Total blobs generated across all cases: 250
- subject_x_blob_y size 30000
- all_subject_x_blobs.csv size 150000
- merged_ecg_data.csv size 7500000

# ecg_train/
$ python autoencoder_script.py --data_path ../ecg_downloader/data/merged_ecg_data.csv --input_shape 30000,1 --conv_filters 64 --conv_kernel_size 250 --dense_units 64 --epochs 100 --batch_size 128 --num_sample 250 --sample_length 30000

# ecg_inference/
$ python main_script.py --model_path ../ecg_train/autoencoder_model.h5 --file_path ../ecg_downloader/data/output_blobs_case_51/all_subject_51_blobs.csv --output_path . --num_repetitions 5
