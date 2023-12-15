import os
import pandas as pd
import numpy as np
import vitaldb

class ECGProcessor:
    def __init__(self, case_id_range, ecg_track_name, blob_duration, interval, blobs_per_patient):
        self.case_id_range = case_id_range
        self.ecg_track_name = ecg_track_name
        self.blob_duration = blob_duration
        self.interval = interval
        self.blobs_per_patient = blobs_per_patient

    def process_case(self, case_id):
        blob_size = int(self.blob_duration / self.interval)

        vf = vitaldb.VitalFile(case_id, [self.ecg_track_name])
        ecg_signal = vf.to_numpy([self.ecg_track_name], interval=self.interval)[:, 0]

        ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

        blobs = [ecg_signal[i:i + blob_size][:30000] for i in range(0, min(self.blobs_per_patient * blob_size, len(ecg_signal)), blob_size)]

        output_dir = f"./data/output_blobs_case_{case_id}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i, blob in enumerate(blobs, 1):
            blob_df = pd.DataFrame({'ECG': blob})
            blob_df.to_csv(f"{output_dir}/subject_{case_id}_blob_{i}.csv", index=False)

        all_blobs_df = pd.concat([pd.read_csv(f"{output_dir}/subject_{case_id}_blob_{i}.csv") for i in range(1, len(blobs) + 1)], ignore_index=True)
        all_blobs_df.to_csv(f"{output_dir}/all_subject_{case_id}_blobs.csv", index=False)

        return blobs

    def run_processing(self):
        all_dataframes = []
        total_blobs_generated = 0
    
        for case_id in range(*self.case_id_range):
            blobs = self.process_case(case_id)
            total_blobs_generated += len(blobs)
            print(f"Case ID {case_id}: Total number of generated blobs: {len(blobs)}")
    
            if len(blobs) < self.blobs_per_patient:
                print(f"Warning: Insufficient blobs found for case {case_id}. Trying the next case...")
                adjusted_end_case = self.case_id_range[1] + 1
                print(f"Adjusting end_case to {adjusted_end_case}")
                self.case_id_range = (self.case_id_range[0], adjusted_end_case)
                continue
    
            for i, blob in enumerate(blobs, 1):
                blob_file_path = f"./data/output_blobs_case_{case_id}/subject_{case_id}_blob_{i}.csv"
                if os.path.exists(blob_file_path):
                    blob_df = pd.read_csv(blob_file_path)
                    all_dataframes.append(blob_df)
                else:
                    print(f"Warning: Blob file not found for case {case_id}, blob {i}")
    
        if all_dataframes:
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            merged_df.to_csv("./data/merged_ecg_data.csv", index=False)
    
            print(f"Total blobs generated across all cases: {total_blobs_generated}")
            print("Export completed.")
        else:
            print("No blobs found. Check the input data.")