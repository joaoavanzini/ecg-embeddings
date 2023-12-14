import argparse
from ecg_processing import ECGProcessor

def main():
    parser = argparse.ArgumentParser(description='Process ECG data for given case ID range.')
    parser.add_argument('--start_case', type=int, required=True, help='Starting case ID')
    parser.add_argument('--end_case', type=int, required=True, help='Ending case ID (exclusive)')
    parser.add_argument('--ecg_track_name', type=str, default='SNUADC/ECG_II', help='ECG track name')
    parser.add_argument('--blob_duration', type=int, default=310, help='Blob duration')
    parser.add_argument('--interval', type=float, default=1/100, help='Interval')
    parser.add_argument('--blobs_per_patient', type=int, default=10, help='Number of blobs per patient')

    args = parser.parse_args()

    ecg_processor = ECGProcessor(
        case_id_range=(args.start_case, args.end_case),
        ecg_track_name=args.ecg_track_name,
        blob_duration=args.blob_duration,
        interval=args.interval,
        blobs_per_patient=args.blobs_per_patient
    )
    ecg_processor.run_processing()

if __name__ == "__main__":
    main()