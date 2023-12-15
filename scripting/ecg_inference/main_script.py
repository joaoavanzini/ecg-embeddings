import argparse
import os
import pandas as pd
from subject_processor import SubjectProcessor

def main():
    parser = argparse.ArgumentParser(description='Process subjects using an encoder model and save embeddings.')
    parser.add_argument('--model_path', type=str, help='Path to the encoder model file (in .h5 format)')
    parser.add_argument('--file_path', type=str, help='Path to the inference data file (in .csv format)')
    parser.add_argument('--output_path', type=str, help='Path to save the embeddings CSV file')
    parser.add_argument('--num_repetitions', type=int, default=10, help='Number of repetitions in the data')
    parser.add_argument('--sequence_length', type=int, default=30000, help='Length of each sequence')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels in the data')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    processor = SubjectProcessor(args.model_path, args.file_path, args.num_repetitions, args.sequence_length, args.num_channels)
    embeddings = processor.process_subject()

    embeddings_df = pd.DataFrame([embeddings])
    embeddings_df.to_csv(os.path.join(args.output_path, 'encoder_embeddings.csv'), index=False)

if __name__ == "__main__":
    main()
