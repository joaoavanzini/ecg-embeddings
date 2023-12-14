import argparse
import os
from case_info_fetcher import CaseInfoFetcher

def main():
    parser = argparse.ArgumentParser(description='Fetch and process case information.')
    parser.add_argument('--start_case', type=int, help='Start subject ID for filtering')
    parser.add_argument('--end_case', type=int, help='End subject ID for filtering')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path')
    
    args = parser.parse_args()

    case_info_fetcher = CaseInfoFetcher(api_url="https://api.vitaldb.net/cases")
    case_info_fetcher.all_cases_info = case_info_fetcher.get_all_case_info()

    if case_info_fetcher.all_cases_info is not None:
        selected_columns = ['caseid', 'sex', 'age', 'height', 'weight', 'bmi']

        df_info_selected = case_info_fetcher.select_columns(selected_columns)

        start_case = args.start_case if args.start_case else df_info_selected['caseid'].min()
        end_case = args.end_case if args.end_case else df_info_selected['caseid'].max()

        selected_subjectids = list(range(start_case, end_case + 1))
        df_info_selected_subjects = case_info_fetcher.filter_subjects(selected_subjectids)

        output_dir = './data/'
        os.makedirs(output_dir, exist_ok=True)

        if args.output_csv:
            output_csv_path = os.path.join(output_dir, args.output_csv)
        else:
            output_csv_path = os.path.join(output_dir, 'output.csv')

        df_info_selected_subjects.to_csv(output_csv_path, columns=selected_columns, index=False)
        print(f'Data saved to {output_csv_path}')
    else:
        print("Unable to retrieve case information.")

if __name__ == "__main__":
    main()
