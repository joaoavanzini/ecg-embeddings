import requests
import pandas as pd
from io import StringIO

class CaseInfoFetcher:
    def __init__(self, api_url):
        self.api_url = api_url
        self.all_cases_info = None

    def get_all_case_info(self):
        response = requests.get(self.api_url)

        if response.status_code == 200:
            csv_text = response.text.lstrip('\ufeff')
            df_info = pd.read_csv(StringIO(csv_text))
            df_info = df_info.fillna("Invalid Value (NaN)")
            return df_info
        else:
            print(f"Error retrieving case information. Status code: {response.status_code}")
            return None

    def select_columns(self, selected_columns):
        if self.all_cases_info is not None:
            return self.all_cases_info[selected_columns]
        else:
            return None

    def filter_subjects(self, selected_subjectids):
        if self.all_cases_info is not None:
            return self.all_cases_info[self.all_cases_info['caseid'].isin(selected_subjectids)]
        else:
            return None

    def create_subjects_table(self, selected_columns):
        if self.all_cases_info is not None:
            df_info_selected_subjects_table = self.all_cases_info[selected_columns]
            df_info_selected_subjects_table = df_info_selected_subjects_table.rename(columns={'caseid': 'id'})
            return df_info_selected_subjects_table
        else:
            return None
