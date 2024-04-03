import os
import re

import pandas as pd


class InnerCVReader:
    RESULTS_DIR = 'assets/results'

    @classmethod
    def read_results(cls):
        files = cls.get_all_files()

        results_l = []
        for file in files:
            print(file)
            if file == '.DS_Store':
                continue

            cv_ = pd.read_csv(f'{cls.RESULTS_DIR}/{file}')
            group = re.sub('_inner.csv$', '', file)
            cv_['group'] = group
            cv_['freq'] = group.split('_')[-1].title()

            results_l.append(cv_)

        results_df = pd.concat(results_l).reset_index(drop=True)

        return results_df

    @classmethod
    def get_all_files(cls):
        files = os.listdir(cls.RESULTS_DIR)
        return files

