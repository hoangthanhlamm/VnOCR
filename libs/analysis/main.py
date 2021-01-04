from libs.analysis.analysis import DataAnalysis
from config import csv_path

import os


def main():
    datasets = ['train.csv', 'val.csv', 'test.csv']

    print("Analysis...\n")
    for dataset in datasets:
        print("Dataset: ", dataset)
        path = os.path.join(csv_path, dataset)

        analysis = DataAnalysis(path)
        analysis.width_analysis()
        print('-------------------------')
        analysis.length_analysis()

    print('==============================')


if __name__ == '__main__':
    main()
