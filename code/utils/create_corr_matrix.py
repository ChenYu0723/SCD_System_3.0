# -*- coding: utf-8 -*-
# File Name:  create_corr_matrix.py
# Date:       2020/5/4 4:21 下午
# Author:     yuchen

import os
import numpy as np
import pandas as pd

os.chdir('../..')


def create_corr_matrix():
    all_flow = pd.read_csv('data/station_flow/station_inFlow.csv')
    print(all_flow)
    corr_matrix = all_flow.corr()
    print(corr_matrix)
    corr_matrix[corr_matrix.isnull()] = 0
    corr_matrix = corr_matrix.abs()
    print(corr_matrix)
    corr_matrix.to_csv('data/corr_matrix/inFlow_corr_matrix.csv', header=False, index=False)


# ==== normalize matrix
def normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return np.array([(i - mn) / (mx - mn) for i in data])


def normalize_corr():
    infile = 'data/corr_matrix/inFlow_corr_matrix.csv'
    od_df = pd.read_csv(infile, header=None)
    # print(od_df)
    od_mx = np.array(od_df)
    print(od_mx)
    od_mx_nor = normalize(od_mx)
    print(od_mx_nor)
    od_df_nor = pd.DataFrame(od_mx_nor)
    print(od_df_nor)

    od_df_nor.to_csv('data/corr_matrix/inFlow_corr_matrix_nor.csv', header=False, index=False)


if __name__ == '__main__':
    create_corr_matrix()
    # normalize_corr()
    print('end')
