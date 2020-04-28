# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 13:54
# @Author  : Chen Yu

import os
import pickle
import numpy as np
import pandas as pd

os.chdir('../..')


def create_od(filter_hour=None):
    infile = "data/raw_data/metroData_filtered.csv"
    infile2 = 'data/raw_data/transferStations.pkl'
    transferStation_info = pickle.load(open(infile2, 'rb'))
    infile3 = 'data/raw_data/metroStations.csv'
    station_id_df = pd.read_csv(infile3)
    station_id_ls = sorted(list(station_id_df['stationID']))
    od_matrix_df = pd.DataFrame(0, index=station_id_ls, columns=station_id_ls)

    startDay = 20170501
    endDay = 20170831
    startHour = 6
    endHour = 22

    user_count = 0
    trip_count = 0
    with open(infile, 'r') as f:
        for line in f:
            line = line.rstrip().split(',')
            if len(line) == 1:
                user_count += 1
                if user_count % 100 == 0:
                    print('user count:', user_count)
                continue
            transDay = line[0]
            if int(transDay) < startDay or int(transDay) > endDay:
                continue
            transTime = line[1]
            transHour = int(transTime[:2])
            if filter_hour is None:
                if transHour < startHour or transHour >= endHour:
                    continue
            else:
                if transHour != filter_hour:
                    continue

            inStation = int(line[2])
            outStation = int(line[3])
            try:
                inStation = transferStation_info[1][inStation]
            except KeyError:
                pass
            try:
                outStation = transferStation_info[1][outStation]
            except KeyError:
                pass

            if (inStation not in od_matrix_df.index) or (outStation not in od_matrix_df.index):
                continue
            od_matrix_df.loc[inStation, outStation] += 1
            trip_count += 1

    print('all user count:', user_count)
    print('all trip count:', trip_count)
    if filter_hour is None:
        od_matrix_df.to_csv('data/od_matrix_all.csv', header=False, index=False)
    else:
        od_matrix_df.to_csv('data/od_matrix_filtered/od_matrix_%d.csv' % filter_hour, header=False, index=False)


# ==== normalize od matrix
def normalize(data):
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return np.array([(i - mn) / (mx - mn) for i in data])


def normalize_od():
    infile = 'data/od_matrix_all.csv'
    infile = 'data/od_matrix_filtered/od_matrix_12.csv'
    od_df = pd.read_csv(infile, header=None)
    # print(od_df)
    od_mx = np.array(od_df)
    print(od_mx)
    od_mx_nor = normalize(od_mx)
    print(od_mx_nor)
    od_df_nor = pd.DataFrame(od_mx_nor)
    print(od_df_nor)

    # od_df_nor.to_csv('data/od_matrix_all_nor.csv', header=False, index=False)
    od_df_nor.to_csv('data/od_matrix_filtered/od_matrix_12_nor.csv', header=False, index=False)


def od_to_0_1():
    infile = 'data/od_matrix_all_nor.csv'
    infile = 'data/od_matrix_filtered/od_matrix_12_nor.csv'
    od_df = pd.read_csv(infile, header=None)
    od_mx = np.array(od_df)
    od_mx[od_mx >= .3] = 1
    od_mx[od_mx < .3] = 0
    od_mx = od_mx.T
    print(od_mx)
    od_df_0_1 = pd.DataFrame(od_mx)
    # od_df_0_1.to_csv('data/od_matrix_all_nor_0_1.csv', header=False, index=False)
    od_df_0_1.to_csv('data/od_matrix_filtered/od_matrix_12_nor_0_1.csv', header=False, index=False)


def create_flow(filter_hour):
    all_flow = pd.read_csv('data/station_inFlow.csv')
    filter_sta_ts = (filter_hour - 6) * 4
    filter_end_ts = (filter_hour + 1 - 6) * 4 - 1
    filter_flow = pd.DataFrame(columns=all_flow.columns)

    for i in all_flow.index:
        c = i % 64
        if (c >= filter_sta_ts) and (c <= filter_end_ts):
            filter_flow = filter_flow.append(all_flow.iloc[i])
    filter_flow.to_csv('data/station_flow_filtered/station_inFlow_%d.csv' % filter_hour, index=False)


if __name__ == '__main__':
    '''
    all user count: 102020
    all trip count: 17008644
    '''
    # create_od(12)
    # normalize_od()
    # od_to_0_1()

    # create_flow(12)
