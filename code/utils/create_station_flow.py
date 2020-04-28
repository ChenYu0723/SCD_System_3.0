# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

path = os.path.abspath('../..')

infile = path + '/data/raw_data/metroData_ODflow_15.csv'
raw_data = pd.read_csv(infile)
# print(raw_data)

infile2 = path + '/data/raw_data/metroStations.csv'
sta_info = pd.read_csv(infile2)
# print(sta_info)
sta_id_ls = sorted(list(sta_info['stationID']))
# print(sta_id_ls)
# print(len(sta_id_ls))

inFlow_df = pd.DataFrame(columns=sta_id_ls)
all_ts = int(len(raw_data)/322)
for t in range(all_ts):
    df = raw_data[raw_data[' timeslot']==t]
    ls = list(df[' inFlow'])
    inFlow_df.loc[t] = ls
    if t % 100 == 0:
        print('dealed:', t, 'all:', all_ts)

# print(inFlow_df)
inFlow_df.to_csv('/home/chen/Pycharm Projects/ITS/SCD_System_2.0/data/true_data/station_inFlow.csv', index=False)