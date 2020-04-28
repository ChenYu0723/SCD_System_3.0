# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np

pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)

os.chdir('../..')

infile = 'data/raw_data/metroData_ODflow_15.csv'
raw_data = pd.read_csv(infile)
# print(raw_data)

infile2 = 'data/raw_data/metroStations.csv'
sta_info = pd.read_csv(infile2)
# print(sta_info)
sta_id_ls = sorted(list(sta_info['stationID']))
# print(sta_id_ls)
# print(len(sta_id_ls))


# ==== 选择生成inflow 还是 outflow ====
# choose_flow = 'inflow'
choose_flow = 'outflow'


flow_df = pd.DataFrame(columns=sta_id_ls)
all_ts = int(len(raw_data)/322)
for t in range(all_ts):
    df = raw_data[raw_data[' timeslot'] == t]
    if choose_flow == 'inflow':
        ls = list(df[' inFlow'])
    elif choose_flow == 'outflow':
        ls = list(df[' outFlow'])
    flow_df.loc[t] = ls
    if t % 100 == 0:
        print('dealed:', t, 'all:', all_ts)

# print(Flow_df)
# flow_df.to_csv('data/station_flow/station_inFlow.csv', index=False)
flow_df.to_csv('data/station_flow/station_outFlow.csv', index=False)
