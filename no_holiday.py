# -*- coding:utf-8 -*-
import pandas as pd
import datetime
import numpy as np

# frame = pd.read_csv( 'E:/school_compet/no_holiday_all.csv')


def get_data(frame, time_stamp, loc_id):
    week = time_stamp.day // 7
    week_day = time_stamp.weekday()
    hour = time_stamp.hour
    # if week == 4:
    #     week = 3
    x = frame[(frame["week"] == week) & (frame["week_day"] == week_day) & (frame["hour"] == hour) & (frame["loc_id"] == loc_id)]
    x = x.sort_values(by="month")["num_of_people"].values.tolist()
    # print('get_data==: ',x)
    x = noise(x)
    return x

def get_week_data(frame, time_stamp, loc_id):
    week_day = time_stamp.weekday()
    hour = time_stamp.hour
    x = frame[(frame["week_day"] == week_day) & (frame["hour"] == hour) & (frame["loc_id"] == loc_id)]
    x = x.sort_values(by=["month",'week'])["num_of_people"].values.tolist()
    # print('get_data==: ',x)
    print('has noise: ', x)
    x = noise(x)
    print('has no noise: ', x)
    return x

def get_week_3months_data(frame,time_stamp,loc_id):
    week_day = time_stamp.weekday()
    hour = time_stamp.hour
    x = frame[(frame['month']==9)&(frame['week']>1)&(frame["week_day"] == week_day) & (frame["hour"] == hour) & (frame["loc_id"] == loc_id)]
    y = frame[(frame['month']>9)&(frame["week_day"] == week_day) & (frame["hour"] == hour) & (frame["loc_id"] == loc_id)]
    x = x.append(y,ignore_index=False)
    x = x.sort_values(by=["month",'week'])["num_of_people"].values.tolist()
    # print('get_data==: ',x)
    print('has noise: ', x)
    x = noise(x)
    print('has no noise: ', x)
    return x

def noise(data):

    rows = len(data)
    if rows <4:
        return data
    q1 = 3 * (rows + 1) / 4
    q3 = 1 * (rows + 1) / 4
    temp = list(np.argsort(data))[::-1]
    percent = q1 - int(q1)
    begin = int(q1)-1
    end = int(q1)
    Q1 = data[temp[begin]] * (percent) + data[temp[end]] * (1 - percent)
    percent = q3 - int(q3)
    begin = int(q3)-1
    end = int(q3)
    Q3 = data[temp[begin]]* (percent) + data[temp[end]] * (1 - percent)
    IQR = Q3 - Q1
    while IQR==0 and end<rows:
        IQR = data[temp[end]]-Q1
        end = end+1

    # print('q1', Q1, 'q3', Q3)
    # print('iqr', IQR)
    maxN = Q3 + 1.5 * IQR
    minN = Q1 - 1.5 * IQR
    # print(maxN, ' ========= ', minN)
    for i in range(rows):
        if data[i] > maxN:
            print(data[i],' maxN ',maxN)
            data[i] = maxN
        elif data[i] < minN:
            print(data[i], ' minN ', minN)
            data[i] = minN
    # with open('has_no_noise.csv','a')as f:
    #     f.write(','.join([str(x) for x in data]))
    #     f.write('\n')
    return data

# data = get_data(frame, datetime.datetime(2017, 11, 1, 2, 0, 0), 1)
# print(data)
