import torch
import numpy as np
import math
from datetime import datetime
import pandas as pd
from torch import Tensor


def stop_distribution_reader(path):
    file = open(path, 'r')

    stop_distribution = file.readlines()

    file.close()

    for i in range(len(stop_distribution)):
        stop_distribution[i] = stop_distribution[i][:-2]

    for i in range(len(stop_distribution)):
        top_stops = stop_distribution[i].split(' ')
        stop_distribution[i] = [int(i) for i in top_stops]

    return stop_distribution


def stop_location_reader(path):
    # stop_locations read from file
    # 'stops_locations.txt'
    file = open(path, 'r')
    locations = file.readlines()
    file.close()

    locations = [loc[:-1] for loc in locations]
    locations = [loc.split(' ', 1) for loc in locations]

    stop_distribution = {}
    for i in range(len(locations)):
        loc_info = locations[i][1][1:-1]
        lat, lon, _ = loc_info.split(', ')
        stop_distribution[int(locations[i][0])] = [int(lat), int(lon)]

    return stop_distribution


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


def min_max_scalar(data):
    min_v = np.amin(data)
    max_v = np.amax(data)
    range_v = max_v - min_v
    if range_v > 0:
        normalised = (data - min_v) / range_v
    else:
        normalised = np.zeros(data.size())
    return normalised


def external_dimension_creator(year, month, day_per_month, strptimestring, max_temps, min_temps, solar_exposure,
                               rainfalls):
    max_dec = [0 for i in range(day_per_month)]
    min_dec = [0 for i in range(day_per_month)]
    solar_dec = [0 for i in range(day_per_month)]
    rf_dec = [0 for i in range(day_per_month)]

    cnt = 0
    for i in range(len(max_temps)):
        if max_temps['Year'][i] == year and max_temps['Month'][i] == month:
            max_dec[cnt] = max_temps['Maximum temperature (Degree C)'][i]
            cnt += 1

    cnt = 0
    for i in range(len(min_temps)):
        if min_temps['Year'][i] == year and min_temps['Month'][i] == month:
            min_dec[cnt] = min_temps['Minimum temperature (Degree C)'][i]
            cnt += 1

    cnt = 0
    for i in range(len(solar_exposure)):
        if solar_exposure['Year'][i] == year and solar_exposure['Month'][i] == month:
            solar_dec[cnt] = solar_exposure['Daily global solar exposure (MJ/m*m)'][i]
            cnt += 1

    cnt = 0
    for i in range(len(rainfalls)):
        if rainfalls['Year'][i] == year and rainfalls['Month'][i] == month:
            rf_dec[cnt] = rainfalls['Rainfall amount (millimetres)'][i]
            cnt += 1

    # check if the data is nan or not
    for i in range(day_per_month):
        if math.isnan(max_dec[i]):
            max_dec[i] = sum(max_dec[:i]) * 1.0 / len(max_dec[:i])

    for i in range(day_per_month):
        if math.isnan(solar_dec[i]):
            solar_dec[i] = sum(solar_dec[:i]) * 1.0 / len(solar_dec[:i])

    for i in range(day_per_month):
        if math.isnan(min_dec[i]):
            min_dec[i] = sum(min_dec[:i]) * 1.0 / len(min_dec[:i])

    for i in range(day_per_month):
        if math.isnan(rf_dec[i]):
            rf_dec[i] = sum(rf_dec[:i]) * 1.0 / len(rf_dec[:i])

    max_dec_to_hour = [0 for i in range(day_per_month * 24)]
    day = 0
    for i in range(day_per_month * 24):
        if i == 0:
            max_dec_to_hour[i] = max_dec[day]
        elif i % 24 == 0:
            day += 1
            max_dec_to_hour[i] = max_dec[day]
        elif i > 0:
            max_dec_to_hour[i] = max_dec[day]

    min_dec_to_hour = [0 for i in range(day_per_month * 24)]
    day = 0
    for i in range(day_per_month * 24):
        if i == 0:
            min_dec_to_hour[i] = min_dec[day]
        elif i % 24 == 0:
            day += 1
            min_dec_to_hour[i] = min_dec[day]
        elif i > 0:
            min_dec_to_hour[i] = min_dec[day]

    solar_dec_to_hour = [0 for i in range(day_per_month * 24)]
    day = 0
    for i in range(day_per_month * 24):
        if i == 0:
            solar_dec_to_hour[i] = solar_dec[day]
        elif i % 24 == 0:
            day += 1
            solar_dec_to_hour[i] = solar_dec[day]
        elif i > 0:
            solar_dec_to_hour[i] = solar_dec[day]

    rf_dec_to_hour = [0 for i in range(day_per_month * 24)]
    day = 0
    for i in range(day_per_month * 24):
        if i == 0:
            rf_dec_to_hour[i] = rf_dec[day]
        elif i % 24 == 0:
            day += 1
            rf_dec_to_hour[i] = rf_dec[day]
        elif i > 0:
            rf_dec_to_hour[i] = rf_dec[day]

    max_transfer = min_max_scalar(max_dec_to_hour)
    min_transfer = min_max_scalar(min_dec_to_hour)
    solar_transfer = min_max_scalar(solar_dec_to_hour)
    rain_transfer = min_max_scalar(rf_dec_to_hour)
    external_dimension = [max_transfer, min_transfer, solar_transfer, rain_transfer]

    external_dimension_tensor = torch.FloatTensor(external_dimension)
    external_dimension_tensor = external_dimension_tensor.t()

    # create one hot for 7 days
    external_day_of_week = [-1 for i in range(day_per_month * 24)]
    start_day = datetime.strptime(strptimestring, "%Y%m%d").weekday() + 1

    for i in range(day_per_month * 24):
        if i == 0:
            external_day_of_week[i] = start_day % 7
        elif i % 24 == 0:
            start_day += 1
            start_day = start_day % 7
            external_day_of_week[i] = start_day
        elif i > 0:
            external_day_of_week[i] = start_day % 7

    class_count = 7
    one_hot_for_day = one_hot(external_day_of_week, class_count)

    normal_external_dimension_tensor = torch.cat([external_dimension_tensor, one_hot_for_day], dim=1)

    return normal_external_dimension_tensor


def data_deal_cnn(path, day, mode):
    data = pd.read_csv(path)
    matrix_data = torch.zeros(day * 24, 2, 100, 100)

    print(path + str(len(data)))
    print('Start.')
    for i in range(len(data)):
        if i % 50000 == 0:
            print(i)
        boarding_time = data['BT'][i]
        alighting_time = data['AT'][i]
        b_day = -1
        b_hour = -1
        a_day = -1
        a_hour = -1
        b_date, b_time = boarding_time.split()
        if mode == '-1':
            b_day = int(b_date.split('/')[-1])
        elif mode == '0':
            b_day = int(b_date.split('/')[0])
        b_hour = int(b_time.split(':')[0])
        a_date, a_time = alighting_time.split()
        if mode == '-1':
            a_day = int(a_date.split('/')[-1])
        elif mode == '0':
            a_day = int(a_date.split('/')[0])
        a_hour = int(a_time.split(':')[0])
        matrix_data[(b_day - 1) * 24 + b_hour][0][int(data['bslat_new'][i]) - 1][int(data['bslon_new'][i]) - 1] += 1
        matrix_data[(a_day - 1) * 24 + a_hour][1][int(data['aslat_new'][i]) - 1][int(data['aslon_new'][i]) - 1] += 1

    print(path + ' finished')
    return matrix_data


def data_deal_gnn(path, day, stop_distribution, stop_distribution_boarding, stop_distribution_alighting, mode):
    data = pd.read_csv(path)
    cols = 10 + 1  # 16 + 1
    rows = len(stop_distribution)

    matrix_data = torch.zeros(day * 24, 2, rows, cols - 1)
    ground_truth = torch.zeros(day * 24, 2, rows)

    print(path + str(len(data)))
    print('Start.')
    for i in range(len(data)):
        if i % 50000 == 0:
            print(i)
        boarding_time = data['BT'][i]
        alighting_time = data['AT'][i]
        b_day = -1
        b_hour = -1
        a_day = -1
        a_hour = -1
        b_date, b_time = boarding_time.split()
        if mode == '-1':
            b_day = int(b_date.split('/')[-1])
        elif mode == '0':
            b_day = int(b_date.split('/')[0])
        b_hour = int(b_time.split(':')[0])
        a_date, a_time = alighting_time.split()
        if mode == '-1':
            a_day = int(a_date.split('/')[-1])
        elif mode == '0':
            a_day = int(a_date.split('/')[0])
        a_hour = int(a_time.split(':')[0])

        BSID = stop_distribution.index(data['BSID'][i])
        if data['ASID'][i] in stop_distribution_boarding[BSID]:
            ASID = stop_distribution_boarding[BSID].index(data['ASID'][i])
            matrix_data[(b_day - 1) * 24 + b_hour][0][BSID][ASID - 1] += 1
        ground_truth[(b_day - 1) * 24 + b_hour][0][BSID] += 1

        ASID = stop_distribution.index(data['ASID'][i])
        if data['BSID'][i] in stop_distribution_alighting[ASID]:
            BSID = stop_distribution_alighting[ASID].index(data['BSID'][i])
            matrix_data[(a_day - 1) * 24 + a_hour][1][ASID][BSID - 1] += 1

        ground_truth[(a_day - 1) * 24 + a_hour][1][ASID] += 1
    print(path + ' finished')

    return matrix_data, ground_truth


def min_max_scalar_for_torch_min_max_gnn(data: Tensor):
    stop_nums = len(data[1])
    normalised = torch.zeros(2, stop_nums)
    min_in = torch.min(data[0])
    max_in = torch.max(data[0])
    range_in = max_in - min_in
    if range_in > 0:
        normalised_in = (data[0] - min_in) / range_in
    else:
        normalised_in = torch.zeros(stop_nums)

    min_out = torch.min(data[1])
    max_out = torch.max(data[1])
    range_out = max_out - min_out
    if range_out > 0:
        normalised_out = (data[1] - min_out) / range_out
    else:
        normalised_out = torch.zeros(stop_nums)

    normalised[0] = normalised_in
    normalised[1] = normalised_out

    return normalised, min_in, max_in, min_out, max_out


# build for train function
def min_max_scalar_for_torch_gnn(data):
    stop_nums = len(data[1])
    normalised: Tensor = torch.zeros(2, stop_nums)
    min_in = torch.min(data[0])
    max_in = torch.max(data[0])
    range_in = max_in - min_in
    if range_in > 0:
        normalised_in = (data[0] - min_in) / range_in
    else:
        normalised_in = torch.zeros(stop_nums)

    min_out = torch.min(data[1])
    max_out = torch.max(data[1])
    range_out = max_out - min_out
    if range_out > 0:
        normalised_out = (data[1] - min_out) / range_out
    else:
        normalised_out = torch.zeros(stop_nums)

    normalised[0] = normalised_in
    normalised[1] = normalised_out

    return normalised
