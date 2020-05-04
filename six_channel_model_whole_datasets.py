import torch
import pandas as pd
import math
import numpy as np
import torch.nn as nn
import modules
import utils

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    # External Dimension
    max_temps = pd.read_csv('data/maximum_temp.csv')
    min_temps = pd.read_csv('data/minimum_temp.csv')
    solar_exps = pd.read_csv('data/solar_exposure.csv')
    rainfalls = pd.read_csv('data/rainfall.csv')

    jul_normal_tensor = utils.external_dimension_creator(2018, 7, 31, "20180701", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    aug_normal_tensor = utils.external_dimension_creator(2018, 8, 31, "20180801", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    sep_normal_tensor = utils.external_dimension_creator(2018, 9, 30, "20180901", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    oct_normal_tensor = utils.external_dimension_creator(2018, 10, 31, "20181001", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    nov_normal_tensor = utils.external_dimension_creator(2018, 11, 30, "20181101", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    dec_normal_tensor = utils.external_dimension_creator(2018, 12, 31, "20181201", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    jan_normal_tensor = utils.external_dimension_creator(2019, 1, 31, "20190101", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    feb_normal_tensor = utils.external_dimension_creator(2019, 2, 28, "20190201", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    mar_normal_tensor = utils.external_dimension_creator(2019, 3, 31, "20190301", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    apr_normal_tensor = utils.external_dimension_creator(2019, 4, 30, "20190401", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    may_normal_tensor = utils.external_dimension_creator(2019, 5, 31, "20190501", max_temps, min_temps, solar_exps,
                                                         rainfalls)
    jun_normal_tensor = utils.external_dimension_creator(2019, 6, 30, "20190601", max_temps, min_temps, solar_exps,
                                                         rainfalls)

    external_dimension_for_day = torch.cat((jul_normal_tensor[504:], aug_normal_tensor, sep_normal_tensor,
                                            oct_normal_tensor, nov_normal_tensor, dec_normal_tensor, jan_normal_tensor,
                                            feb_normal_tensor, mar_normal_tensor, apr_normal_tensor, may_normal_tensor,
                                            jun_normal_tensor), 0)

    # Create Data for ST CNN part
    matrix_data_7 = utils.data_deal_CNN('data/result_jul.csv', 31, '-1')
    matrix_data_8 = utils.data_deal_CNN('data/result_aug.csv', 31, '-1')
    matrix_data_9 = utils.data_deal_CNN('data/result_sep.csv', 30, '-1')
    matrix_data_10 = utils.data_deal_CNN('data/result_oct.csv', 31, '-1')
    matrix_data_11 = utils.data_deal_CNN('data/result_nov.csv', 30, '-1')
    matrix_data_12 = utils.data_deal_CNN('data/result_dec.csv', 31, '0')

    matrix_data_1 = utils.data_deal_CNN('data/result_jan.csv', 31, '-1')
    matrix_data_2 = utils.data_deal_CNN('data/result_feb.csv', 28, '-1')
    matrix_data_3 = utils.data_deal_CNN('data/result_mar.csv', 31, '-1')
    matrix_data_4 = utils.data_deal_CNN('data/result_apr.csv', 30, '-1')
    matrix_data_5 = utils.data_deal_CNN('data/result_may.csv', 31, '-1')
    matrix_data_6 = utils.data_deal_CNN('data/result_jun.csv', 30, '-1')

    matrix_data_CNN = torch.cat((matrix_data_7, matrix_data_8, matrix_data_9, matrix_data_10, matrix_data_11,
                                 matrix_data_12, matrix_data_1, matrix_data_2, matrix_data_3, matrix_data_4,
                                 matrix_data_5, matrix_data_6), 0)

    # Create three channels
    # different hours
    timeline_data_C_ST = torch.zeros((365 * 24 - 504), 3, 2, 100, 100)
    # different day same time
    timeline_data_P_ST = torch.zeros((365 * 24 - 504), 3, 2, 100, 100)
    # different week same time
    timeline_data_T_ST = torch.zeros((365 * 24 - 504), 3, 2, 100, 100)
    # result_ST = torch.zeros((365 * 24 - 504), 2, 100, 100)

    i = 0
    for T in range(504, 365 * 24):
        if T % 10000 == 0:
            print(T)
        timeline_data_C_ST[i][0][0] = matrix_data_CNN[T - 1][0]
        timeline_data_C_ST[i][1][0] = matrix_data_CNN[T - 2][0]
        timeline_data_C_ST[i][2][0] = matrix_data_CNN[T - 3][0]
        timeline_data_C_ST[i][0][1] = matrix_data_CNN[T - 1][1]
        timeline_data_C_ST[i][1][1] = matrix_data_CNN[T - 2][1]
        timeline_data_C_ST[i][2][1] = matrix_data_CNN[T - 3][1]
        timeline_data_P_ST[i][0][0] = matrix_data_CNN[T - 24][0]
        timeline_data_P_ST[i][1][0] = matrix_data_CNN[T - 24 - 24][0]
        timeline_data_P_ST[i][2][0] = matrix_data_CNN[T - 24 - 24 - 24][0]
        timeline_data_P_ST[i][0][1] = matrix_data_CNN[T - 24][1]
        timeline_data_P_ST[i][1][1] = matrix_data_CNN[T - 24 - 24][1]
        timeline_data_P_ST[i][2][1] = matrix_data_CNN[T - 24 - 24 - 24][1]
        timeline_data_T_ST[i][0][0] = matrix_data_CNN[T - 7 * 24 * 1][0]
        timeline_data_T_ST[i][1][0] = matrix_data_CNN[T - 7 * 24 * 2][0]
        timeline_data_T_ST[i][2][0] = matrix_data_CNN[T - 7 * 24 * 3][0]
        timeline_data_T_ST[i][0][1] = matrix_data_CNN[T - 7 * 24 * 1][1]
        timeline_data_T_ST[i][1][1] = matrix_data_CNN[T - 7 * 24 * 2][1]
        timeline_data_T_ST[i][2][1] = matrix_data_CNN[T - 7 * 24 * 3][1]
        i += 1

    board = math.ceil((365 * 24 - 504) * 0.8)

    train_X_C_ST, test_X_C_ST = timeline_data_C_ST[0:board], timeline_data_C_ST[board:-1]
    train_X_P_ST, test_X_P_ST = timeline_data_P_ST[0:board], timeline_data_P_ST[board:-1]
    train_X_T_ST, test_X_T_ST = timeline_data_T_ST[0:board], timeline_data_T_ST[board:-1]

    train_X_Ext_ST, test_X_Ext_ST = external_dimension_for_day[0:board], external_dimension_for_day[board:-1]

    # build a mask
    data_jul = pd.read_csv('data/result_jul.csv')
    data_aug = pd.read_csv('data/result_aug.csv')
    data_sep = pd.read_csv('data/result_sep.csv')
    data_oct = pd.read_csv('data/result_oct.csv')
    data_nov = pd.read_csv('data/result_nov.csv')
    data_dec = pd.read_csv('data/result_dec.csv')
    data_jan = pd.read_csv('data/result_jan.csv')
    data_feb = pd.read_csv('data/result_feb.csv')
    data_mar = pd.read_csv('data/result_mar.csv')
    data_apr = pd.read_csv('data/result_apr.csv')
    data_may = pd.read_csv('data/result_may.csv')
    data_jun = pd.read_csv('data/result_jun.csv')

    intersection_matrix_in = np.zeros((100, 100))
    intersection_matrix_out = np.zeros((100, 100))

    plot_data_lists = [data_jul, data_aug, data_sep, data_oct, data_nov,
                       data_dec, data_jan, data_feb, data_mar, data_apr,
                       data_may, data_jun]

    for plot_data in plot_data_lists:
        for i in range(len(plot_data)):
            intersection_matrix_in[plot_data['bslat_new'][i] - 1][plot_data['bslon_new'][i] - 1] += 1
            intersection_matrix_out[plot_data['aslat_new'][i] - 1][plot_data['aslon_new'][i] - 1] += 1

    x_in = intersection_matrix_in
    mask_in = x_in < 1

    x_out = intersection_matrix_out
    mask_out = x_out < 1

    # Create Data for GNN part
    stop_distribution_BSID = utils.stop_distribution_reader('BSID_top_list_10.txt')
    stop_distribution_ASID = utils.stop_distribution_reader('ASID_top_list_10.txt')

    stop_distribution = [stop[0] for stop in stop_distribution_ASID]

    cols = 10 + 1  # 10 + 1
    rows = len(stop_distribution)

    matrix_data_7, ground_truth_7 = utils.data_deal_GNN('data/result_jul.csv', 31,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_8, ground_truth_8 = utils.data_deal_GNN('data/result_aug.csv', 31,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_9, ground_truth_9 = utils.data_deal_GNN('data/result_sep.csv', 30,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_10, ground_truth_10 = utils.data_deal_GNN('data/result_oct.csv', 31,
                                                          stop_distribution,
                                                          stop_distribution_BSID,
                                                          stop_distribution_ASID, '-1'
                                                          )
    matrix_data_11, ground_truth_11 = utils.data_deal_GNN('data/result_nov.csv', 30,
                                                          stop_distribution,
                                                          stop_distribution_BSID,
                                                          stop_distribution_ASID, '-1'
                                                          )
    matrix_data_12, ground_truth_12 = utils.data_deal_GNN('data/result_dec.csv', 31,
                                                          stop_distribution,
                                                          stop_distribution_BSID,
                                                          stop_distribution_ASID, '0'
                                                          )
    matrix_data_1, ground_truth_1 = utils.data_deal_GNN('data/result_jan.csv', 31,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_2, ground_truth_2 = utils.data_deal_GNN('data/result_feb.csv', 28,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_3, ground_truth_3 = utils.data_deal_GNN('data/result_mar.csv', 31,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_4, ground_truth_4 = utils.data_deal_GNN('data/result_apr.csv', 30,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_5, ground_truth_5 = utils.data_deal_GNN('data/result_may.csv', 31,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )
    matrix_data_6, ground_truth_6 = utils.data_deal_GNN('data/result_jun.csv', 30,
                                                        stop_distribution, stop_distribution_BSID,
                                                        stop_distribution_ASID, '-1'
                                                        )

    # Create three channels
    # different hours
    timeline_data_C_GNN = torch.zeros((365 * 24 - 504), 3, 2, rows, cols - 1)
    # different day same time
    timeline_data_P_GNN = torch.zeros((365 * 24 - 504), 3, 2, rows, cols - 1)
    # different week same time
    timeline_data_T_GNN = torch.zeros((365 * 24 - 504), 3, 2, rows, cols - 1)
    result_GNN = torch.zeros((365 * 24 - 504), 2, rows)
    i = 0

    matrix_data_GNN = torch.cat((matrix_data_7, matrix_data_8, matrix_data_9, matrix_data_10, matrix_data_11,
                                 matrix_data_12, matrix_data_1, matrix_data_2, matrix_data_3, matrix_data_4,
                                 matrix_data_5, matrix_data_6), 0)

    ground_truth_GNN = torch.cat((ground_truth_7, ground_truth_8, ground_truth_9, ground_truth_10,
                                  ground_truth_11, ground_truth_12, ground_truth_1, ground_truth_2,
                                  ground_truth_3, ground_truth_4, ground_truth_5, ground_truth_6))

    for T in range(504, 365*24):
        if T % 10000 == 0:
            print(T)
        timeline_data_C_GNN[i][0][0] = matrix_data_GNN[T - 1][0]
        timeline_data_C_GNN[i][1][0] = matrix_data_GNN[T - 2][0]
        timeline_data_C_GNN[i][2][0] = matrix_data_GNN[T - 3][0]
        timeline_data_C_GNN[i][0][1] = matrix_data_GNN[T - 1][1]
        timeline_data_C_GNN[i][1][1] = matrix_data_GNN[T - 2][1]
        timeline_data_C_GNN[i][2][1] = matrix_data_GNN[T - 3][1]
        timeline_data_P_GNN[i][0][0] = matrix_data_GNN[T - 24][0]
        timeline_data_P_GNN[i][1][0] = matrix_data_GNN[T - 24 - 24][0]
        timeline_data_P_GNN[i][2][0] = matrix_data_GNN[T - 24 - 24 - 24][0]
        timeline_data_P_GNN[i][0][1] = matrix_data_GNN[T - 24][1]
        timeline_data_P_GNN[i][1][1] = matrix_data_GNN[T - 24 - 24][1]
        timeline_data_P_GNN[i][2][1] = matrix_data_GNN[T - 24 - 24 - 24][1]
        timeline_data_T_GNN[i][0][0] = matrix_data_GNN[T - 7 * 24][0]
        timeline_data_T_GNN[i][1][0] = matrix_data_GNN[T - 7 * 24 * 2][0]
        timeline_data_T_GNN[i][2][0] = matrix_data_GNN[T - 7 * 24 * 3][0]
        timeline_data_T_GNN[i][0][1] = matrix_data_GNN[T - 7 * 24][1]
        timeline_data_T_GNN[i][1][1] = matrix_data_GNN[T - 7 * 24 * 2][1]
        timeline_data_T_GNN[i][2][1] = matrix_data_GNN[T - 7 * 24 * 3][1]
        result_GNN[i][0] = ground_truth_GNN[T][0]
        result_GNN[i][1] = ground_truth_GNN[T][1]
        i += 1

    board = math.ceil((365 * 24 - 504) * 0.8)

    external_dimension_for_day_GNN = external_dimension_for_day

    train_X_C_GNN, test_X_C_GNN = timeline_data_C_GNN[0:board], timeline_data_C_GNN[board:-1]
    train_X_P_GNN, test_X_P_GNN = timeline_data_P_GNN[0:board], timeline_data_P_GNN[board:-1]
    train_X_T_GNN, test_X_T_GNN = timeline_data_T_GNN[0:board], timeline_data_T_GNN[board:-1]

    train_X_Ext_GNN, test_X_Ext_GNN = external_dimension_for_day_GNN[0:board], external_dimension_for_day_GNN[board:-1]

    train_Y_GNN, test_Y_GNN = result_GNN[0:board], result_GNN[board:-1]

    stop_number_location = utils.stop_location_reader('stops_locations.txt')

    # ================================================================================
    gCNN = modules.GCModel()
    optimizer_GNN = torch.optim.Adam(gCNN.parameters(), lr=0.001)
    criterion_GNN = nn.MSELoss(size_average=True)

    loss_list_GNN = []

    for epoch in range(2):
        for i in range(len(train_X_C_GNN)):
            output = gCNN(train_X_C_GNN[i], train_X_P_GNN[i], train_X_T_GNN[i], train_X_Ext_GNN[i], train_X_C_ST[i],
                          train_X_P_ST[i], train_X_T_ST[i], mask_in, mask_out, stop_distribution, stop_number_location)
            output_2, min_in, max_in, min_out, max_out = utils.minmaxscalar_for_torch_min_max_GNN(train_Y_GNN[i])

            loss = torch.sqrt(criterion_GNN(output, output_2))
            print(loss)
            loss_list_GNN.append(loss)

            loss.backward()
            optimizer_GNN.step()
        optimizer_GNN.zero_grad()

    sum_loss = torch.zeros(1)

    for i in range(len(test_Y_GNN)):
        output = gCNN(test_X_C_GNN[i], test_X_P_GNN[i], test_X_T_GNN[i], test_X_Ext_GNN[i], test_X_C_ST[i],
                      test_X_P_ST[i], test_X_T_ST[i], mask_in, mask_out, stop_distribution, stop_number_location)
        _, min_in, max_in, min_out, max_out = utils.minmaxscalar_for_torch_min_max_GNN(test_Y_GNN[i])
        loss = torch.sqrt(criterion_GNN(output[0][0] * (max_in - min_in) + min_in, test_Y_GNN[i][0]))
        sum_loss += loss
        loss = torch.sqrt(criterion_GNN(output[0][1] * (max_out - min_out) + min_out, test_Y_GNN[i][1]))
        sum_loss += loss

        print(sum_loss)

    with open('result.txt', 'wt') as f:
        f.write(str(sum_loss))
    f.close()
