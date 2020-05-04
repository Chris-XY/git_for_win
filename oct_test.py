import torch
import pandas as pd
import math
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import collections
import networkx as nx
import utils, modules


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

    oct_normal_tensor = utils.external_dimension_creator(2018, 10, 31, "20181001", max_temps, min_temps, solar_exps,
                                                         rainfalls)

    external_dimension_for_day = oct_normal_tensor[504:]

    # Create Data for ST CNN part
    matrix_data_10 = utils.data_deal_cnn('data/result_oct.csv', 31, '-1')

    matrix_data_CNN = matrix_data_10

    # Create three channels
    # different hours
    timeline_data_C_ST = torch.zeros((31 * 24 - 504), 3, 2, 100, 100)
    # different day same time
    timeline_data_P_ST = torch.zeros((31 * 24 - 504), 3, 2, 100, 100)
    # different week same time
    timeline_data_T_ST = torch.zeros((31 * 24 - 504), 3, 2, 100, 100)
    # result_ST = torch.zeros((365 * 24 - 504), 2, 100, 100)

    i = 0
    for T in range(504, 31 * 24):
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
    #
    # board = math.ceil((365 * 24 - 504) * 0.8)
    #
    # train_X_C_ST, test_X_C_ST = timeline_data_C_ST[0:board], timeline_data_C_ST[board:-1]
    # train_X_P_ST, test_X_P_ST = timeline_data_P_ST[0:board], timeline_data_P_ST[board:-1]
    # train_X_T_ST, test_X_T_ST = timeline_data_T_ST[0:board], timeline_data_T_ST[board:-1]

    # train_X_Ext_ST, test_X_Ext_ST = external_dimension_for_day[0:board], external_dimension_for_day[board:-1]
    print('build a mask')
    # build a mask
    data_oct = pd.read_csv('data/result_oct.csv')

    intersection_matrix_in = np.zeros((100, 100))
    intersection_matrix_out = np.zeros((100, 100))

    plot_data_lists = [data_oct]

    for plot_data in plot_data_lists:
        for i in range(len(plot_data)):
            intersection_matrix_in[plot_data['bslat_new'][i] - 1][plot_data['bslon_new'][i] - 1] += 1
            intersection_matrix_out[plot_data['aslat_new'][i] - 1][plot_data['aslon_new'][i] - 1] += 1

    x_in = intersection_matrix_in
    mask_in = x_in < 1

    x_out = intersection_matrix_out
    mask_out = x_out < 1


    print('GNN part start')
    # Create Data for GNN part
    # create a graph
    G = nx.Graph()

    # use inflow and outflow for each edge's weight
    edge_flow = collections.defaultdict(int)
    for i in range(len(data_oct)):
        temp_edge = (data_oct['BSID'][i], data_oct['ASID'][i])
        if temp_edge not in edge_flow:
            edge_flow[temp_edge] += 1

    for key, value in edge_flow.items():
        G.add_edge(key[0], key[1], weight=value)

    print(2)

    result_flow_pg = nx.pagerank(G, weight='weight')
    result_flow_pg = {k: v for k, v in sorted(result_flow_pg.items(), key=lambda item: item[1], reverse=True)}

    stop_distribution_pg = {}
    for key, _ in result_flow_pg.items():
        stop_distribution_pg[key] = []
        for stop in G[key]:
            stop_distribution_pg[key].append(stop)

    for key in stop_distribution_pg.keys():
        stopDict = {}
        for stop in stop_distribution_pg[key]:
            if stop != key:
                val = result_flow_pg[stop]
                stopDict[stop] = val
        stop_distribution_pg[key] = stopDict

    for key in stop_distribution_pg.keys():
        stop_distribution_pg[key] = {k: v for k, v in
                                     sorted(stop_distribution_pg[key].items(), key=lambda item: item[1], reverse=True)}

    max_len = 0
    for key, item in stop_distribution_pg.items():
        if len(item) > max_len:
            max_len = len(item)

    stop_distribution_BSID = [[0 for _ in range(max_len + 1)] for i in range(len(stop_distribution_pg))]
    stop_distribution_ASID = [[0 for _ in range(max_len + 1)] for i in range(len(stop_distribution_pg))]

    print(3)

    i = 0
    for key in stop_distribution_pg.keys():
        stop_distribution_BSID[i][0] = key
        stop_distribution_ASID[i][0] = key
        i += 1

    for i in range(len(stop_distribution_BSID)):
        j = 1
        for key in stop_distribution_pg[stop_distribution_BSID[i][0]].keys():
            stop_distribution_BSID[i][j] = key
            if j < 21:
                j += 1
            else:
                continue

    for i in range(len(stop_distribution_ASID)):
        j = 1
        for key in stop_distribution_pg[stop_distribution_ASID[i][0]].keys():
            stop_distribution_ASID[i][j] = key
            if j < 21:
                j += 1
            else:
                continue

    # save all stops list
    with open('data/BSID_top_list_october.txt', 'wt') as f:
        for stoplist in stop_distribution_BSID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    with open('data/ASID_top_list_october.txt', 'wt') as f:
        for stoplist in stop_distribution_ASID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    stop_distribution_BSID = [stop[:11] for stop in stop_distribution_BSID]
    stop_distribution_ASID = [stop[:11] for stop in stop_distribution_ASID]

    # save top 16 lists
    with open('data/BSID_top_list_10_october.txt', 'wt') as f:
        for stoplist in stop_distribution_BSID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    with open('data/ASID_top_list_10_october.txt', 'wt') as f:
        for stoplist in stop_distribution_ASID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()
    print(stop_distribution_ASID)

    stop_distribution = [stop[0] for stop in stop_distribution_ASID]

    cols = 10 + 1  # 10 + 1
    rows = len(stop_distribution)

    matrix_data_10, ground_truth_10 = utils.data_deal_gnn('data/result_oct.csv', 31,
                                                          stop_distribution,
                                                          stop_distribution_BSID,
                                                          stop_distribution_ASID, '-1'
                                                          )

    # Create three channels
    # different hours
    timeline_data_C_GNN = torch.zeros((31 * 24 - 504), 3, 2, rows, cols - 1)
    # different day same time
    timeline_data_P_GNN = torch.zeros((31 * 24 - 504), 3, 2, rows, cols - 1)
    # different week same time
    timeline_data_T_GNN = torch.zeros((31 * 24 - 504), 3, 2, rows, cols - 1)
    result_GNN = torch.zeros((31 * 24 - 504), 2, rows)
    i = 0

    matrix_data_GNN = matrix_data_10

    ground_truth_GNN = ground_truth_10

    for T in range(504, 31*24):
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

    external_dimension_for_day_GNN = external_dimension_for_day



    stop_number_location = utils.stop_location_reader('stops_locations.txt')

    translinkDB = modules.TranslinkDataset(timeline_data_C_GNN, timeline_data_P_GNN, timeline_data_T_GNN,
                                           external_dimension_for_day_GNN, timeline_data_C_ST, timeline_data_P_ST,
                                           timeline_data_T_ST, result_GNN)

    train_size = 200
    test_size = len(translinkDB) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(translinkDB, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    gCNN = modules.GCModel(mask_in, mask_out, stop_distribution, stop_number_location)
    optimizer_GNN = torch.optim.Adam(gCNN.parameters(), lr=0.001)
    criterion_GNN = nn.MSELoss()

    loss_list_GNN = []

    j = 0
    for batch in train_loader:
        for i in range(20):
            data = [batch[0][i], batch[1][i], batch[2][i], batch[3][i],
                         batch[4][i], batch[5][i], batch[6][i]]
            output = gCNN(data)
            output_2, min_in, max_in, min_out, max_out = utils.min_max_scalar_for_torch_min_max_gnn(batch[7][i])

            loss = torch.sqrt(criterion_GNN(output, output_2))
            print(loss)
            loss_list_GNN.append(loss)

            loss.backward()
            optimizer_GNN.step()
    optimizer_GNN.zero_grad()

    sum_loss = torch.zeros(1)

    for batch in test_loader:
        for i in range(8):
            data = [batch[0][i], batch[1][i], batch[2][i], batch[3][i],
                    batch[4][i], batch[5][i], batch[6][i]]

            output = gCNN(data)
            _, min_in, max_in, min_out, max_out = utils.min_max_scalar_for_torch_min_max_gnn(batch[7][i])
            loss = torch.sqrt(criterion_GNN(output[0][0] * (max_in - min_in) + min_in, batch[7][i][0]))
            sum_loss += loss
            loss = torch.sqrt(criterion_GNN(output[0][1] * (max_out - min_out) + min_out, batch[7][i][1]))
            sum_loss += loss

            print(sum_loss)

    with open('result.txt', 'wt') as f:
        f.write(str(sum_loss))
    f.close()