import collections

import pandas as pd
import numpy as np
import networkx as nx

if __name__ == '__main__':
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

    print(1)

    # Create Data for GNN part
    data_one_year = [data_jul, data_aug, data_sep, data_oct, data_nov, data_dec, data_jan,
                     data_feb, data_mar, data_apr, data_may, data_jun]
    # create a graph
    G = nx.Graph()

    # use inflow and outflow for each edge's weight
    edge_flow = collections.defaultdict(int)
    for data_month in data_one_year:
        for i in range(len(data_month)):
            temp_edge = (data_month['BSID'][i], data_month['ASID'][i])
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
    with open('BSID_top_list.txt', 'wt') as f:
        for stoplist in stop_distribution_BSID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    with open('ASID_top_list.txt', 'wt') as f:
        for stoplist in stop_distribution_ASID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    stop_distribution_BSID = [stop[:11] for stop in stop_distribution_BSID]
    stop_distribution_ASID = [stop[:11] for stop in stop_distribution_ASID]

    # save top 16 lists
    with open('BSID_top_list_10.txt', 'wt') as f:
        for stoplist in stop_distribution_BSID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()

    with open('ASID_top_list_10.txt', 'wt') as f:
        for stoplist in stop_distribution_ASID:
            for i in stoplist:
                line = str(i) + ' '
                f.write(line)
            f.write('\n')
    f.close()
    print(stop_distribution_ASID)

    stop_distribution = [stop[0] for stop in stop_distribution_ASID]
    print(4)

    # stop_lat, stop_lon, data_missing: -1 / data_found: 1
    stop_number_location = dict.fromkeys(stop_distribution, [-1, -1, -1])

    stop_info_twelve_months = [data_jul, data_aug, data_sep, data_oct, data_nov, data_dec, data_jan, data_feb, data_mar, data_apr,
                 data_may, data_jun]
    for stop, stop_loc in stop_number_location.items():
        print(stop)
        i = 7
        for stop_info in stop_info_twelve_months:
            i += 1
            print(i % 12)
            for i in range(len(stop_info)):
                if stop == stop_info['BSID'][i]:
                    stop_number_location[stop] = [stop_info['bslat_new'][i], stop_info['bslon_new'][i], 1]
                    break
                if stop == stop_info['ASID'][i]:
                    stop_number_location[stop] = [stop_info['aslat_new'][i], stop_info['aslon_new'][i], 1]
                    change = 'Find Location'
                    break

            if stop_number_location[stop][-1] == 1:
                break
        print(stop_number_location[stop])

    # save stops and it's corresponding location
    with open('stops_locations.txt', 'wt') as f:
        for stop, location in stop_number_location.items():
            line = str(stop) + ' ' + str(location)
            f.write(line)
            f.write('\n')
    f.close()