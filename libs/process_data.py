import numpy as np
import matplotlib.pyplot as plt
from shapely.wkt import loads
import geopandas as gpd
import os
from tqdm import trange
import json
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .utils import print_log
from scipy.spatial.distance import cosine


def processFlowData(data_dir, log):
    print_log('Processing od_matrix_list...', log=log)

    if os.path.exists(os.path.join(data_dir, "od_matrix_list.npy")) == False:

        nyc_zone_path = os.path.join(data_dir, "nyc-taxi-zones.csv")
        nyc_zone_data = pd.read_csv(nyc_zone_path)

        zone_num = nyc_zone_data['LocationID'].nunique()  # 260,nyc地区一共有260个地区
        zone_value_count = nyc_zone_data['LocationID'].value_counts()  # 原始数据中，56有两个，103有三个

        data = nyc_zone_data[['LocationID', 'zone', 'the_geom', 'borough']]
        manhattan_index = (data['borough'] == 'Manhattan')
        manhattan_data = data[manhattan_index]  # 因为有三个103，所以其实只有67个地区
        manhattan_data = manhattan_data.drop_duplicates(subset='LocationID', keep=False)  # 去除掉重复的103

        print_log('\tSaving manhattenZones...', log=log)
        np.save(os.path.join(data_dir, "manhattenZones.npy"), manhattan_data)

        STD_PREFS = np.array(manhattan_data['zone'])
        manhattan_data['pref_name'] = manhattan_data['zone']
        geom = manhattan_data['the_geom']

        geometryList = []
        for element in geom:
            element = loads(element)
            geometryList.append(element)

        manhattan_data['the_geom'] = manhattan_data['the_geom'].apply(loads)

        # 将DataFrame转换为GeoDataFrame
        manhattan_data_geo = gpd.GeoDataFrame(manhattan_data, geometry=geometryList)
        manhattan_data_geo.plot(figsize=(10, 10))
        plt.show()
        plt.close()

        manhattan_data_geo['adjacent_prefs'] = manhattan_data_geo.apply(
            lambda x: STD_PREFS[manhattan_data_geo.geometry.intersects(x['geometry'])], axis=1)
        manhattan_data_geo = manhattan_data_geo.sort_values('LocationID')
        manhattan_data_geo = manhattan_data_geo.reset_index(drop=True)
        manhattan_data_geo = manhattan_data_geo.reset_index()

        print_log('\tSaving IDandIndex_dict...', log=log)
        # 得到LocationID和index的对应关系字典，方便后续的OD_matrix的求取
        IDandIndex_dict = manhattan_data_geo.set_index('LocationID')['index'].to_dict()
        with open(os.path.join(data_dir, "IDandIndex_dict.json"), 'w') as file:
            json.dump(IDandIndex_dict, file)

        adj_matrix = np.zeros((len(manhattan_data_geo), len(manhattan_data_geo)))
        for i in trange(len(manhattan_data_geo)):
            # 得到一个bool类型的Series，里面存储了第i个区域与其他区域是否邻接的bool值
            # 注意！！！此处我们没有处理两个区域是对角的情况不算是邻接矩阵的情况
            adj_pref_index = manhattan_data_geo.geometry.intersects(manhattan_data_geo.geometry.iloc[i])
            adj_matrix[i] = adj_pref_index

        print_log('\tSaving Manhattan-adj...', log=log)
        np.save(os.path.join(data_dir, "Manhattan-adj.npy"), adj_matrix)

        originDataPath = os.path.join(data_dir, "yellow_tripdata_2019-01.parquet")
        data_jan = pq.read_table(originDataPath).to_pandas()
        # (7508535,) 7696617

        manhattenZonesData = pd.DataFrame(manhattan_data, columns=['LocationID', 'zone', 'the_geom', 'borough'])
        manhattenZonesID = manhattenZonesData['LocationID']

        timestamp_jan_start = pd.Timestamp('2019-01-01')
        timestamp_jan_end = pd.Timestamp('2019-01-31 23:59:59')

        data_list = [data_jan]
        manhatten_data_jan_list = []
        # 我们只采用乘客上车的时间作为一个交通流的发生时间，且对原始数据进行清洗使得时间严格限制在1-6月的时间中。且以1.1的零点零分零秒作为起始时间，同时将时间转换为秒方便处理
        # 经过check_data可知，数据集中存在比地图中更多的区域，因此将涉及到多余地区编号的数据去除
        for element in data_list:
            element = element[(element['tpep_pickup_datetime'] >= timestamp_jan_start) & (
                    element['tpep_pickup_datetime'] <= timestamp_jan_end)]
            element = element[(element['PULocationID'] >= 1) & (element['PULocationID'] <= 263)]
            element = element[(element['DOLocationID'] >= 1) & (element['DOLocationID'] <= 263)]
            # 限定只采取manhattan区域的OD订单
            element = element[
                (element['PULocationID'].isin(manhattenZonesID) & element['DOLocationID'].isin(manhattenZonesID))]
            data = element[['tpep_pickup_datetime', 'PULocationID', 'DOLocationID']]
            manhatten_data_jan_list.append(data)

        manhatten_data_jan = pd.DataFrame()
        for element in manhatten_data_jan_list:
            manhatten_data_jan = pd.concat([manhatten_data_jan, element])

        manhatten_data_jan['tpep_pickup_datetime'] = (
                manhatten_data_jan['tpep_pickup_datetime'] - timestamp_jan_start).dt.total_seconds()

        # 检验在第一个5min中有多少个OD订单数：263
        test = manhatten_data_jan[manhatten_data_jan['tpep_pickup_datetime'] < 300]

        # manhatten_data_jan(6504432,3)，时间，起始ID，目的地ID
        manhatten_data_jan = manhatten_data_jan.sort_values('tpep_pickup_datetime')
        manhatten_data_jan = manhatten_data_jan.values

        print_log('\tSaving manhatten_data_jan...', log=log)
        np.save(os.path.join(data_dir, "manhatten_data_jan.npy"), manhatten_data_jan)

        od_matrix_list = []
        j = 0
        # 31天可以划分为8928个5分钟片段
        for i in tqdm(range(1, 8929)):
            od_matrix = np.zeros((len(manhattenZonesID), len(manhattenZonesID)))
            while manhatten_data_jan[j][0] < i * 300 and manhatten_data_jan[j][0] >= (i - 1) * 300:
                od_matrix[IDandIndex_dict[str(int(manhatten_data_jan[j][1]))]][
                    IDandIndex_dict[str(int(manhatten_data_jan[j][2]))]] += 1
                j += 1
                if j == len(manhatten_data_jan):
                    break
            od_matrix_list.append(od_matrix)

        print_log('\tSaving od_matrix_list...', log=log)
        np.save(os.path.join(data_dir, "od_matrix_list.npy"), od_matrix_list)

    else:
        print_log('Processing trafficFlow...', log=log)

        odPath = os.path.join(data_dir, "od_matrix_list.npy")
        od_matrxi_list = np.load(odPath)

        data_list = []
        for od_matrix in od_matrxi_list:
            row_sums = np.sum(od_matrix, axis=1)
            col_sums = np.sum(od_matrix, axis=0)
            trafic_flow = row_sums + col_sums
            # 与produce_raw_nycDataAndODmatrix.py中检验的test一直，第一个5min有263个OD订单，根据我们的转换交通流量的方法，一共有526交通流量，检验正确
            # test = np.sum(trafic_flow)
            data_list.append(trafic_flow)

        manhattanTrafficFlow = np.array(data_list)
        # (8928,66,1)
        manhattanTrafficFlow = torch.unsqueeze(torch.tensor(manhattanTrafficFlow), -1)

        print_log('\tSaving trafficFlow...', log=log)
        np.save(os.path.join(data_dir, "manhattanTrafficFlow.npy"), manhattanTrafficFlow)

    return manhattan_data, IDandIndex_dict, adj_matrix, manhatten_data_jan, od_matrxi_list, manhattanTrafficFlow


def processTime2VecData(data_dir, num_nodes, t2vDim, manhatten_data_jan, IDandIndex_dict, log):
    print_log('Processing time2vec...', log=log)

    i = 0
    j = 0
    # 31天可以划分为8928个5分钟片段
    newData_list = []
    time2vec = []
    seconds_in_day = 24 * 60 * 60
    seconds_in_week = 7 * 24 * 60 * 60

    for index in tqdm(range(1, 8929)):
        # 此循环以5min为间隔划分时间数据
        while manhatten_data_jan[j][0] < index * 300 and manhatten_data_jan[j][0] >= (index - 1) * 300:
            j += 1
            if j == len(manhatten_data_jan):
                break
        newData = manhatten_data_jan[i:j]
        i = j
        newData_list.append(newData)  # (list:8928){ndarry}

        zoneTime_list = []
        newData = pd.DataFrame(newData)
        # 此循环对5min内的某地区id的所有时间信息求和再计算time2vec
        for number in range(66):
            pick_is_zero = newData[1].astype(int).astype(str).map(IDandIndex_dict) == number
            # drop_is_zero = newData[2] == number
            # time_list = newData.loc[pick_is_zero | drop_is_zero ].iloc[:,0] #Series
            time_list = newData.loc[pick_is_zero].iloc[:, 0]  # Series

            time_list = pd.DataFrame(time_list)
            time_list = pd.DataFrame(time_list.sum())

            time_list.columns = ['seconds']
            time_list['day_sin_time'] = np.sin(2 * np.pi * time_list.seconds / seconds_in_day)
            time_list['day_cos_time'] = np.cos(2 * np.pi * time_list.seconds / seconds_in_day)
            time_list['week_sin_time'] = np.sin(2 * np.pi * time_list.seconds / seconds_in_week)
            time_list['week_cos_time'] = np.cos(2 * np.pi * time_list.seconds / seconds_in_week)

            zoneTime_list.append(time_list)  # list:66{(1,5)}

        time2vec.append(zoneTime_list)  # list:8928(list:263{(1,5)})
        # -----------------------------

    # (8928,66,1,5)
    time2vec = np.array(time2vec)
    # (8928,66,5)
    time2vec = time2vec.reshape(-1, num_nodes, t2vDim)

    print_log('\tSaving time2vec...', log=log)
    np.save(os.path.join(data_dir, "time2vec.npy"), time2vec)

    return time2vec


def processEventData(data_dir, od_matirx_list, event_flag, log):
    print_log('Processing event_list...', log=log)

    fir_data = od_matirx_list[:-1]
    sec_data = od_matirx_list[1:]
    event_list = []

    for i, j in zip(fir_data, sec_data):
        event = j - i
        event[(event < event_flag) & (event > -event_flag)] = 0
        event[event >= event_flag] = 1
        event[event <= -event_flag] = 2

        event_list.append(event)
    event_list = np.array(event_list)
    print_log('\tSaving event_list...', log=log)
    np.save(os.path.join(data_dir, "event_list.npy"), event_list)

    return event_list


def processDistanceMatrix(data_dir, phy_adj, log):
    print_log('Processing distanceMatrix...', log=log)

    nyc_zone_data = pd.read_csv(os.path.join(data_dir, "nyc-taxi-zones.csv"))
    zone_num = nyc_zone_data['LocationID'].nunique()  # 260
    zone_value_count = nyc_zone_data['LocationID'].value_counts()  # 56有两个，103有三个

    data = nyc_zone_data[['LocationID', 'zone', 'the_geom', 'borough']]
    # data = data.set_index('LocationID')

    manhattan_index = (data['borough'] == 'Manhattan')
    manhattan_data = data[manhattan_index]  # 因为有三个103，所以其实只有67个地区
    manhattan_data = manhattan_data.drop_duplicates(subset='LocationID', keep=False)  # 去除掉重复的103

    STD_PREFS = np.array(manhattan_data['zone'])
    manhattan_data['pref_name'] = manhattan_data['zone']
    manhattan_data = manhattan_data.sort_values('LocationID')
    # data = manhattan_data.reset_index()
    geom = manhattan_data['the_geom']

    geometryList = []
    for element in geom:
        element = loads(element)
        geometryList.append(element)

    manhattan_data['the_geom'] = manhattan_data['the_geom'].apply(loads)

    # 将DataFrame转换为GeoDataFrame
    manhattan_data_geo = gpd.GeoDataFrame(manhattan_data, geometry=geometryList)

    # 设置坐标系
    manhattan_data_geo.set_crs(epsg=4326, inplace=True)

    adj = phy_adj  # (66,66)
    distance = np.zeros((len(adj), len(adj)))
    for i in trange(len(adj)):
        # 得到目标区域的邻接Series，其中存储的是目标区域与其他区域是否邻接的bool类型值
        bool_row = adj[i] == 1
        # 由Series得到目标区域邻接区域的GeoSeries
        adj_polygon_series = manhattan_data_geo.geometry[bool_row]
        # 目标区域的polygon
        target_poly = manhattan_data_geo.iloc[i]['geometry']
        # 计算得到distance_series
        # distance_series = adj_polygon_series.distance(target_poly)
        distance_series = adj_polygon_series.apply(lambda polygon: polygon.centroid.distance(target_poly.centroid))
        distance[i][bool_row] = distance_series

    print_log('\tSaving distanceMatrix...', log=log)
    np.save(os.path.join(data_dir, "Manhattan-distance.npy"), distance)

    return distance


def cosine_similarity(flow1, row_index, flow2, col_index):
    vector1 = np.concatenate((flow1, np.array([row_index])))
    vector2 = np.concatenate((flow2, np.array([col_index])))
    if np.all(vector1 == 0) or np.all(vector2 == 0):
        similarity = 0
    else:
        similarity = 1 - cosine(vector1, vector2)
    return similarity


def processNewDiagA(data_dir, num_nodes, distance, trafficFlow, log):
    print_log('Processing newDiagA...', log=log)
    newA_list = []
    newDiagA_list = []

    for num in trange(trafficFlow.shape[0]):
        newA = np.zeros((num_nodes, num_nodes))

        indices = np.nonzero(distance)
        for value, row_index, col_index in zip(distance[indices], indices[0], indices[1]):
            value_backwards = torch.tensor((1 / (value * 1000)))
            similarity = F.relu(torch.tensor(
                cosine_similarity(trafficFlow[num][row_index], row_index, trafficFlow[num][col_index], col_index)))
            # similarity = F.relu(1-torch.tensor(cosine(trafic_flow[num][row_index], trafic_flow[num][col_index])))
            # if row_index == 21 & col_index == 34:
            #     print()
            newA[row_index][col_index] = F.relu(value_backwards + similarity)

        newA_list.append(newA)
        adj_diag = np.eye(num_nodes)
        newDigeA = newA + adj_diag
        newDiagA_list.append(newDigeA)

    np.save(os.path.join(data_dir, "newA_list.npy"), newA_list)
    np.save(os.path.join(data_dir, "newDiagA_list.npy"), newDiagA_list)

    return newDiagA_list


def processIndexData(data_dir, trafficFlow, log):
    print_log('Processing indexData...', log=log)
    index_list = []
    for i in tqdm(range(trafficFlow.shape[0])):
        if i + 24 < trafficFlow.shape[0] + 1:
            mid = np.zeros((3), dtype=int)
            mid[0] = i
            mid[1] = i + 12
            mid[2] = i + 24
            index_list.append(mid)
    index_list = np.array(index_list)
    train_index = int(index_list.shape[0] * 0.6)
    val_index = int(index_list.shape[0] * 0.8)

    train_index_data = index_list[:train_index, :]
    val_index_data = index_list[train_index:val_index, :]
    test_index_data = index_list[val_index:, :]

    print_log('\tSaving indexData...', log=log)
    np.savez(os.path.join(data_dir, "index.npz"), train=train_index_data, val=val_index_data, test=test_index_data)

    index = np.load(os.path.join(data_dir, "index.npz"))
    return index


def processAdaptiveAdj(batchTraficFlow, distance, device):
    batchTraficFlow = batchTraficFlow[:, -1, :, :]
    batchSize, numNodes, dim = batchTraficFlow.shape  # (16,66,8)\(16,66,128)

    adaptiveAdj_list = []

    for num in range(0, batchTraficFlow.shape[0]):
        adaptiveAdj = np.zeros((numNodes, numNodes))
        indices = np.nonzero(distance)
        for value, row_index, col_index in zip(distance[indices], indices[0], indices[1]):
            value_backwards = torch.tensor((1 / (value * 1000)))
            similarity = F.relu(
                1 - F.cosine_similarity(batchTraficFlow[num][row_index], batchTraficFlow[num][col_index], dim=0))
            adaptiveAdj[row_index][col_index] = F.relu(value_backwards + similarity)
        adaptiveAdj = adaptiveAdj + np.eye(numNodes)
        adaptiveAdj_list.append(adaptiveAdj)

    adaptiveAdj_list = np.array(adaptiveAdj_list)
    adaptiveAdj_list = torch.tensor(adaptiveAdj_list)
    adaptiveAdj_list = adaptiveAdj_list.to(device)

    return adaptiveAdj_list


def processNeighbors(phy_adj):
    neighbors_list = []
    for i in range(phy_adj.shape[0]):
        neighbor = np.nonzero(phy_adj[i])[0]
        neighbors_list.append(neighbor.tolist())
    return neighbors_list


def process_outAndinDegree(od_matrxi_list):
    out_degree_list = []
    in_degree_list = []
    for od_matrix in od_matrxi_list:
        out_degree = np.sum(od_matrix != 0, axis=1)
        out_degree_list.append(out_degree)
        in_degree = np.sum(od_matrix != 0, axis=0)
        in_degree_list.append(in_degree)

    out_degree = np.array(out_degree_list)
    in_degree = np.array(in_degree_list)

    return out_degree, in_degree
