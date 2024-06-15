import numpy as np
import torch
import os
from torch.utils.data import DataLoader,TensorDataset
from .utils import StandardScaler, traffic_dataset, generate_event_pickle, padding_full, Data_Batch, open_pkl_file
from .process_data import processFlowData, processTime2VecData, processEventData, processDistanceMatrix, processIndexData, processNewDiagA, process_outAndinDegree
import json
from tqdm import tqdm
from .utils import Data_Batch, print_log
from train_hawkes import get_intensity


def generate_data(data_dir, num_nodes, t2vDim, event_flag, log):
    print_log('Preparing data...', log=log)

    if os.path.exists(os.path.join(data_dir, "manhattanTrafficFlow.npy")) == False:
        manhattenZones, IDandIndex_dict, phy_adj, manhatten_data_jan, od_matrxi_list, trafficFlow= processFlowData(data_dir, log)

    else:
        manhattenZones = np.load(os.path.join(data_dir, "manhattenZones.npy"),allow_pickle=True)
        with open(os.path.join(data_dir, "IDandIndex_dict.json"), 'r') as file:
            IDandIndex_dict = json.load(file)
        phy_adj = np.load(os.path.join(data_dir, "Manhattan-adj.npy")) # (66,66)
        manhatten_data_jan = np.load(os.path.join(data_dir, "manhatten_data_jan.npy"))
        od_matrxi_list = np.load(os.path.join(data_dir, "od_matrix_list.npy")) # (8928,66,66)
        trafficFlow = np.load(os.path.join(data_dir, "manhattanTrafficFlow.npy")) # (8928,66,1)

    if os.path.exists(os.path.join(data_dir, "time2vec.npy")) == False:
        t2v = processTime2VecData(data_dir, num_nodes, t2vDim, manhatten_data_jan, IDandIndex_dict, log)
    else:
        t2v = np.load(os.path.join(data_dir, "time2vec.npy")) # (8928,66,5)

    if os.path.exists(os.path.join(data_dir, "event_list.npy")) == False:
        event_list = processEventData(data_dir, od_matrxi_list, event_flag, log)
    else:
        event_list = np.load(os.path.join(data_dir, "event_list.npy")) # (8927,66,66)

    if os.path.exists(os.path.join(data_dir, "Manhattan-distance.npy")) == False:
        distance = processDistanceMatrix(data_dir, phy_adj, log)
    else:
        distance = np.load(os.path.join(data_dir, "Manhattan-distance.npy")) # (66,66)

    if os.path.exists(os.path.join(data_dir, "newDiagA_list.npy")) == False:
        newDiagA_list = processNewDiagA(data_dir, num_nodes, distance, trafficFlow, log)
    else:
        newDiagA_list = np.load(os.path.join(data_dir, "newDiagA_list.npy"))

    return phy_adj, od_matrxi_list, trafficFlow, t2v, event_list, distance, newDiagA_list

def load_data(data_dir, generate_A, intensity_score_list, od_matrxi_list, trafficFlow, t2v, batch_size, log):
    adj = generate_A
    data = trafficFlow
    out_degree, in_degree = process_outAndinDegree(od_matrxi_list)
    # (8928,66)  (8928,66)

    input_data = data[2::2]  # (4463,66,1)
    target_data = data[3::2]  # (4463,66,1)

    t2v = t2v[2::2]  # (4463,66,5)
    t2v = t2v[:, :, 1:]  # (4463,66,4)

    input_od_matrix = od_matrxi_list[2::2]  # (4463,66,66)
    target_od_matrix = od_matrxi_list[3::2]  # (4463,66,66)

    input_in_degree = in_degree[2::2]  # (4463,66)
    target_in_degree = in_degree[3::2]  # (4463,66)

    input_out_degree = out_degree[2::2]  # (4463,66)
    target_out_degree = out_degree[3::2]  # (4463,66)

    train_index = int(len(input_data) * 0.6)  # 2677
    val_index = int(len(input_data) * 0.8)  # 3570

    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(input_data[:train_index]),  # (2677,66,1)
            torch.Tensor(target_data[:train_index]),  # (2677,66,1)
            torch.Tensor(adj[:train_index]),  # (2677,66,66)
            torch.Tensor(intensity_score_list[:train_index]),  # (2677,66,66)
            torch.Tensor(t2v[:train_index]),  # (2677,66,4)
            torch.Tensor(target_od_matrix[:train_index]),  # (2677,66,66)
            torch.Tensor(target_in_degree[:train_index]),
            torch.Tensor(target_out_degree[:train_index])
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        # DataLoader:669
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(input_data[train_index:val_index]),  # (893,66,1)
            torch.Tensor(target_data[train_index:val_index]),  # (893,66,1)
            torch.Tensor(adj[train_index:val_index]),  # (893,66,66)
            torch.Tensor(intensity_score_list[train_index:val_index]),  # (893,66,66)
            torch.Tensor(t2v[train_index:val_index]),  # (893,66,4)
            torch.Tensor(target_od_matrix[train_index:val_index]),  # (893,66,66)
            torch.Tensor(target_in_degree[train_index:val_index]),
            torch.Tensor(target_out_degree[train_index:val_index])
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        # DataLoader:223
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(input_data[val_index:]),  # (893,66,1)
            torch.Tensor(target_data[val_index:]),  # (893,66,1)
            torch.Tensor(adj[val_index:]),  # (893,66,66)
            torch.Tensor(intensity_score_list[val_index:]),  # (893,66,66)
            torch.Tensor(t2v[val_index:]),  # (893,66,4)
            torch.Tensor(target_od_matrix[val_index:]),  # (893,66,66)
            torch.Tensor(target_in_degree[val_index:]),
            torch.Tensor(target_out_degree[val_index:])
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
        # DataLoader:223
    )
    return train_loader, val_loader, test_loader

def generate_hawkes_data(data_dir, event_list, phy_adj, log):
    print_log('Preparing hawkes event data...', log=log)

    all_event_data = event_list  # all_event_data(16979,307,307)

    # 得到哪些节点之间会存在联系
    non_zero_index = np.nonzero(phy_adj)
    # num_nonzero=np.count_nonzero(dist)
    p1, p2 = non_zero_index

    # 生成对应的详细事件细节信息文件
    # pkl文件记录了一共有多少对节点之间发生了事件
    # 如train.pkl中的list：21，表示有21对节点在某些时刻发生了事件
    # 每一个对节点中又有长度为一共有多少个时间index发生了事件且记录了发生事件的详细信息list
    # 如 list21中的第一个list：2131，表示这一对节点一共发生了2131个事件
    # 事件的详细信息记录为：type_event事件类型、time_since_last_event距离上一个事件的时间差、node_ij节点对、index：事件发生的时间index
    if os.path.exists(os.path.join(data_dir, "all.pkl")) == False:
        generate_event_pickle('all.pkl', all_event_data, p1, p2, data_dir)

    # 得到pkl文件中具体的每对节点发生事件的时间间隔list、事件类型list、这对节点一共发生了多少事件(list)
    # 如train.pkl中，train_time_duration{list:21}/type_train{list:21}/seq_lens_train{list:21}
    all_time_duration, type_all, seq_lens_all, all_event_timeIndex, node_ij_all = open_pkl_file(
        os.path.join(data_dir, "all.pkl"), 'all')

    # 填充数据,使得数据补齐到相同的维度
    type_size = 2
    # train_time_duration的shape为（21，max(seq_lens_train)+1），表示21对节点中，每一对节点中每一件事件发生的时间差值
    # type_train的shape为（21，max(seq_lens_train)+1），表示21对节点中，每一对节点中每一件事件发生的类型，第一个类型为type_size
    all_time_duration, type_all = padding_full(all_time_duration, type_all, seq_lens_all, type_size)

    # 构建事件DataLoader
    all_event_dataset = Data_Batch(all_time_duration, type_all, seq_lens_all)

    event_batch = 64
    all_event_loader = DataLoader(all_event_dataset, batch_size=event_batch, shuffle=False)

    print_log('hawkes event data is done', log=log)

    return all_event_loader

def generate_intensity(event_16batch_dataset, hawkes_model, p1, p2, nodes, device):
    batch_intensity_list = []
    batch_type_list = []

    for m in range(event_16batch_dataset.shape[0]):
        adj_event_list = []
        event_data = event_16batch_dataset[m]
        # event_data = np.array(event_data)
        for idx, (i, j) in enumerate(zip(p1, p2)):
            # 如果两个节点在所有的时间节点上都没有事件发生，即event_data的值为0时直接跳过
            if torch.all(event_data[:, i, j] == 0):
                continue
            else:
                # 得到两个节点有事件发生的时间节点列表
                events = torch.nonzero(event_data[:, i, j])
                previous = 0
                node_event_list = []
                # 记录下这两节点发生的事件的详细信息：包括事件类型、距离上一次事件发生的事件、节点元组、事件发生的时间index
                for k in events:
                    e = {
                        'type_event': event_data[k, i, j] - 1,
                        'time_since_last_event': k - previous,
                        'node_ij': (i, j),
                        'event_time_index': k
                    }
                    previous = k
                    node_event_list.append(e)
                adj_event_list.append(node_event_list)
                # adj_event_list存储的是每一对存在事件的节点对所发生的事件
                # adj_event_list存储node_event_list，node_event_list存储e（dict）

        if len(adj_event_list) == 0:
            batch_intensity_list.append(None)
            batch_type_list.append(None)
            continue

        time_durations = []
        type_seqs = []
        seq_lens = []
        event_timeIndex = []
        node_ij = []

        for i in range(len(adj_event_list)):  # 当前12个事件节点中一共有多少对节点发生了事件
            seq_lens.append(len(adj_event_list[i]))  # 存储每一对节点发生了多少事件
            type_seqs.append(torch.LongTensor([float(event['type_event']) for event in adj_event_list[i]]).to(device))
            time_durations.append(
                torch.FloatTensor([float(event['time_since_last_event']) for event in adj_event_list[i]]).to(device))
            event_timeIndex.append(
                torch.FloatTensor([float(event['event_time_index']) for event in adj_event_list[i]]).to(device))
            node_ij.append(adj_event_list[i][0]['node_ij'])  # 存储每一对节点

        type_size = 2
        max_len = max(seq_lens)  # 53对节点中，发生最多事件的数量作为最终的长度
        batch_size = len(time_durations)  # 53
        time_duration_padded = torch.zeros(size=(batch_size, max_len + 1)).to(device)
        type_train_padded = torch.zeros(size=(batch_size, max_len + 1), dtype=torch.long).to(device)
        for idx in range(batch_size):
            time_duration_padded[idx, 1:seq_lens[idx] + 1] = time_durations[idx]
            type_train_padded[idx, 0] = type_size
            type_train_padded[idx, 1:seq_lens[idx] + 1] = type_seqs[idx]

        event_dataset = Data_Batch(time_duration_padded, type_train_padded, seq_lens)
        event_loader = DataLoader(event_dataset, batch_size=8, shuffle=False)

        intensity = get_intensity(event_loader, hawkes_model, device)  # (12个时间节点中有事件发生的节点对数，某一对节点对发生最多的事件数量，2)
        longest_event_timeIndex = max(event_timeIndex, key=lambda x: x.size(0))
        allTimes_intensity = np.zeros((event_data.shape[0], nodes, nodes))  # (12,307,307) （十二个时间节点，307，307）
        allTime_type = np.zeros((event_data.shape[0], nodes, nodes))  # (12,307,307)（十二个时间节点，307，307）

        for i in range(intensity.shape[0]):  # 取出12个时间节点中发生了事件的节点对信息
            node_i = node_ij[i][0]
            node_j = node_ij[i][1]
            for j in range(intensity.shape[1]):  # j表示是最长事件长度中的哪一个下标
                time_index = longest_event_timeIndex[j]
                time_index = time_index.item()
                allTimes_intensity[int(time_index), int(node_i), int(node_j)] = max(intensity[i, j, 0],
                                                                                    intensity[i, j, 1])
                allTime_type[int(time_index), int(node_i), int(node_j)] = np.argmax(intensity[i, j, :]) + 1

        need_intensity = allTimes_intensity[-1]  # #（307，307）问题：取了第12个时刻的intensity！！！基本上都是0
        need_type = allTime_type[-1]  # （307，307）问题：取了第12个时刻的intensity！！！基本上都是0
        batch_intensity_list.append(need_intensity)  # test = np.count_nonzero(need_intensity)/need_intensity.size = 0
        batch_type_list.append(need_type)

    return batch_intensity_list, batch_type_list


