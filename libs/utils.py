import numpy as np
import torch
import pickle
import os


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=False, tolerance=1e-1):
        self.max_round = max_round#10
        self.num_round = 0

        self.epoch_count = 1
        self.best_epoch = 1

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round, (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance

class Data_Batch:
    def __init__(self, duration, events_type, seq_len):
        self.duration = duration
        self.events_type = events_type
        self.seq_len = seq_len

    def __len__(self):
        return self.events_type.shape[0]

    def __getitem__(self, index):
        sample = {
            'event_seq': self.events_type[index],
            'duration_seq': self.duration[index],
            'seq_len': self.seq_len[index]
        }
        return sample


def generate_event_pickle(file_name, event_data, p1, p2, data_dir):
    adj_event_list = []
    for idx, (i, j) in enumerate(zip(p1, p2)):
        # 如果两个节点在所有的时间节点上都没有事件发生，即event_data的值为0时直接跳过
        if np.all(event_data[:, i, j] == 0):
            continue
        else:
            # 得到两个节点有事件发生的时间节点列表
            events = np.nonzero(event_data[:, i, j])
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
    data = {file_name.split('.')[0]: adj_event_list}
    with open(os.path.join(data_dir, file_name), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def open_pkl_file(path, description):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        data = data[description]

    time_durations = []
    type_seqs = []
    seq_lens = []
    event_timeIndex = []
    node_ij = []

    for i in range(len(data)):
        seq_lens.append(len(data[i]))
        type_seqs.append(torch.LongTensor([float(event['type_event']) for event in data[i]]))
        time_durations.append(torch.FloatTensor([float(event['time_since_last_event']) for event in data[i]]))
        event_timeIndex.append(torch.FloatTensor([float(event['event_time_index']) for event in data[i]]))
        node_ij.append(data[i][0]['node_ij'])
    return time_durations, type_seqs, seq_lens, event_timeIndex, node_ij


def padding_full(time_duration, type_train, seq_lens_list, type_size):
    max_len = max(seq_lens_list)  # 53对节点中，发生最多事件的数量作为最终的长度
    batch_size = len(time_duration)  # 53
    time_duration_padded = torch.zeros(size=(batch_size, max_len + 1))
    type_train_padded = torch.zeros(size=(batch_size, max_len + 1), dtype=torch.long)
    for idx in range(batch_size):
        time_duration_padded[idx, 1:seq_lens_list[idx] + 1] = time_duration[idx]
        type_train_padded[idx, 0] = type_size
        type_train_padded[idx, 1:seq_lens_list[idx] + 1] = type_train[idx]
    return time_duration_padded, type_train_padded


def generate_simulation(durations, seq_len):
    max_seq_len = max(seq_len)
    simulated_len = max_seq_len * 5
    sim_durations = torch.zeros(durations.shape[0], simulated_len)  # (8,seq_len中的最大值的5倍)
    sim_duration_index = torch.zeros(durations.shape[0], simulated_len, dtype=torch.long)  # (8,seq_len中的最大值的5倍)
    total_time_seqs = []
    for idx in range(durations.shape[0]):  # 0-7
        # 取出8对节点中，每一对节点，所有发生的事件的时间index，
        # range(1, seq_len[idx]+2)])表示1到seq_len[idx]+1
        # [:seq_len[idx]+1]表示从0到seq_len[idx]，因此time_seq[-1]是所有事件的index的和
        time_seq = torch.stack([torch.sum(durations[idx][:i]) for i in range(1, seq_len[idx] + 2)])
        # 最后一个时间index是之前所有事件发生时间index
        total_time = time_seq[-1].item()
        total_time_seqs.append(total_time)  # 正态分布发生simulated_len个事件
        sim_time_seq, _ = torch.sort(torch.empty(simulated_len).uniform_(0, total_time))
        sim_duration = torch.zeros(simulated_len)
        # print(sim_time_seq)
        # print(time_seq)

        for idx2 in range(time_seq.shape.__getitem__(-1)):
            # 某一对节点中，所有发生事件的时间index
            # 如果随机生成的事件时间大于真实某个事件的时间，记录下位置index信息
            duration_index = sim_time_seq > time_seq[idx2].item()
            # 将随机生成的事件时间（大于真实的值）改为两者之差
            sim_duration[duration_index] = sim_time_seq[duration_index] - time_seq[idx2]
            # sim_duration_index记录了sim_duration哪些位置因为哪个真实事件时间改变了值
            sim_duration_index[idx][duration_index] = idx2

        sim_durations[idx, :] = sim_duration[:]
    total_time_seqs = torch.tensor(total_time_seqs)
    # print(sim_duration_index)
    # print(total_time_seqs)
    # print(sim_durations)

    # sim_durations记录了8对节点对中随机生成的并根据真实的事件更正的事件
    # total_time_seqs所有真实事件的时间index
    # sim_duration_index记录了8对节点对中随机生成的并根据真实的事件更正的事件是根据哪些具体的事件index改变的
    return sim_durations, total_time_seqs, sim_duration_index

class StandardScaler:
    # 按均值、方差归一化

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class traffic_dataset():
    def __init__(self, flow_dataset, flow_label_set, t2v_dataset, event_dataset, od_dataset, od_label_set, scaler):
        flow_dataset = np.array(flow_dataset)
        flow_dataset[..., 0] = scaler.transform(flow_dataset[..., 0])
        flow_dataset = np.split(flow_dataset,flow_dataset.shape[0])
        self.flow_dataset = flow_dataset
        self.flow_label_set = flow_label_set
        self.t2v_dataset = t2v_dataset
        self.event_dataset = event_dataset
        self.od_dataset = od_dataset
        self.od_label_set = od_label_set

    def __getitem__(self, index):
        self.flow_data = self.flow_dataset[index]
        self.flow_label = self.flow_label_set[index]
        self.t2v_data = self.t2v_dataset[index]
        self.event_data = self.event_dataset[index]
        self.od_data = self.od_dataset[index]
        self.od_label = self.od_label_set[index]
        sample = {
            'flow_data': self.flow_data,
            'flow_label': self.flow_label,
            't2v_data':self.t2v_data,
            'event_data': self.event_data,
            'od_data': self.od_data,
            'od_label': self.od_label
        }

        return sample

    def __len__(self):
        return len(self.flow_dataset)

def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()




