import torch
from torch.utils.data import DataLoader
import dgl
import numpy as np
from scipy.sparse import csr_matrix
from model.GCN_TTP import GCN_hawkes, discriminator
import torch.optim as optim
import os


def hawkes_train(device, num_nodes, epochs, event_list, newDiagA_list, trafic_flow, parameter_path):
    # event_list(8927,66,66)
    # newDiagA_list(8928,66,66)
    # trafic_flow(8928,66,1)

    history_event = torch.FloatTensor(event_list)
    history_event = DataLoader(history_event, batch_size=2, drop_last=True)
    # {DataLoader:4463}
    newDiagA_list = torch.FloatTensor(newDiagA_list)

    trafic_flow = torch.from_numpy(trafic_flow)
    trafic_flow = trafic_flow.float()

    time_split = torch.FloatTensor([10, 5]).to(device)

    index = newDiagA_list.shape[0]
    if index % 2 == 0:
        index = index - 2
    else:
        index - 1
    true_A = newDiagA_list[2::2]  # (4463,66,66)
    origin_A = newDiagA_list[:index:2]  # (4463,66,66)
    true_A, origin_A = true_A.to(device), origin_A.to(device)

    cal_GCN = False
    h_begin1 = torch.FloatTensor([0.])
    h_begin2 = torch.FloatTensor([0.])

    graph_two = []
    mat0, mat1 = csr_matrix(newDiagA_list[0]), csr_matrix(newDiagA_list[1])
    g0, g1 = dgl.from_scipy(mat0), dgl.from_scipy(mat1)
    graph_two.append(g0)
    graph_two.append(g1)

    input_feat_two = []
    data0, data1 = trafic_flow[0], trafic_flow[1]  # (66,1)
    input_feat_two.append(data0)
    input_feat_two.append(data1)

    M = GCN_hawkes(1, 99, 198, num_nodes)
    D = discriminator()
    M = M.to(device)
    D = D.to(device)
    optimize = optim.Adam(params=M.parameters(), lr=0.0001)
    # M.load_state_dict(torch.load('/home/user/yht/paper_code/oneMonth/data/nyc_taxi_2019_01/hawkes_parameters_210'))


    print('\nTraining Hawkes.')
    for epoch in range(epochs):
        hawkes_loss_sum = 0

        for idx, (two_history_event, true_adj) in enumerate(zip(history_event, true_A)):

            if idx == 0:
                cal_GCN = True

            optimize.zero_grad()

            generate_adj, h_begin1, h_begin2, intensity_score = M(graph_two, input_feat_two, time_split,
                                                                  two_history_event, cal_GCN, h_begin1, h_begin2,
                                                                  origin_A[idx], device)

            h_begin1 = h_begin1.detach()
            h_begin2 = h_begin2.detach()

            hawkes_loss = D(true_adj, generate_adj)
            hawkes_loss_sum = hawkes_loss_sum + hawkes_loss.item()

            optimize.step()
            cal_GCN = False

        print('\nEpoch({}):loss:{}'.format(epoch + 1, hawkes_loss_sum))
    print('\n\nTraining finished.')

    if os.path.exists(parameter_path):
        os.remove(parameter_path)
    torch.save(M.state_dict(), parameter_path)



def cal_intensityAdj(device, num_nodes, event_list, newDiagA_list, trafic_flow, parameter_path):
    history_event = torch.FloatTensor(event_list)
    history_event = DataLoader(history_event, batch_size=2, drop_last=True)
    # {DataLoader:4463}
    newDiagA_list = torch.FloatTensor(newDiagA_list)

    trafic_flow = torch.from_numpy(trafic_flow)
    trafic_flow = trafic_flow.float()

    # test = torch.Tensor([10,5]).to('cuda:1')
    time_split = torch.FloatTensor([10, 5]).to(device)
    true_A = newDiagA_list[2::2]  # (4463,263,263)
    origin_A = newDiagA_list[:8926:2]  # (4463,263,263)
    true_A, origin_A = true_A.to(device), origin_A.to(device)

    cal_GCN = False

    graph_two = []
    mat0, mat1 = csr_matrix(newDiagA_list[0]), csr_matrix(newDiagA_list[1])
    g0, g1 = dgl.from_scipy(mat0), dgl.from_scipy(mat1)
    graph_two.append(g0)
    graph_two.append(g1)

    input_feat_two = []
    data0, data1 = trafic_flow[0], trafic_flow[1]  # (66,1)
    input_feat_two.append(data0)
    input_feat_two.append(data1)

    M = GCN_hawkes(1, 99, 198, num_nodes)
    M = M.to(device)
    optimize = optim.Adam(params=M.parameters(), lr=0.0001)

    M.load_state_dict(torch.load(parameter_path))

    generate_A = []
    intensity_score_list = []
    h_begin1 = torch.FloatTensor([0.]).to(device)
    h_begin2 = torch.FloatTensor([0.]).to(device)

    for idx, Two_history_event in enumerate(history_event):

        if idx == 0:
            cal_GCN = True

        optimize.zero_grad()
        generate_adj, h_begin1, h_begin2, intensity_score = M(graph_two, input_feat_two, time_split, Two_history_event,
                                                              cal_GCN, h_begin1, h_begin2, origin_A[idx], device)

        generate_adj = generate_adj.detach()
        generate_adj = generate_adj.cpu()
        generate_A.append(np.array(generate_adj))

        intensity_score = intensity_score.detach()
        intensity_score = intensity_score.cpu()
        intensity_score_list.append(np.array(intensity_score))

        h_begin1 = h_begin1.detach()
        h_begin2 = h_begin2.detach()
        cal_GCN = False

    generate_A = np.stack(generate_A)  # (4463,66,66)
    intensity_score_list = np.stack(intensity_score_list)  # (4463,66,66)

    # if os.path.exists('/home/user/yht/paper_code/Manhattan/data/nyc_taxi_2019_01/generate_A.npy'):
    #     os.remove('/home/user/yht/paper_code/Manhattan/data/nyc_taxi_2019_01/generate_A.npy')
    # np.save('/home/user/yht/paper_code/Manhattan/data/nyc_taxi_2019_01/generate_A.npy', generate_A)

    return generate_A, intensity_score_list


if __name__ == "__main__":
    train = True
    device = 'cuda:1'
    if train:
        train_intensityAdj(device)
    else:
        cal_intensityAdj(device)
