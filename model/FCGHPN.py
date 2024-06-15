import torch.nn as nn
import torch
from model.multi_head_attention import MHA
from model.GCN_TTP import GCN
from scipy.sparse import csr_matrix
import dgl
import torch.nn.functional as F


class FCGHPN(nn.Module):
    def __init__(self, gcn_in_feats, gcn_hid_feats, gcn_output_size, att_input_size, att_output_size, att_heads,
                 num_nodes):
        super(FCGHPN, self).__init__()
        self.gcn = GCN(gcn_in_feats, gcn_hid_feats, gcn_output_size)
        self.muti_head_att = MHA(att_input_size, att_output_size, att_heads)
        self.weight = nn.Linear(2, 1)
        self.relu = nn.ReLU()
        self.weight_in = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        self.weight_out = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        nn.init.uniform_(self.weight_in, a=0, b=1e-4)
        nn.init.uniform_(self.weight_out, a=0, b=1e-4)

    def forward(self, device, adj_matrix, intensity_score, input_data, time2vec_batch, neighbors_list):
        # adj_matrix(16,66,66) intensity_score(16,66,66) input_data(16,66,1) time2vec_batch(16,66,4) neighbors_list{list:66}

        node_embedding_batch = []
        for idx in range(input_data.shape[0]):
            input_data_idx = input_data[idx]  # (66,1)
            adj_matrix_idx = F.softmax(adj_matrix[idx], dim=1)  # (66,66)
            adj_matrix_idx[adj_matrix_idx < 1e-8] = 0  # 改
            mat = csr_matrix(adj_matrix_idx)
            graph = dgl.from_scipy(mat).to(device)
            node_embedding = self.gcn(graph, input_data_idx)
            node_embedding_batch.append(node_embedding)

        node_embedding_batch = torch.stack(node_embedding_batch, dim=0)  # (16,66,4)

        time2vec_batch = time2vec_batch.to(device)
        timeAndGcn_embedding = torch.cat([node_embedding_batch, time2vec_batch], dim=-1)  # (16,66,8)

        query = timeAndGcn_embedding  # (16,66,8)

        key_list = []
        for item in neighbors_list:
            key_hid = []
            for number in item:
                key_hid.append(query[:, [number], :])
            key_list.append(key_hid)

        key = key_list  # {list:66}(list:x)
        value = key

        finally_embedding = self.muti_head_att(query, key, value, intensity_score, device)  # (16,66,1)
        # finally_embedding = self.relu(finally_embedding)
        #finally_embedding = torch.round(finally_embedding)
        embedding = finally_embedding  # (16,66,1)
        # 这里是进行OD处理————————————————————————————————————————————————————————————————————————————————————————

        expanded_tensor_a = finally_embedding.unsqueeze(3)  # (16,66,1,1)
        expanded_tensor_a_re = expanded_tensor_a.repeat(1, 1, 66, 1)  # (16,66,66,1)
        expanded_tensor_b = finally_embedding.unsqueeze(1)  # (16,1,66,1)
        expanded_tensor_b_re = expanded_tensor_b.repeat(1, 66, 1, 1)  # (16,66,66,1)
        # od_batch_martrix = torch.cat((expanded_tensor_a_re,expanded_tensor_b_re),dim=-1) # (16,66,66,2)
        # nn.sigmod（expanded_tensor_a_re）*nn.leaky——relu（expanded_tensor_b_re）
        predict_od_matrix = torch.sigmoid(expanded_tensor_a_re) * expanded_tensor_b_re  # 改
        # od_batch_martrix = self.weight(od_batch_martrix) #(16,66,66,1)
        # predict_od_matrix = od_batch_martrix.squeeze(dim=-1) #(16,66,66)
        predict_od_matrix = predict_od_matrix.squeeze(dim=-1)  # 改
        # predict_od_matrix = self.relu(predict_od_matrix)
        #predict_od_matrix = torch.round(predict_od_matrix)

        # testa = expanded_tensor_a_re.detach().cpu().numpy()
        # testb = expanded_tensor_b_re.detach().cpu().numpy()
        # testc = od_batch_martrix.detach().cpu().numpy()
        # testd = finally_embedding.detach().cpu().numpy()

        # 这里是进行in_out处理————————————————————————————————————————————————————————————————————————————————————————
        sq_embedding = embedding.squeeze()  # (16,66)
        in_degree = torch.matmul(sq_embedding, self.weight_in)
        out_degree = torch.matmul(sq_embedding, self.weight_out, )
        # return embedding,predict_od_matrix
        return embedding, predict_od_matrix, in_degree, out_degree
