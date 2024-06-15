import torch
from torch import nn
from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import numpy as np


class hawkes_parameter(nn.Module):
    def __init__(self,num_nodes,num_event_type):
        # 66,3
        super(hawkes_parameter, self).__init__()
        self.input_w=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        # input_w=nn.Parameter(torch.FloatTensor(66,66,3))
        self.input_u = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # input_u=nn.Parameter(torch.FloatTensor(66,66))
        self.input_d=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        # input_d=nn.Parameter(torch.FloatTensor(66,66,3))
        self.forget_w=nn.Parameter(torch.FloatTensor(num_nodes,num_nodes,num_event_type),requires_grad=True)
        # forget_w=nn.Parameter(torch.FloatTensor(66,66,3))
        self.forget_u=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # forget_u=nn.Parameter(torch.FloatTensor(66,66))
        self.forget_d=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        # forget_d=nn.Parameter(torch.FloatTensor(66,66,3))
        self.output_w=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        # output_w=nn.Parameter(torch.FloatTensor(66,66,3))
        self.output_u=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # output_u=nn.Parameter(torch.FloatTensor(66,66))
        self.output_d=nn.Parameter(torch.FloatTensor(num_nodes, num_nodes,num_event_type), requires_grad=True)
        # output_d=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_w = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_w=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_u = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_u=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_d = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_d=nn.Parameter(torch.FloatTensor(66,66,3))
        self.input_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # input_w2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.input_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # input_u2=nn.Parameter(torch.FloatTensor(66,66))
        self.input_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # input_d2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.forget_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # forget_w2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.forget_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # forget_u2=nn.Parameter(torch.FloatTensor(66,66))
        self.forget_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # forget_d2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.output_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # output_w2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.output_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)
        # output_u2=nn.Parameter(torch.FloatTensor(66,66))
        self.output_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # output_d2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_w2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_w2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_u2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_u2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.limit_d2 = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes, num_event_type), requires_grad=True)
        # limit_d2=nn.Parameter(torch.FloatTensor(66,66,3))
        self.sigmoid=torch.nn.Sigmoid()
        self.tanh=torch.nn.Tanh()
        # 从均匀分布U(a,b)中生成值，a是分布的下界，b是分布的上界
        nn.init.uniform_(self.input_w, a=0, b=1e-4)
        nn.init.uniform_(self.input_u, a=0, b=1e-4)
        nn.init.uniform_(self.input_d, a=0, b=1e-4)
        nn.init.uniform_(self.output_u, a=0, b=1e-4)
        nn.init.uniform_(self.output_w, a=0, b=1e-4)
        nn.init.uniform_(self.output_d, a=0, b=1e-4)
        nn.init.uniform_(self.forget_u, a=0, b=1e-4)
        nn.init.uniform_(self.forget_d, a=0, b=1e-4)
        nn.init.uniform_(self.forget_w, a=0, b=1e-4)
        nn.init.uniform_(self.limit_w, a=0, b=1e-4)
        nn.init.uniform_(self.limit_u, a=0, b=1e-4)
        nn.init.uniform_(self.limit_d, a=0, b=1e-4)
        nn.init.uniform_(self.input_u2, a=0, b=1e-4)
        nn.init.uniform_(self.input_d2, a=0, b=1e-4)
        nn.init.uniform_(self.input_w2, a=0, b=1e-4)
        nn.init.uniform_(self.output_u2, a=0, b=1e-4)
        nn.init.uniform_(self.output_d2, a=0, b=1e-4)
        nn.init.uniform_(self.output_w2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_u2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_d2, a=0, b=1e-4)
        nn.init.uniform_(self.forget_w2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_d2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_u2, a=0, b=1e-4)
        nn.init.uniform_(self.limit_w2, a=0, b=1e-4)

    def forward(self,history_event1,u_begin1,history_event2,u_begin2):
        # history_event1:(66,66),u_begin1(66,66,3),history_event2:(66,66),u_begin2(66,66,3)

        ht1 = self.tanh(u_begin1)
        # 在第一轮循环中，可知ht1是神经霍克斯矩阵的初始隐状态信息

        if torch.any(torch.isnan(self.input_w)):
            print('error')

        in_put = self.sigmoid(torch.matmul(history_event1, self.input_w) + torch.matmul(self.input_u, ht1) + self.input_d)
        # 相当于公式18

        out_put = self.sigmoid(torch.matmul(history_event1,self.output_w ) + torch.matmul(self.output_u ,ht1) + self.output_d)
        # 相当于公式20

        for_get = self.sigmoid(torch.matmul(history_event1,self.forget_w)+ torch.matmul(self.forget_u ,ht1)+ self.forget_d)
        # 相当于公式19

        z = self.tanh(torch.matmul(history_event1, self.limit_w ) + self.limit_u * ht1 + self.limit_d)
        # 相当于公式21

        ht2 = self.tanh(u_begin2)
        # ht2是神经霍克斯矩阵的初始隐状态信息
        in_put2 = self.sigmoid(torch.matmul(history_event2, self.input_w2) + torch.matmul(self.input_u2, ht2) + self.input_d2)
        # 相当于公式18
        out_put2 = self.sigmoid(torch.matmul(history_event2, self.output_w2) + torch.matmul(self.output_u2, ht2) + self.output_d2)
        # 相当于公式20
        for_get2 = self.sigmoid(torch.matmul(history_event2, self.forget_w2) + torch.matmul(self.forget_u2, ht2) + self.forget_d2)
        # 相当于公式19
        z2 = self.tanh(torch.matmul(history_event2, self.limit_w2) + self.limit_u2 * ht2 + self.limit_d2)
        # 相当于公式21
        ct = for_get * ht1+ in_put * z
        # 相当于公式15
        ct2 = for_get2 * ht2+ in_put2 * z2
        # 相当于公式16

        if torch.any(torch.isnan(ct)):
            print('error')

        if torch.any(torch.isnan(ct2)):
            print('error')

        return ct,out_put,ct2,out_put2

class generator(nn.Module):
    def __init__(self, num_nodes):
        # num_nodes=66
        super(generator, self).__init__()

        self.w1 = nn.Parameter(torch.FloatTensor(num_nodes, 1), requires_grad=True)

        self.w2 = nn.Parameter(torch.FloatTensor(num_nodes, 1), requires_grad=True)

        self.w = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes), requires_grad=True)

        self.linear = nn.Linear(3, 1)

        self.hakwkes_paramet = hawkes_parameter(num_nodes, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus()
        nn.init.uniform_(self.w2, a=0, b=1)
        nn.init.uniform_(self.w, a=0, b=1)
        nn.init.uniform_(self.w1, a=0, b=1)

    def forward(self, time, history_event, u_begin1, u_begin2):
        # time=FloatTensor([10, 5]),history_event(2,66,1),u_begin1(66,1),u_begin2(66,1)

        ct1, ot1, ct2, ot2 = self.hakwkes_paramet(history_event[0], u_begin1, history_event[1], u_begin2)
        # history_event[0]:(66,66),u_begin1（66，66，3）,history_event[1]:(66,66),u_begin2（66，66，3）
        # ct1(66,66,3),ot1(66,66,3),ct2(66,66,3),ot2(66,66,3)==>ct1对应公式15,ct2对应公式16

        q1 = -torch.matmul(history_event[0], self.w1)
        # (66,1)，相当于公式17
        q2 = -torch.matmul(history_event[1], self.w2)
        # (66,1)，相当于公式17

        cell = ct2 + (ct1 - ct2) * (torch.exp((q1) * time[0]) + torch.exp(q2 * time[1]))
        # 相当于公式14，(66,66,3)

        u_begin1 = ot1 * self.tanh(cell)
        # 根据LSTM公式更新隐状态1(66,66,3)
        u_begin2 = ot2 * self.tanh(cell)
        # 根据LSTM公式更新隐状态2(66,66,3)

        intensity_socre = self.softplus(torch.matmul(self.w, u_begin2))
        # 经过softplus函数得到正输出，(66,66,3)
        intensity_socre = self.linear(intensity_socre).squeeze()
        # (66,66)

        if torch.any(torch.isnan(intensity_socre)):
            print('error')

        return intensity_socre, u_begin1, u_begin2



class GCN(nn.Module):
    def __init__(self, inDim, hiddenDim, outDim):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(inDim, hiddenDim,allow_zero_in_degree=True).cuda()
        self.conv2 = GraphConv(hiddenDim, outDim,allow_zero_in_degree=True).cuda()

    def forward(self, g, input):
        hidden = self.conv1(g, input)
        hidden = F.relu(hidden)
        output = self.conv2(g, hidden)
        return output

class GCN_hawkes(nn.Module):
    def __init__(self,in_feats, h_feats, output_size,num_nodes):
        super(GCN_hawkes, self).__init__()
        self.gcn = GCN(in_feats, h_feats, output_size)
        self.hawkes = generator(num_nodes)
        self.finally_w = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes),requires_grad=True)
        nn.init.uniform_(self.finally_w,a=0,b=1e-4)

    def forward(self,g_two,data_two,time,history_event,cal_GCN,h_begin1,h_begin2,origin_A,device):
        # history_event(2,66,66),cal_GCN(True)
        if cal_GCN == True:
            h_begin_1 = self.gcn(g_two[0].to(device),data_two[0].to(device)) #(66,198)
            h_begin_2 = self.gcn(g_two[1].to(device),data_two[1].to(device)) #(66,198)
            h_begin_1,h_begin_2 = h_begin_1.reshape(66,66,-1),h_begin_2.reshape(66,66,-1)
        else:
            h_begin_1 = h_begin1
            h_begin_2 = h_begin2

        history_event = history_event.to(device)
        intensity_score,h_begin_1,h_begin_2 = self.hawkes(time,history_event,h_begin_1,h_begin_2)

        generate_adj = torch.matmul(self.finally_w,intensity_score*origin_A)
        torch.cuda.empty_cache()

        return generate_adj,h_begin_1,h_begin_2,intensity_score

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.lose_functional = torch.nn.MSELoss()
    def forward(self, ture_adj ,generate_adj):
        loss = self.lose_functional(generate_adj,ture_adj)
        loss.backward()
        return loss



