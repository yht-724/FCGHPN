import torch
import torch.nn as nn
import numpy as np

class MHA(nn.Module):
    def __init__(self,input_dim,output_dim,n_heads):
        super(MHA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads

        assert output_dim % n_heads ==0

        # self.w_q = nn.Parameter(torch.FloatTensor(input_dim,output_dim), requires_grad=True)
        # self.w_k = nn.Parameter(torch.FloatTensor(input_dim,output_dim), requires_grad=True)
        # self.w_v = nn.Parameter(torch.FloatTensor(input_dim,output_dim), requires_grad=True)
        self.w_q = nn.Linear(input_dim, output_dim)
        self.w_k = nn.Linear(input_dim, output_dim)
        self.w_v = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(output_dim,16)
        self.fc2 = nn.Linear(16,1)

        self.scale = torch.sqrt(torch.FloatTensor([output_dim // n_heads]))

        # nn.init.uniform_(self.w_q, a=0, b=1e-4)
        # nn.init.uniform_(self.w_k, a=0, b=1e-4)
        # nn.init.uniform_(self.w_v, a=0, b=1e-4)
        # nn.init.uniform_(self.w_q.weight, a=0, b=1e-8)
        # nn.init.uniform_(self.w_k.weight, a=0, b=1e-8)
        # nn.init.uniform_(self.w_v.weight, a=0, b=1e-8)
        # nn.init.uniform_(self.fc1.weight, a=0, b=1e-8)
        # nn.init.uniform_(self.fc2.weight, a=0, b=1e-8)

    def forward(self,query,key,value,intensity_score,device):
        # query(4,263,8),key{list:263}(list:x),value{list:263}(list:x),intensity_score(4,263,263)

        batch_size = query.shape[0]
        # del Q = self.w_q(torch.Tensor(query).to(device)) #(batch_size,num_zone,input_dim)->(batch_size,num_zone,output_dim)
        #Q = torch.matmul(query,self.w_q) #(batch_size,num_zone,input_dim)->(batch_size,num_zone,output_dim)
        Q = self.w_q(query)

        finally_embedding_list = []
        for i in range(Q.shape[1]):
            Q_item = Q[:,i,:].reshape(batch_size,-1,self.output_dim) #(batch_size,1,output_dim)
            Q_item = Q_item.view(batch_size, -1, self.n_heads, self.output_dim // self.n_heads).permute(0, 2, 1, 3)
            #(batch_size,1,output_dim)->(batch_size,1,n_heads,output_dim // n_heads)->(batch_size,n_heads,1,output_dim // n_heads)

            key_item_list = key[i] #{list:x}(batch_size,1,input_dim)
            #del K_item_cpu = np.concatenate([k for k in key_item_list],axis=1) #(batch_size,x,input_dim)
            K_item_cpu = torch.cat(key_item_list, dim=1)  # (batch_size,x,input_dim)
            # K_item_gpu = self.w_k(torch.Tensor(K_item_cpu).to(device)) #(batch_size,x,output_dim)
            #K_item_gpu = torch.matmul(K_item_cpu,self.w_k)  # (batch_size,x,output_dim)
            K_item_gpu = self.w_k(K_item_cpu)
            K_item = K_item_gpu.view(batch_size, -1, self.n_heads, self.output_dim // self.n_heads).permute(0, 2, 1, 3)
            # (batch_size,x,output_dim)->(batch_size,x,n_heads,output_dim // n_heads)->(batch_size,n_heads,x,output_dim // n_heads)

            # del V_item_gpu = self.w_v(torch.Tensor(K_item_cpu).to(device)) #(batch_size,x,output_dim)
            #V_item_gpu = torch.matmul(K_item_cpu,self.w_v) # (batch_size,x,output_dim)
            V_item_gpu = self.w_v(K_item_cpu)
            V_item = V_item_gpu.view(batch_size, -1, self.n_heads, self.output_dim // self.n_heads).permute(0, 2, 1, 3)
            # (batch_size,x,output_dim)->(batch_size,x,n_heads,output_dim // n_heads)->(batch_size,n_heads,x,output_dim // n_heads)

            attention_score_item = torch.matmul(Q_item, K_item.permute(0, 1, 3, 2)) / self.scale.to(device)
            # (batch_size,n_heads,1,output_dim // n_heads)*(batch_size,n_heads,output_dim // n_heads,x)->(batch_size,n_heads,1,x)
            attention_score_item = torch.softmax(attention_score_item, dim=-1)

            finally_embedding_item = torch.matmul(attention_score_item,V_item)
            # (batch_size,n_heads,1,x)*(batch_size,n_heads,x,output_dim // n_heads)->(batch_size,n_heads,1,output_dim // n_heads)
            finally_embedding_item = finally_embedding_item.permute(0, 2, 1, 3).contiguous()
            # (batch_size,n_heads,1,output_dim // n_heads)->(batch_size,1,n_heads,output_dim // n_heads)
            finally_embedding_item = finally_embedding_item.view(batch_size, -1, self.n_heads * (self.output_dim // self.n_heads))
            # (batch_size,1,n_heads,output_dim // n_heads)->(batch_size,1,output_dim)

            # del finally_embedding_item = finally_embedding_item.detach().cpu().numpy()
            finally_embedding_list.append(finally_embedding_item)

        #del finally_embedding = np.concatenate([h for h in finally_embedding_list],axis=1) #(batch_size,263,output_dim)
        finally_embedding = torch.cat(finally_embedding_list,dim=1) #(batch_size,263,output_dim)
        # del finally_embedding = torch.Tensor(finally_embedding).permute(0,2,1).contiguous()
        finally_embedding = finally_embedding.permute(0, 2, 1).contiguous()
        # #(batch_size,263,output_dim)->(batch_size,output_dim,263)
        intensity_score = intensity_score.to(device)
        finally_embedding = torch.matmul(finally_embedding,intensity_score)
        # (batch_size,output_dim,263)*(batch_size,263,263)->(batch_size,output_dim,263)
        finally_embedding = finally_embedding.permute(0,2,1).contiguous() # (batch_size,263,output_dim)

        # del finally_embedding = finally_embedding.to(device)
        finally_embedding = self.fc1(finally_embedding) # (batch_size,263,16)
        finally_embedding = self.fc2(finally_embedding) # (batch_size,263,1)

        return finally_embedding

