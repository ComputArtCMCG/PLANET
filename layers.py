from asyncio import coroutines
from venv import create
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,sys
from nnutils import create_var
from nnutils import index_select_ND

class ProteinGAT(nn.Module):
    def __init__(self,feature_dims,nheads,key_dims,value_dims,update_iters,device):
        super().__init__()
        self.nheads = nheads
        self.feature_dims = feature_dims
        self.key_dims = key_dims
        self.value_dims = value_dims
        self.update_iters = update_iters
        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
        self.linear_Q = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_K = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_V = nn.Linear(feature_dims,nheads*value_dims,bias=False)
        self.linear_out = nn.Sequential(nn.Linear(value_dims,feature_dims),nn.LeakyReLU(0.1))
        self.linear_last = nn.Sequential(nn.Linear(2*feature_dims,feature_dims),nn.LeakyReLU(0.1))
        self.device = device

        #self.super_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
        #self.res_att = nn.Sequential(nn.Linear(feature_dims,nheads*feature_dims,bias=False),nn.Tanh())
        #self.linear_att = nn.Linear(feature_dims,1)
        #self.aggregate = nn.Sequential(nn.Linear(nheads*feature_dims,feature_dims),nn.LeakyReLU(0.1))

        #para group
        #self.interaction_modules = [self.res_embedding,self.linear_Q,self.linear_K,self.linear_V,self.linear_out]

    def split(self,q,k,v):
        q = torch.reshape(q,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,0,2]) #[h,n,k]
        k = torch.reshape(k,shape=[self.node_size,self.nheads,self.key_dims]).permute([1,2,0]) #[h,k,n]
        v = torch.reshape(v,shape=[self.node_size,self.nheads,self.value_dims]).permute([1,0,2]) #[h,n,v]
        return q,k,v
    
    def forward(self,fresidues,distance_weight,res_mask):
        fresidues = self.res_embedding(fresidues)
        fresidues =  fresidues + torch.sum(torch.unsqueeze(fresidues,0) * distance_weight,dim=1)
        initial_features = fresidues
        self.node_size = fresidues.shape[0]
        for _ in range(self.update_iters):
            q = self.linear_Q(fresidues)  #[batch,nodes,nheads*key]
            k = self.linear_K(fresidues)
            v = self.linear_V(fresidues)
            #v = fresidues.unsqueeze(0)   #[1,n,f]
            q,k,v = self.split(q,k,v)
            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) 
            attention_matrix = attention_matrix + res_mask #[h,n,n]
            attention_matrix = torch.softmax(attention_matrix,dim=-1) #  * distance_weight #[h,n,n]
            #attention_matrix = (attention_matrix + torch.transpose(attention_matrix,-2,-1)) / 2. #[h,n,n]
            #attention_matrix = attention_matrix / torch.sum(attention_matrix,dim=-1,keepdim=True)  #normalize to 1
            v = torch.matmul(attention_matrix,v).permute([1,0,2]) #[h,n,f] -- [n,h,f]
            fresidues = self.linear_out(torch.sum(v,dim=1)) #[n,f]
        fresidues = self.linear_last(torch.cat([fresidues,initial_features],dim=-1))
        return fresidues

class LigandGAT(nn.Module):
    def __init__(self,feature_dims,nheads,key_dims,value_dims,update_iters,device):
        super().__init__()
        self.nheads = nheads
        self.feature_dims = feature_dims
        self.key_dims = key_dims
        self.value_dims = value_dims
        self.update_iters = update_iters
        self.device = device
        
        self.fbonds_embedding = nn.Linear(31+6,feature_dims)
        self.linear_Q = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_K = nn.Linear(feature_dims,nheads*key_dims,bias=False)
        self.linear_V = nn.Linear(feature_dims,nheads*value_dims,bias=False)
        self.fbonds_out = nn.Linear(value_dims,feature_dims,bias=False)
        self.linear_last = nn.Sequential(nn.Linear(31+feature_dims,feature_dims),nn.LeakyReLU(0.1))
        #para group
        #self.interaction_modules = [self.fbonds_embedding,self.linear_Q,self.linear_K,self.linear_V,self.fbonds_out,self.linear_last,self.linear_lig_interaction]
    

    def split(self,q,k,v):
        q = torch.reshape(q,shape=[self.bonds_size,1,self.nheads,self.key_dims]).permute([0,2,1,3])  #[b,1,h,k] -- #[b,h,1,k]
        k = torch.reshape(k,shape=[self.bonds_size,self.nei_size,self.nheads,self.key_dims]).permute([0,2,3,1])  #[b,h,k,nei]
        v = torch.reshape(v,shape=[self.bonds_size,self.nei_size,self.nheads,self.value_dims]).permute([0,2,1,3])  #[b,h,nei,v]
        return q,k,v

    def forward(self,fatoms,fbonds,agraph,bgraph,lig_scope):
        self.bonds_size = fbonds.shape[0]
        self.nei_size = bgraph.shape[1]
        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)
        fbonds_input = self.fbonds_embedding(fbonds) #[fbonds,feature_dim]
        message = F.leaky_relu(fbonds_input,negative_slope=0.1)
        
        nei_mask = torch.where(bgraph!=0.,0.,-1e7).reshape(self.bonds_size,1,1,self.nei_size)  #[fbonds,1,1,nei]
        for _ in range(self.update_iters-1):
            nei_message = index_select_ND(message,0,bgraph)  #[fbonds,nei,feature_dim]
            q = self.linear_Q(message)  
            k = self.linear_K(nei_message)
            v = self.linear_V(nei_message)
            #v = nei_message.reshape([self.bonds_size,1,self.nei_size,self.feature_dims])
            q,k,v = self.split(q,k,v)
            attention_matrix = torch.matmul(q,k)/ math.sqrt(self.key_dims) + nei_mask   #[b,h,1,nei]
            attention_matrix = torch.softmax(attention_matrix,dim=-1)
            #print(attention_matrix)
            nei_message = torch.matmul(attention_matrix,v).permute([0,2,1,3])  #[b,h,1,v] -- [b,1,h,v]
            nei_message = self.fbonds_out(torch.mean(nei_message,dim=2)).reshape([self.bonds_size,-1])
            message = F.leaky_relu(fbonds_input + nei_message,negative_slope=0.1)
        
        nei_message = torch.sum(index_select_ND(message,0,agraph),dim=1)
        fatoms = self.linear_last(torch.cat([fatoms,nei_message],dim=-1))
        
        return fatoms

class ProLig(nn.Module):
    def __init__(self,feature_dims,nheads,update_iters,device):
        super().__init__()
        self.feature_dims = feature_dims
        self.device = device
        self.prolig_attention = ProteinLigandAttention(feature_dims,nheads,update_iters,device)

        #ligand interaction prediction weight
        self.linear_lig_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims,bias=False),
            nn.LeakyReLU(0.1),
        )
        self.linear_lig_interaction_last = nn.Sequential(
            nn.Linear(feature_dims,1,bias=False),
            nn.Sigmoid(),
        )
        
        self.linear_pocket_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
        )
        self.linear_ligand_interaction = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
        )
        
        self.pro_lig_interaction = nn.Sequential(
            nn.Linear(feature_dims,1,bias=False),
            nn.Sigmoid(),
        )
        
        self.linear_pocket_affinity = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
        )
        self.linear_ligand_affinity = nn.Sequential(
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
        )
       
        self.linear_affinity = nn.Sequential(
            nn.Linear(feature_dims,1,bias=False),
            nn.LeakyReLU(0.1),
        )

    def forward(self,fresidues,fatoms,res_scope,lig_scope):
        fatoms,fresidues = self.prolig_attention(fatoms,fresidues,lig_scope,res_scope)
        predicted_lig_interactions,predicted_interactions,predicted_affinities = [],[],[]
        for ((start_res,res_count),(start_atom,atom_count)) in zip(res_scope,lig_scope):
            
            #shape = [atom_count,res_count,self.feature_dims]
            pocket_features = torch.unsqueeze(fresidues[start_res:start_res+res_count],dim=0) #[1,res,features]
            ligand_features = torch.unsqueeze(fatoms[start_atom:start_atom+atom_count],dim=1) #[atoms,1,features]
           # pocket_features = torch.unsqueeze(fresidues[start_res:start_res+res_count],dim=0) #[1,res,features]
           # ligand_features = torch.unsqueeze(fatoms[start_atom:start_atom+atom_count],dim=1) #[atoms,1,features]
            
            predicted_lig_interaction = self.linear_lig_interaction_last(torch.multiply(
                self.linear_lig_interaction(ligand_features).unsqueeze(1), \
                self.linear_lig_interaction(ligand_features).unsqueeze(0)
                ))

            predicted_lig_interactions.append(predicted_lig_interaction.reshape([-1]))

            predicted_interaction = self.pro_lig_interaction(torch.multiply(
                self.linear_ligand_interaction(ligand_features),
                self.linear_pocket_interaction(pocket_features)
                ))  #[atom,res,1]
            predicted_interactions.append(torch.reshape(predicted_interaction,shape=[-1]))

            complex_features = torch.multiply(
                self.linear_ligand_affinity(ligand_features),
                self.linear_pocket_affinity(pocket_features)
            )

            predict_affinity = torch.sum(self.linear_affinity(complex_features) * predicted_interaction).reshape([1])

            predicted_affinities.append(predict_affinity)
        
        predicted_lig_interactions = torch.cat(predicted_lig_interactions)
        predicted_interactions = torch.cat(predicted_interactions)
        #print(predicted_interactions)
        predicted_affinities = torch.cat(predicted_affinities)
        return predicted_lig_interactions,predicted_interactions,predicted_affinities
    
    def _forward(self,fresidues,fatoms,res_scope,lig_scope):
        fatoms,fresidues = self.prolig_attention(fatoms,fresidues,lig_scope,res_scope)
        predicted_affinities = []
        for ((start_res,res_count),(start_atom,atom_count)) in zip(res_scope,lig_scope):
            
            #shape = [atom_count,res_count,self.feature_dims]
            pocket_features = torch.unsqueeze(fresidues[start_res:start_res+res_count],dim=0) #[1,res,features]
            ligand_features = torch.unsqueeze(fatoms[start_atom:start_atom+atom_count],dim=1) #[atoms,1,features]
           # pocket_features = torch.unsqueeze(fresidues[start_res:start_res+res_count],dim=0) #[1,res,features]
           # ligand_features = torch.unsqueeze(fatoms[start_atom:start_atom+atom_count],dim=1) #[atoms,1,features]
            

            predicted_interaction = self.pro_lig_interaction(torch.multiply(
                self.linear_ligand_interaction(ligand_features),
                self.linear_pocket_interaction(pocket_features)
                ))  #[atom,res,1]

            complex_features = torch.multiply(
                self.linear_ligand_affinity(ligand_features),
                self.linear_pocket_affinity(pocket_features)
            )

            predict_affinity = torch.sum(self.linear_affinity(complex_features) * predicted_interaction).reshape([1])

            predicted_affinities.append(predict_affinity)
        
        predicted_affinities = torch.cat(predicted_affinities)
        return predicted_affinities

class ProteinEGNN(nn.Module):
    def __init__(self,feature_dims,update_iters,device):
        super().__init__()
        self.update_iters = update_iters
        self.feature_dims = feature_dims
        self.device = device
        self.res_embedding = nn.Sequential(nn.Linear(20,feature_dims,bias=False),nn.LeakyReLU(0.1))
        self.linear_e = nn.Sequential(
            nn.Linear(2*feature_dims+1,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims),
            nn.LeakyReLU(0.1)
        )
        self.linear_inf = nn.Sequential(
            nn.Linear(feature_dims,1),
            nn.Sigmoid()
        )
        
        self.linear_h = nn.Sequential(
            nn.Linear(2*feature_dims,feature_dims),
            nn.LeakyReLU(0.1),
            nn.Linear(feature_dims,feature_dims)
        )

    def forward(self,fresidues,coordinates,res_scope):
        '''
        coordinates [n,3]
        res_map [n,n,1]
        '''
        fresidues = self.res_embedding(fresidues)
        
        for _ in range(self.update_iters):
            fresidues_new = []
            for (start_res,res_count) in res_scope:
                fresidues_batch = fresidues[start_res:start_res+res_count] 
                coordinates_batch = coordinates[start_res:start_res+res_count] 
                broadcast_shape = [fresidues_batch.shape[0],fresidues_batch.shape[0],self.feature_dims]
                coor_diff = coordinates_batch.unsqueeze(1) - coordinates_batch.unsqueeze(0)
                m_input = torch.cat([
                    fresidues_batch.unsqueeze(1).broadcast_to(broadcast_shape),
                    fresidues_batch.unsqueeze(0).broadcast_to(broadcast_shape),
                    torch.sum(torch.square(coor_diff),dim=-1,keepdim=True)
                ],dim=-1)
                m = self.linear_e(m_input)
                #coordinates_batch = coordinates_batch + C * torch.sum(coor_diff*self.linear_x(m),dim=1)
                e = self.linear_inf(m)
                mask = torch.ones([res_count,res_count]) - torch.eye(res_count)
                mask = mask.unsqueeze(-1).to(self.device)
                e = e * mask
                m_pre = torch.sum(m*e,dim=1)
                fresidues_batch = self.linear_h(torch.cat([fresidues_batch,m_pre],dim=-1)) + fresidues_batch
                fresidues_new.append(fresidues_batch)
                #coordinates_new.append(coordinates_batch)
            fresidues = torch.cat(fresidues_new,dim=0)
            #coordinates = torch.cat(coordinates_new,dim=0)
        return fresidues

    ###for speeding screening
    def pre_cal_res_features(self,fresidues,coordinates):
        fresidues = self.res_embedding(fresidues)
        for _ in range(self.update_iters):
            broadcast_shape = [fresidues.shape[0],fresidues.shape[0],self.feature_dims]
            coor_diff = coordinates.unsqueeze(1) - coordinates.unsqueeze(0)
            m_input = torch.cat([
                fresidues.unsqueeze(1).broadcast_to(broadcast_shape),
                fresidues.unsqueeze(0).broadcast_to(broadcast_shape),
                torch.sum(torch.square(coor_diff),dim=-1,keepdim=True)
            ],dim=-1)
            m = self.linear_e(m_input)
            e = self.linear_inf(m)
            mask = torch.ones([fresidues.shape[0],fresidues.shape[0]]) - torch.eye(fresidues.shape[0])
            mask = mask.unsqueeze(-1).to(self.device)
            e = e * mask
            m_pre = torch.sum(m*e,dim=1)
            fresidues = self.linear_h(torch.cat([fresidues,m_pre],dim=-1)) + fresidues
        return fresidues

            
class ProteinLigandAttention(nn.Module):
    def __init__(self,feature_dims,nheads,update_iters,device):
        super().__init__()
        self.feature_dims = feature_dims
        self.nheads = nheads
        self.update_iters = update_iters
        self.device = device
        self.lig_transform = nn.Sequential(nn.Linear(feature_dims,feature_dims,bias=False),nn.LeakyReLU(0.1))
        self.res_transform = nn.Sequential(nn.Linear(feature_dims,feature_dims,bias=False),nn.LeakyReLU(0.1))

        self.lig_Q = nn.Linear(feature_dims,feature_dims,bias=False)
        self.lig_K = nn.Linear(feature_dims,feature_dims,bias=False)
        self.lig_V = nn.Linear(feature_dims,feature_dims,bias=False)
        self.lig_out = nn.Sequential(nn.Linear(feature_dims,feature_dims,bias=False),nn.LeakyReLU(0.1))

        self.res_Q = nn.Linear(feature_dims,feature_dims,bias=False)
        self.res_K = nn.Linear(feature_dims,feature_dims,bias=False)
        self.res_V = nn.Linear(feature_dims,feature_dims,bias=False)
        self.res_out = nn.Sequential(nn.Linear(feature_dims,feature_dims,bias=False),nn.LeakyReLU(0.1))

    def split_update(self,q,k,v):
        q = torch.reshape(q,shape=[-1,self.feature_dims])  #[b,f] 
        k = torch.reshape(k,shape=[-1,self.feature_dims]).permute([1,0])  #[f,a]
        v = torch.reshape(v,shape=[-1,self.feature_dims])  #[a,f]
        attention_matrix = torch.matmul(q,k)   #[b,a]
        attention_matrix = torch.sigmoid(attention_matrix)
        update_message = torch.matmul(attention_matrix,v) #[b,f]
        return update_message

    def forward(self,fatoms,fresidues,lig_scope,res_scope):
        fatoms = self.lig_transform(fatoms)   #[n,f]
        fresidues = self.res_transform(fresidues)  #[m,f]

        for _ in range(self.update_iters):
            fresidues_new,fatoms_new = [],[]
            for ((start_res,res_count),(start_atom,atom_count)) in zip(res_scope,lig_scope):
                fatoms_batch = fatoms[start_atom:start_atom+atom_count]
                fresidues_batch = fresidues[start_res:start_res+res_count]
                atoms_q = self.lig_Q(fatoms_batch)
                atom_k = self.lig_K(fatoms_batch)
                atom_v = self.lig_V(fatoms_batch)
                res_q = self.res_Q(fresidues_batch)
                res_k = self.res_K(fresidues_batch)
                res_v = self.res_V(fresidues_batch)

                fatoms_batch = self.lig_out(self.split_update(atoms_q,res_k,res_v)) / res_count + fatoms_batch
                fresidues_batch = self.res_out(self.split_update(res_q,atom_k,atom_v)) / atom_count + fresidues_batch
                
                fatoms_new.append(fatoms_batch)
                fresidues_new.append(fresidues_batch)
            
            fatoms = torch.cat(fatoms_new,dim=0)
            fresidues = torch.cat(fresidues_new,dim=0)
        return fatoms,fresidues







