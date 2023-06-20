import math,os,sys,torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
from layers import ProteinEGNN,LigandGAT,ProLig
from nnutils import create_var, create_var_gpu

class PLANET(nn.Module):
    def __init__(self,feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,device):
        super().__init__()
        self.device = torch.device(device)
        self.feature_dims = feature_dims
        self.proteinegnn = ProteinEGNN(feature_dims,pro_update_inters,self.device)
        self.ligandgat = LigandGAT(feature_dims,nheads,key_dims,value_dims,lig_update_iters,self.device)
        self.prolig = ProLig(feature_dims,nheads,pro_lig_update_iters,self.device)

        self.ligand_interaction_loss = nn.BCELoss(reduction='sum',)
        self.pro_lig_interaction_loss = nn.BCELoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='none')

    ###used for model training 
    def forward(self,res_batch,mol_batch):
        (fresidues,res_map,res_scope,alpha_coordinates) = res_batch
        fresidues = fresidues.to(self.device)
        #res_map = create_var(res_map)
        alpha_coordinates = create_var(alpha_coordinates)
        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch
        fatoms = fatoms.to(self.device)
        fbonds = fbonds.to(self.device)
        agraph = agraph.to(self.device)
        bgraph = bgraph.to(self.device)

        #distance_weight,res_mask = self.distance_network(fresidues,alpha_coordinates,res_scope,res_map)
        fresidues = self.proteinegnn(fresidues,alpha_coordinates,res_scope)
        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
         
        predicted_lig_interactions,predicted_interactions,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
        return (predicted_lig_interactions,predicted_interactions,predicted_affinities)
    
    def compute_loss(self,predictions,targets,res_batch,mol_batch):
        predicted_lig_interactions,predicted_interactions,predicted_affinities = predictions
        (_,_,res_scope,_) = res_batch
        (_,_,_,_,lig_scope) = mol_batch

        mol_intearctions,pro_lig_interactions,pKs,pK_flags,complex_labels = targets
        mol_intearctions = create_var(mol_intearctions)
        pro_lig_interactions = create_var(pro_lig_interactions)
        pKs = create_var(pKs)
        pK_flags = create_var(pK_flags)

        lig_interaction_count,pro_lig_interaction_count = 0,0
        lig_interaction_loss,pro_lig_interaction_loss = [],[]
        for ((_,res_count),(_,atom_count),complex_label) in zip(res_scope,lig_scope,complex_labels):
            l_loss = self.ligand_interaction_loss(
                predicted_lig_interactions[lig_interaction_count:lig_interaction_count+atom_count**2],
                mol_intearctions[lig_interaction_count:lig_interaction_count+atom_count**2]
                ) / atom_count
            pl_loss = self.pro_lig_interaction_loss(
                predicted_interactions[pro_lig_interaction_count:pro_lig_interaction_count+atom_count*res_count],
                pro_lig_interactions[pro_lig_interaction_count:pro_lig_interaction_count+atom_count*res_count]
                ) / atom_count
            if complex_label == 1:
                lig_interaction_loss.append(l_loss)
            pro_lig_interaction_loss.append(pl_loss)
            lig_interaction_count += atom_count**2
            pro_lig_interaction_count += atom_count*res_count
        lig_interaction_loss = torch.mean(torch.stack(lig_interaction_loss))
        pro_lig_interaction_loss = torch.mean(torch.stack(pro_lig_interaction_loss))
        affinity_loss = torch.sum(self.mse_loss(predicted_affinities,pKs) * pK_flags) / torch.sum(pK_flags)
        return lig_interaction_loss,pro_lig_interaction_loss,affinity_loss

    @staticmethod
    def compute_metrics(predictions,targets):
        predicted_lig_interactions,predicted_interactions,predicted_affinities = predictions
        mol_intearctions,pro_lig_interactions,pKs,pK_flags,_ = targets
        mol_intearctions = create_var(mol_intearctions)
        pro_lig_interactions = create_var(pro_lig_interactions)
        pKs = create_var(pKs)
        pK_flags = create_var(pK_flags)

        lig_interaction_result = torch.where(predicted_lig_interactions>0.5,1.,0.)
        lig_interaction_acc = torch.mean(torch.where(lig_interaction_result==mol_intearctions,1.,0.)) 

        pro_lig_interaction_result = torch.where(predicted_interactions>0.5,1.,0.)
        pro_lig_interaction_acc = torch.mean(torch.where(pro_lig_interaction_result==pro_lig_interactions,1.,0.))

        affinity_mae = torch.sum(torch.abs(predicted_affinities-pKs)* pK_flags) / torch.sum(pK_flags)
        
        return lig_interaction_acc,pro_lig_interaction_acc,affinity_mae

    def load_parameters(self,parameters=os.path.join(os.path.dirname(os.path.abspath(__file__)),'PLANET.param')):
        self.load_state_dict(torch.load(parameters,map_location=self.device))
        
    ###used for screening (only one protein, different mols)
    def cal_res_features_helper(self,fresidues,coordinates):
        fresidues = fresidues.to(self.device)
        coordinates = coordinates.to(self.device)
        fresidues = self.proteinegnn.pre_cal_res_features(fresidues,coordinates)
        return fresidues

    def cal_res_features(self,fresidues,batch_size):
        res_scope = [(i*int(fresidues.shape[0]),int(fresidues.shape[0])) for i in range(0,batch_size)]
        fresidues = fresidues.repeat(batch_size,1)
        return fresidues,res_scope

    def screening(self,fresidues,res_scope,mol_feature_batch):
        (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_feature_batch
        fatoms = self.ligandgat(fatoms,fbonds,agraph,bgraph,lig_scope)
        _,_,predicted_affinities = self.prolig(fresidues,fatoms,res_scope,lig_scope)
        return predicted_affinities
