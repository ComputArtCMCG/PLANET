import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PLANET_model import PLANET
from PLANET_datautils import ProLigDataset
import numpy as np
import scipy.stats as stats
import argparse,math,rdkit,os,pickle

def test_PLANET(PLANET,test_pickle):
    test_dataset = ProLigDataset(test_pickle,batch_size=16,shuffle=False,decoy_flag=False)
    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4,drop_last=False,collate_fn=lambda x:x[0])
    bonded_pairs = test_dataset.get_bonded_atom_pairs()
    PLANET.eval()
    predicted_lig_interactions,predicted_interactions,predicted_affinities,lig_scopes,res_scopes= [],[],[],[],[]
    ligand_interactions,pro_lig_interactions,pKs = [],[],[]
    
    with torch.no_grad():
        for (res_feature_batch,mol_feature_batch,targets) in test_loader:
            try:
                (predicted_lig_interaction,predicted_interaction,predicted_affinity) = PLANET(res_feature_batch,mol_feature_batch)
                ligand_interaction,pro_lig_interaction,pK,_,_ = targets

                predicted_lig_interactions.append(np.array(predicted_lig_interaction.squeeze().detach().cpu()))
                predicted_interactions.append(np.array(predicted_interaction.squeeze().detach().cpu()))   
                predicted_affinities.append(np.array(predicted_affinity.squeeze().detach().cpu()))
                lig_scopes.extend(mol_feature_batch[4])
                res_scopes.extend(res_feature_batch[2])
                
                ligand_interactions.append(np.array(ligand_interaction.squeeze().detach().cpu()))
                pro_lig_interactions.append(np.array(pro_lig_interaction.squeeze().detach().cpu()))
                pKs.append(np.array(pK.squeeze().detach().cpu()))

            except Exception as e:
                print (e)
                continue
    predicted_lig_interactions = np.concatenate(predicted_lig_interactions,axis=0)
    predicted_interactions = np.concatenate(predicted_interactions,axis=0)
    predicted_affinities = np.concatenate(predicted_affinities,axis=0)

    ligand_interactions = np.concatenate(ligand_interactions,axis=0)
    pro_lig_interactions = np.concatenate(pro_lig_interactions,axis=0)
    pKs = np.concatenate(pKs,axis=0)
    
    MAE = np.mean(np.abs(predicted_affinities-pKs))
    RMSE = np.sqrt(np.sum(np.square(predicted_affinities-pKs)) / len(predicted_affinities))
    P_correlation,P_pvalue = stats.stats.pearsonr(predicted_affinities,pKs)
    S_correlation,S_pvalue =  stats.spearmanr(predicted_affinities,pKs)
    print('MAE:{:3f}\tRMSE:{:3f}\tPearson:{:3f}\tP_pvalue:{:3f}\tSpearman:{:3f}\tS_pvalue:{:3f}'.format(MAE,RMSE,P_correlation,P_pvalue,S_correlation,S_pvalue))

    return predicted_lig_interactions,predicted_interactions,predicted_affinities, \
            ligand_interactions,pro_lig_interactions,pKs,lig_scopes,res_scopes,bonded_pairs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--PLANET_file',required=True)
    parser.add_argument('-t','--test',required=True)
    parser.add_argument('-o','--out_path',required=True)
    
    parser.add_argument('-s','--feature_dims', type=int, default=300)
    parser.add_argument('-n','--nheads', type=int, default=8)
    parser.add_argument('-k','--key_dims', type=int, default=300)
    parser.add_argument('-va','--value_dims', type=int, default=300)

    parser.add_argument('-pu','--pro_update_inters', type=int, default=3)
    parser.add_argument('-lu','--lig_update_iters',type=int,default=10)
    parser.add_argument('-pl','--pro_lig_update_iters',type=int,default=1)
    args = parser.parse_args()

    feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters = args.feature_dims,args.nheads,args.key_dims,args.value_dims,\
        args.pro_update_inters,args.lig_update_iters,args.pro_lig_update_iters
    PLANET = PLANET(feature_dims,nheads,key_dims,value_dims,pro_update_inters,lig_update_iters,pro_lig_update_iters,'cuda').cuda()
    PLANET.load_state_dict(torch.load(args.PLANET_file))
    
    predicted_lig_interactions,predicted_interactions,predicted_affinities,\
        ligand_interactions,pro_lig_interactions,pKs,lig_scopes,res_scopes,bonded_pairs = test_PLANET(PLANET,args.test)
    with open(args.out_path,'wb') as pickle_out:
        out_data = [predicted_lig_interactions,predicted_interactions,predicted_affinities,\
            ligand_interactions,pro_lig_interactions,pKs,lig_scopes,res_scopes,bonded_pairs]
        pickle.dump(out_data,pickle_out,protocol=pickle.HIGHEST_PROTOCOL)
