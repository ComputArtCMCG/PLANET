from typing import List
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os,argparse,pickle,sys,random
from chemutils import ComplexPocket,tensorize_all,role_of_5
import numpy as np
from rdkit import Chem
import rdkit
from itertools import chain

class ProLigDataset(Dataset):
    def __init__(self,data_pickle_path,batch_size,shuffle=True,decoy_flag=True):
        with open(data_pickle_path,'rb') as f:
            data = pickle.load(f)
        flag = False
        if shuffle:
            while not flag:  
                random.shuffle(data)
                data_new = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
                flag = self.check(data_new)
            data = data_new
        else:
            data = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        self.decoy_flag = decoy_flag

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.tensorize(idx)

    def check(self,data):  #check each batch has at least one record with pK value (that is from original PDBbind)
        for batch in data:
            pKs = np.array([float(record[1]) for record in batch])
            if np.sum(pKs)==0:
                return False
        return True

    def tensorize(self,idx):
        pocket_batch = []
        for record in self.data[idx]:
            with open(record[0],'rb') as f:
                pocket:ComplexPocket = pickle.load(f)
                pocket_batch.append(pocket)    
        res_feature_batch,mol_feature_batch,mol_intearctions,pro_lig_interactions,pKs,pK_flags,complex_labels = tensorize_all(pocket_batch,self.decoy_flag)
        return res_feature_batch,mol_feature_batch,(mol_intearctions,pro_lig_interactions,pKs,pK_flags,complex_labels)

    def get_bonded_atom_pairs(self) -> List[List[tuple]]:
        bonded_pairs = []
        for record in chain(*self.data):
            with open(record[0],'rb') as f:
                pocket:ComplexPocket = pickle.load(f)
                bonded_pairs.append(pocket.ligand.get_bonded_atoms())
        return bonded_pairs
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir',required=True)
    parser.add_argument('-p','--pdbdir',required=True)

    args = parser.parse_args()
    base_dir = args.dir
    pdb_dir = args.pdbdir
    
    train_csv = os.path.join(base_dir,'TrainSet.csv')
    valid_csv = os.path.join(base_dir,'ValidSet.csv')
    #test_csv = os.path.join(base_dir,'TestSet.csv')
    core_csv = os.path.join(base_dir,'CoreSet.csv')
    for (csv,_type_) in zip([train_csv,valid_csv,core_csv],['train','valid','core']):
        dataframe = pd.read_csv(csv,index_col=0)
        pdb_codes = dataframe['PDB_code']
        pKs = dataframe['pK'].astype(float)
        record_types =  dataframe['type']
        dataset = []
        for (pdb_code,record_type,pK) in zip(pdb_codes,record_types,pKs):
            if str(record_type) == 'UNKNOWN':
                sub_folder = 'extend' 
            else:
                sub_folder = 'origin'
            if os.path.exists(os.path.join(pdb_dir,sub_folder,pdb_code,'{}_pocket.pkl'.format(pdb_code))):
                with open(os.path.join(pdb_dir,sub_folder,pdb_code,'{}_pocket.pkl'.format(pdb_code)),'rb') as f:
                    pocket = pickle.load(f)
                    #if pocket.distance_matrix.shape[0] == 0:
                        #print(pdb_code)
                        #os.system('cp -r /disk1/aquila/PDBbind2020/{}/ /disk1/aquila/test_PDBbind'.format(pdb_code))
                    ###for PDBbind_extend which is not checked,some ligands may not in a 'pocket'
                if sub_folder == 'extend' and len(pocket.pocket_residues) >= 30:
                    try:
                        ligand_mol = pocket.ligand.mol
                        Chem.SanitizeMol(ligand_mol)
                        #if role_of_5(ligand_mol):
                        dataset.append([os.path.join(pdb_dir,sub_folder,pdb_code,'{}_pocket.pkl'.format(pdb_code)),pK])
                    except:
                        continue
                elif sub_folder == 'extend' and len(pocket.pocket_residues) < 30:
                    continue
                else:
                    dataset.append([os.path.join(pdb_dir,sub_folder,pdb_code,'{}_pocket.pkl'.format(pdb_code)),pK])
            else:
                continue
        with open(os.path.join(base_dir,'{}.pkl'.format(_type_)),'wb') as out_f:
            pickle.dump(dataset, out_f, pickle.HIGHEST_PROTOCOL)
        print('{} has total records:{}'.format(_type_,str(len(dataset))))

