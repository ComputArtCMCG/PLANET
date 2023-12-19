import rdkit.Chem as Chem
import torch,sys,os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PLANET_model import PLANET
from chemutils import ProteinPocket,mol_batch_to_graph

class PlanetEstimator():
    def __init__(self,device):
        self.model = PLANET(300,8,300,300,3,10,1,device=device)  #trained PLANET
        self.model.load_parameters()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def set_pocket_from_ligand(self,protein_pdb,ligand_sdf):
        try:
            self.pocket = ProteinPocket(protein_pdb=protein_pdb,ligand_sdf=ligand_sdf)
        except:
            raise RuntimeError('the protein pdb file need to be fixed')
        self.res_features = self.model.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

    def set_pocket_from_coordinate(self,protein_pdb,centeriod_x,centeriod_y,centeriod_z):
        try:
            self.pocket = ProteinPocket(protein_pdb,centeriod_x,centeriod_y,centeriod_z)
        except:
            raise RuntimeError('the protein pdb file need to be fixed')
        self.res_features = self.model.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

    def pre_cal_res_features(self):
        self.res_features = self.model.cal_res_features_helper(self.pocket.res_features,self.pocket.alpha_coordinates)

class VS_SDF_Dataset(Dataset):
    def __init__(self,sdf_file,batch_size=32):
        self.batch_size = batch_size
        self.sdf_supp = Chem.SDMolSupplier(sdf_file,removeHs=False,sanitize=True)
        self.data_index = self.mol_index_from_sdf()

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self,idx):
        return self.tensorize(idx)

    def tensorize(self,idx):
        try:
            mol_batch_idx = self.data_index[idx]
            mol_batch = [self.sdf_supp[i] for i in mol_batch_idx]
            mol_names = [mol.GetProp('_Name') for mol in mol_batch if mol is not None]
            mol_batch = [Chem.AddHs(mol) for mol in mol_batch if mol is not None]
            mol_feature_batch = mol_batch_to_graph(mol_batch)
            mol_smiles = [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in mol_batch if mol is not None]
            return (mol_feature_batch,mol_smiles,mol_names)
        except:
            return (None,None,None)

    def mol_index_from_sdf(self):
        index_list = []
        for i,mol in enumerate(self.sdf_supp):
            if mol is not None:
                index_list.append(i)

        index_list = [index_list[i : i + self.batch_size] for i in range(0, len(index_list), self.batch_size)]
        if len(index_list) >=2  and len(index_list[-1]) <= 5:
            last = index_list.pop()
            index_list[-1].extend(last)
        return index_list

class VS_SMI_Dataset(Dataset):
    def __init__(self,smi_file,batch_size=32):
        self.batch_size = batch_size
        self.contents = self.read_smi(smi_file)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self,idx):
        return self.tensorize(idx)

    def tensorize(self,idx):
        mol_batch_contents = self.contents[idx]
        mol_batch_contents = [(Chem.AddHs(Chem.MolFromSmiles(smi)),smi,name) for (smi,name) in mol_batch_contents if Chem.MolFromSmiles(smi,sanitize=True) is not None]
        mol_feature_batch = mol_batch_to_graph([content[0] for content in mol_batch_contents],auto_detect=False)
        mol_smiles = [content[1] for content in mol_batch_contents]
        mol_names = [content[2] for content in mol_batch_contents]
        return (mol_feature_batch,mol_smiles,mol_names)

    def read_smi(self,smi_file):
        with open(smi_file,'r') as f:
            contents = [line.strip() for line in f]
        try:    
            contents = [(line.split()[0],line.split()[1]) for line in contents]
        except IndexError:
            contents = [(line.split()[0],"UNKOWN") for line in contents]
        contents = [contents[i:i+self.batch_size] for i in range(0,len(contents),self.batch_size)]
        return contents

def workflow(protein_pdb,mol_file,ligand_sdf=None,centeriod_x=None,centeriod_y=None,centeriod_z=None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    estimator = PlanetEstimator(device)
    estimator.model.to(device)
    if ligand_sdf is not None:
        estimator.set_pocket_from_ligand(protein_pdb,ligand_sdf)
    elif centeriod_x is not None and centeriod_y is not None and centeriod_z is not None:
        estimator.set_pocket_from_coordinate(protein_pdb,centeriod_x,centeriod_y,centeriod_z)
    else:
        sys.exit()
    suffix = os.path.basename(mol_file).split('.')[-1]
    if suffix == 'smi':
        dataset = VS_SMI_Dataset(mol_file)
    elif suffix == 'sdf':
        dataset = VS_SDF_Dataset(mol_file)
    else:
        raise NotImplementedError("mol file input formats besides smi and sdf are not supported")
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=2,drop_last=False,collate_fn=lambda x:x[0])
    predicted_affinities,mol_names,smis = [],[],[]
    with torch.no_grad():
        for (mol_feature_batch,smi_batch,mol_name) in dataloader:
            try:
                batch_size = len(smi_batch)
                fresidues_batch,res_scope = estimator.model.cal_res_features(estimator.res_features,batch_size)
                predicted_affinity = estimator.model.screening(fresidues_batch,res_scope,mol_feature_batch)
                predicted_affinities.append((predicted_affinity.view([-1]).cpu().numpy()))
                smis.extend(smi_batch)
                mol_names.extend(mol_name)
            except:
                continue
    predicted_affinities = np.concatenate(predicted_affinities)
    return predicted_affinities,mol_names,smis

def result_to_csv_sdf(predicted_affinities,mol_names,smis,prefix=None):
    if not prefix:
        prefix = 'result'
    out_sdf = prefix+'.sdf'
    out_csv = prefix+'.csv'
    writer = Chem.SDWriter(out_sdf)
    writer.SetProps(['PLANET_affinity'])
    for aff,name,smi in zip(predicted_affinities,mol_names,smis):
        try:
            mol = Chem.MolFromSmiles(smi)
            mol.SetProp('PLANET_affinity', '{:.3f}'.format(aff))
            mol.SetProp('_Name',name)
            writer.write(mol) 
        except:
            continue
    writer.close()
    csv_frame = pd.DataFrame([
        {
            'mol_name':name,'SMILES':smi,'PLANET_affinity':aff,
        }
        for aff,name,smi in zip(predicted_affinities,mol_names,smis)
    ])
    csv_frame.to_csv(out_csv)

if __name__ == "__main__":
    from rdkit import RDLogger
    import argparse
    RDLogger.DisableLog('rdApp.*')
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--protein',required=True)
    parser.add_argument('-l','--ligand',default=None)
    parser.add_argument('-x','--center_x',default=None,type=float)
    parser.add_argument('-y','--center_y',default=None,type=float)
    parser.add_argument('-z','--center_z',default=None,type=float)
    parser.add_argument('-m','--mol_file',required=True)
    parser.add_argument('--prefix',required=False)

    args = parser.parse_args()
    protein_pdb,mol_file,ligand_sdf,centeriod_x,centeriod_y,centeriod_z = args.protein,args.mol_file,args.ligand,args.center_x,args.center_y,args.center_z
    predicted_affinities,mol_names,smis = workflow(protein_pdb,mol_file,ligand_sdf,centeriod_x,centeriod_y,centeriod_z)
    result_to_csv_sdf(predicted_affinities,mol_names,smis,args.prefix)
    
