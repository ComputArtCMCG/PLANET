import os,argparse,pickle,random,json
from rdkit import Chem
from rdkit.RDLogger import ERROR
from chemutils import ComplexPocket
from multiprocessing import Pool

def process_PDBBind(record_dir):
    pdb_name = record_dir.split('/')[-1]
    #print('Processing {}'.format(pdb_name))
    ligand_sdf = os.path.join(record_dir,'{}_ligand.sdf'.format(pdb_name))
    protein_pdb = os.path.join(record_dir,'{}_protein.pdb'.format(pdb_name))
    decoy_sdf = os.path.join(record_dir,'{}_decoy.sdf'.format(pdb_name))
    pK_data_path = '/disk1/aquila/SBDD_model_dock/data/PDBbind2020.json'
    with open(pK_data_path,'rb') as f:
        pK_data = json.load(f)
    try:
        pK=pK_data[pdb_name]
    except KeyError:
        pK=0
    try:
        pocket = ComplexPocket(protein_pdb,ligand_sdf,pK,decoy_sdf)
        pocket_pkl = os.path.join(record_dir,'{}_pocket.pkl'.format(pdb_name))
        with open(pocket_pkl,'wb') as f:
            pickle.dump(pocket, f, pickle.HIGHEST_PROTOCOL) 
    except Exception as e:
        pass
        #print(pdb_name)
        #os.system('cp -r /disk1/aquila/PDBbind2020_all/origin/{}/ /disk1/aquila/PDBbind2020_repair/'.format(pdb_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--dir',required=True)
    parser.add_argument('-n','--njobs',required=True,type=int)
    args = parser.parse_args()
    print(args)
    

    base_dir = args.dir
    njobs = args.njobs

    pool = Pool(njobs)

    all_records = [os.path.join(base_dir,subdir) for subdir in os.listdir(base_dir)]
    pool.map(process_PDBBind,all_records)   
