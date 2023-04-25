from operator import le
from numpy.core.fromnumeric import shape
import pandas as pd
import rdkit,os
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.QED import qed
from scipy.sparse import coo, csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
#from vocab import Vocab
import numpy as np
import torch
from typing import List,Tuple

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000
periodic_table = Chem.GetPeriodicTable()
BLOSUM62 = {
    'ALA':np.array([[ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0]],dtype=np.float32),
    'ARG':np.array([[-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3]],dtype=np.float32),
    'ASN':np.array([[-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3]],dtype=np.float32),
    'ASP':np.array([[-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3]],dtype=np.float32),
    'CYS':np.array([[ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1]],dtype=np.float32),
    'GLN':np.array([[-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2]],dtype=np.float32),
    'GLU':np.array([[-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'GLY':np.array([[ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3]],dtype=np.float32),
    'HIS':np.array([[-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3]],dtype=np.float32),
    'ILE':np.array([[-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3]],dtype=np.float32),
    'LEU':np.array([[-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1]],dtype=np.float32),
    'LYS':np.array([[-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2]],dtype=np.float32),
    'MET':np.array([[-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1]],dtype=np.float32),
    'PHE':np.array([[-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1]],dtype=np.float32),
    'PRO':np.array([[-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2]],dtype=np.float32),
    'SER':np.array([[ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2]],dtype=np.float32),
    'THR':np.array([[ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0]],dtype=np.float32),
    'TRP':np.array([[-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3]],dtype=np.float32),
    'TYR':np.array([[-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1]],dtype=np.float32),
    'VAL':np.array([[ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]],dtype=np.float32)
}
PROTEIN_ELEMENTS = ['H','C','N','O','S','P','Se']
LIGADND_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'H', 'unknown']

ATOM_FDIM = len(LIGADND_ELEMENTS) + 6 + 5 + 4 + 4 + 1  #31
BOND_FDIM = 6  
MAX_NB = 10

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x==item) for item in allowable_set]

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), LIGADND_ELEMENTS) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + onek_encoding_unk(str(atom.GetHybridization()),['SP','SP2','SP3','other'])
            + [atom.GetIsAromatic()])

def bond_features(bond):
    bt = bond.GetBondType()
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
        bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def near_pocket(pdb_line,centeroid):
    coordinates = pdb_line[30:55]
    x = float(coordinates[0:8])
    y = float(coordinates[8:16])
    z = float(coordinates[16:24])
    distance = np.sqrt(np.sum(np.square(np.array([x,y,z],dtype=np.float32)-centeroid)))
    if distance > 12.0 :
        return False
    else:
        return True

def mass_center_from_pdb(pdb_lines):
    pdb_lines = [line for line in pdb_lines if line[76:78].strip().capitalize() in PROTEIN_ELEMENTS]
    atoms_mass = np.reshape(np.array([periodic_table.GetAtomicWeight(line[76:78].strip().capitalize()) for line in pdb_lines]),[-1,1])
    coordinates = np.array([
        [float(line[30:38]),float(line[38:46]),float(line[46:54])] 
        for line in pdb_lines],dtype=np.float32)
    mass_center = np.sum(coordinates * atoms_mass,axis=0,keepdims=True) / np.sum(atoms_mass)
    alpha_position =  np.array([
        [float(line[30:38]),float(line[38:46]),float(line[46:54])] for line in pdb_lines if line[12:16].strip()=="CA"],dtype=np.float32)

    return mass_center,coordinates,alpha_position

class ProteinPocket():
    def __init__(self,protein_pdb,centeriod_x=None,centeriod_y=None,centeriod_z=None,ligand_sdf=None):
        if ligand_sdf is not None:
            ligand = Mol(Chem.SDMolSupplier(ligand_sdf,sanitize=False)[0])
            centeriod = ligand.compute_centeroid()
        elif centeriod_x is not None and centeriod_y is not None and centeriod_z is not None:
            centeriod = np.array([centeriod_x,centeriod_y,centeriod_z],dtype=np.float32)
        self.determine_pocket_residues(protein_pdb,centeriod)
        self.res_features = torch.from_numpy(np.concatenate([residue.get_feature() for residue in self.pocket_residues],axis=0))
        self.alpha_coordinates = torch.from_numpy(np.concatenate([residue.get_alpha_position() for residue in self.pocket_residues],axis=0))
        self.res_count = len(self.pocket_residues)
    
    def determine_pocket_residues(self,protein_pdb,centeroid):
        self.pocket_residues = []
        with open(protein_pdb,'r') as pdb_file:
            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')]
        pocket_residues_id = []
        pocket_atoms_content = [line for line in pdb_content if near_pocket(line,centeroid)]
        for atom_line in pocket_atoms_content:
            res_num,chain_id = str(atom_line[22:27]).strip(),atom_line[21]
            pocket_residues_id.append((res_num,chain_id))
        pocket_residues_id = set(pocket_residues_id)
        for (index,chain) in pocket_residues_id:
            try:
                residue_content = [line for line in pdb_content if str(line[22:27]).strip()==index and line[21] == chain]
                self.pocket_residues.append(Residue(residue_content))
            except:
                continue

class ComplexPocket():
    def __init__(self,protein_pdb,ligand_sdf,pK=0,decoy_sdf=None):
        with open(protein_pdb,'r') as pdb_file:
            pdb_content = [line.strip() for line in pdb_file if line.startswith('ATOM')]
        ligand = Chem.SDMolSupplier(ligand_sdf,removeHs=False)[0]
        self.ligand = Mol(ligand)
        
        self.pK = pK

        if decoy_sdf is not None and os.path.exists(decoy_sdf):
            self.decoys = [Mol(mol) for mol in Chem.SDMolSupplier(decoy_sdf,removeHs=False) if mol is not None]
        else:
            self.decoys = []
        self.decoys_count = len(self.decoys)

        self.pocket_residues = []
        pocket_residues_id = []
        pocket_atoms_content = [line for line in pdb_content if near_pocket(line,self.ligand.compute_centeroid())]

        for atom_line in pocket_atoms_content:
            res_num,chain_id = str(atom_line[22:27]).strip(),atom_line[21]
            pocket_residues_id.append((res_num,chain_id))
        pocket_residues_id = set(pocket_residues_id)
        for (index,chain) in pocket_residues_id:
            try:
                residue_content = [line for line in pdb_content if str(line[22:27]).strip()==index and line[21] == chain]
                self.pocket_residues.append(Residue(residue_content))
            except:
                continue
        
        self.pro_lig_interaction = self.get_interaction_label()
        self.res_features = np.concatenate([residue.get_feature() for residue in self.pocket_residues],axis=0)
        self.res_count = len(self.pocket_residues)
        self.distance_matrix = self.get_distance_matrix()

    def get_interaction_label(self):
        pro_lig_interaction = np.zeros(shape=[self.ligand.mol.GetNumAtoms(),len(self.pocket_residues)],dtype=np.float32)
        ligand_conformer = self.ligand.mol.GetConformer()
        for atom in self.ligand.mol.GetAtoms():
            atom_index = atom.GetIdx()
            atom_coordinate = np.array([
                ligand_conformer.GetAtomPosition(atom_index)[0],ligand_conformer.GetAtomPosition(atom_index)[1],
                ligand_conformer.GetAtomPosition(atom_index)[2]
            ])
            for residue_index in range(len(self.pocket_residues)):
                for res_atom_coordinate in self.pocket_residues[residue_index].coordinates:
                    if np.sqrt(np.sum(np.square(res_atom_coordinate - atom_coordinate))) > 4.0:
                        continue
                    else:
                        pro_lig_interaction[atom_index,residue_index] = 1
                        break
        return pro_lig_interaction  #[atom,res]
    
    def get_distance_matrix(self):
        mass_coordinates = np.array([residue.get_mass_center() for residue in self.pocket_residues],dtype=np.float32)
        alpha_coordinates = np.array([residue.get_alpha_position() for residue in self.pocket_residues],dtype=np.float32)
        mass_matrix = np.sqrt(np.sum(np.square(
            mass_coordinates.reshape([-1,1,3]) - mass_coordinates.reshape([1,-1,3])
        ),axis=-1,keepdims=True))
        alpha_matrix = np.sqrt(np.sum(np.square(
            alpha_coordinates.reshape([-1,1,3]) - alpha_coordinates.reshape([1,-1,3])
        ),axis=-1,keepdims=True))
        distance_matrix = np.concatenate([mass_matrix,alpha_matrix],axis=-1)
        return distance_matrix

class Residue():
    def __init__(self,residue_content):
        residue_content = self.process_alternate(residue_content)
        self.mass_center,self.coordinates,self.alpha_position = mass_center_from_pdb(residue_content)
        residue_type = residue_content[0][17:20]
        if residue_type == 'MSE':
            residue_type = 'MET'
        self.residue_type = residue_type
        self.feature = BLOSUM62[self.residue_type]
    
    def get_feature(self):
        return self.feature

    def get_mass_center(self):
        return self.mass_center

    def get_alpha_position(self):
        return self.alpha_position

    def process_alternate(self,contents):
        alternate_flag = list(set([line[16] for line in contents if line[16]!=" "]))
        if len(alternate_flag) > 0:
            contents = [line for line in contents if line[16]==alternate_flag[0] or line[16]==" "]
        return contents

class Mol():
    def __init__(self,mol,threeD=True):
        self.mol = mol
        self.atom_count = mol.GetNumAtoms()
        self.threeD=threeD

    def get_atom_coordinates(self):
        ligand_conformer = self.mol.GetConformer()
        atom_coordinates = np.array([
            [ligand_conformer.GetAtomPosition(atom.GetIdx())[0],ligand_conformer.GetAtomPosition(atom.GetIdx())[1],
            ligand_conformer.GetAtomPosition(atom.GetIdx())[2]] for atom in self.mol.GetAtoms()
            ],dtype=np.float32)
        if np.sum(atom_coordinates[:,-1]) == 0:
            self.threeD=False
        return atom_coordinates
    
    def get_interaction_label(self):
        atom_coordinates = self.get_atom_coordinates()
        atom_distances = np.sqrt(np.sum(np.square(atom_coordinates.reshape(-1,1,3) - atom_coordinates.reshape(1,-1,3)),axis=-1))
        interaction = np.where(atom_distances<4.0,1.,0.)
        return interaction

    def compute_centeroid(self) -> np.ndarray: 
        atom_coordinates = self.get_atom_coordinates()
        return np.mean(atom_coordinates,axis=0)

    def get_bonded_atoms(self) -> List[Tuple]:
        bonded_atom_pairs = []
        for bond in self.mol.GetBonds():
            bonded_atom_pairs.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
            bonded_atom_pairs.append((bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()))
        return bonded_atom_pairs

def random_ligand_decoy(protein_pockets,decoy_flag):
    if decoy_flag:
        check = True
        while check:
            complex_labels = []
            for pocket in protein_pockets:
                if pocket.pK != 0:
                    complex_labels.append(int(np.random.choice([0,1],size=1,p=[0.5,0.5])))
                else:
                    complex_labels.append(int(np.random.choice([0,1],size=1,p=[0.5,0.5])))
            complex_labels = np.array(complex_labels)
            pK_flags = np.zeros(len(protein_pockets))
            for i,pocket in enumerate(protein_pockets):
                if pocket.decoys_count==0:
                    complex_labels[i]=1
            for i,complex_label in enumerate(complex_labels):
                if complex_label == 1 and protein_pockets[i].pK != 0: ###when complex and  has pK data
                    pK_flags[i]=1
    
            if np.sum(pK_flags)>0 and np.sum(complex_labels)>0:
                check=False
    else:  #for valid set and core set and train set (affinity_only)
        complex_labels = np.ones(len(protein_pockets))
        pK_flags = np.ones(len(protein_pockets))

    mol_batch = []  ##self-designed class Mol
    for i,complex_label in enumerate(complex_labels):
        if complex_label == 1:
            mol_batch.append(protein_pockets[i].ligand)   #ligand (self-designed class Mol)

        else:
            decoy_index = np.random.randint(0,protein_pockets[i].decoys_count)
            mol_batch.append(protein_pockets[i].decoys[decoy_index])
    
    pKs = []
    for i,pK_flag in enumerate(pK_flags):
        if pK_flag == 1:
            pKs.append(protein_pockets[i].pK)
        else:
            pKs.append(0)  #will be masked by pK_flags
    
    pK_flags=torch.from_numpy(pK_flags.astype(np.float32))
    pKs = torch.from_numpy(np.array(pKs,dtype=np.float32))
    complex_labels = torch.from_numpy(complex_labels.astype(np.float32))
    return pKs,pK_flags,complex_labels,mol_batch

def tensorize_all(protein_pockets,decoy_flag=True):
    pKs,pK_flags,complex_labels,mol_batch = random_ligand_decoy(protein_pockets,decoy_flag)
    res_feature_batch = tensorize_protein_pocket(protein_pockets)
    mol_feature_batch,mol_intearctions = tensorize_ligand([mol for mol in mol_batch])
    pro_lig_interactions = []
    for i,label in enumerate(complex_labels):
        if label == 1:
            pro_lig_interactions.append(protein_pockets[i].pro_lig_interaction.reshape([-1]))
        else:
            pro_lig_interactions.append(np.zeros([mol_batch[i].atom_count,protein_pockets[i].res_count]).reshape([-1]))  
    pro_lig_interactions = torch.from_numpy(np.concatenate(pro_lig_interactions).astype(np.float32))
    return res_feature_batch,mol_feature_batch,mol_intearctions,pro_lig_interactions,pKs,pK_flags,complex_labels

def tensorize_protein_pocket(protein_pockets):
    fresidues,res_scope,alpha_coordinates = [],[],[]
    residues_count = sum([len(pocket.pocket_residues) for pocket in protein_pockets])
    #distance_matrix = np.ones(shape=[residues_count,residues_count,2],dtype=np.float32)*1e6
    res_map = np.zeros(shape=[residues_count,residues_count,1],dtype=np.float32)
    total_residues = 0
    for pocket in protein_pockets:
        residue_count = len(pocket.pocket_residues)
        for residue in pocket.pocket_residues:
            fresidues.append(residue.get_feature())
            alpha_coordinates.append(residue.get_alpha_position())
        res_scope.append((total_residues,residue_count))
        #distance_matrix[total_residues:total_residues+residue_count,total_residues:total_residues+residue_count] = pocket.distance_matrix
        res_map[total_residues:total_residues+residue_count,total_residues:total_residues+residue_count] = 1.
        total_residues += residue_count
    fresidues = torch.from_numpy(np.reshape(np.array(fresidues,dtype=np.float32),[-1,20]))
    res_map = torch.from_numpy(res_map)
    alpha_coordinates = torch.from_numpy(np.reshape(np.array(alpha_coordinates,dtype=np.float32),[-1,3])) 
    #distance_matrix = torch.from_numpy(distance_matrix)
    return (fresidues,res_map,res_scope,alpha_coordinates)

def tensorize_ligand(mol_batch:List[Mol]):
    (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch_to_graph([mol.mol for mol in mol_batch]) 
    mol_intearctions = [mol.get_interaction_label().reshape([-1]) for mol in mol_batch]
    mol_intearctions = torch.from_numpy(np.concatenate(mol_intearctions).astype(np.float32))
    return (fatoms, fbonds, agraph, bgraph, lig_scope),mol_intearctions   

def tensorize_molecules(mol_batch:List[Chem.rdchem.Mol]):
    (fatoms, fbonds, agraph, bgraph, lig_scope) = mol_batch_to_graph(mol_batch)
    return (fatoms, fbonds, agraph, bgraph, lig_scope)

def mol_batch_to_graph(mol_batch:List[Chem.rdchem.Mol],auto_detect=True):
    padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
    fatoms,fbonds = [],[padding] #Ensure bond is 1-indexed
    in_bonds,all_bonds = [],[(-1,-1)] #Ensure bond is 1-indexed
    lig_scope = []
    total_atoms = 0

    for mol in mol_batch:
        if auto_detect:
            Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
        n_atoms = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            fatoms.append(atom_features(atom))
            in_bonds.append([])

        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            x = a1.GetIdx() + total_atoms
            y = a2.GetIdx() + total_atoms

            b = len(all_bonds)
            all_bonds.append((x,y))
            fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
            in_bonds[y].append(b)

            b = len(all_bonds)
            all_bonds.append((y,x))
            fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
            in_bonds[x].append(b)

        lig_scope.append((total_atoms,n_atoms))
        total_atoms += n_atoms

    total_bonds = len(all_bonds)
    fatoms = torch.stack(fatoms, 0)
    fbonds = torch.stack(fbonds, 0)
    agraph = torch.zeros(total_atoms,MAX_NB).long()
    bgraph = torch.zeros(total_bonds,MAX_NB).long()

    for a in range(total_atoms):
        for i,b in enumerate(in_bonds[a]):
            agraph[a,i] = b   # index of edges pointed to atom 

    for b1 in range(1, total_bonds):
        x,y = all_bonds[b1]
        for i,b2 in enumerate(in_bonds[x]):
            if all_bonds[b2][0] != y:
                bgraph[b1,i] = b2  # index of edges pointed to edge 

    return (fatoms, fbonds, agraph, bgraph, lig_scope)

def role_of_5(mol):
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Descriptors.MolLogP(mol)
    mol_ha = Descriptors.NumHAcceptors(mol)
    mol_hd = Descriptors.NumHDonors(mol)
    mol_rb = Descriptors.NumRotatableBonds(mol)
    qed_score = qed(mol)
    meet = [mol_weight>=300 and mol_weight<=500, mol_logp<5,mol_ha<=10,mol_hd<=5,mol_rb<=10]
    if sum(meet) >=4 and qed_score >=0.3 :
        return True
    else:
        return False
        
