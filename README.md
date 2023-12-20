# **PLANET**

### PLANET: _**P**rotein-**L**igand **A**ffinity prediction **NET**work_

Predicting protein-ligand binding affinity is still a central issue in drug design. No wonder various deep learning models have been developed in recent years to tackle this issue in one aspect or another. So far most of them merely focus on reproducing the binding affinity of known binders (i.e. so-called “scoring power”).<br>
Here, we have developed a graph neural network model called PLANET (Protein-Ligand Affinity prediction NETwork). This model takes the graph-represented 3D structure of the binding pocket on the target protein and the 2D structural graph of the ligand molecule as inputs. PLANET was trained through a multi-objective process with three related tasks, i.e. deriving protein-ligand binding affinity, protein-ligand contact map, and intra-ligand distance matrix. <br>
As tested on the CASF-2016 benchmark, PLANET exhibited a comparable level of scoring power as some other machine learning models that rely on 3D protein-ligand complex structures as inputs. Besides, it exhibited notably better performance in virtual screening trials on the DUD-E and LIT-PCBA benchmarks. Compared to the popular conventional docking program GLIDE, PLANET took less than one percent of computation time to finish the same virtual screening job without a significant loss in accuracy because it did not need to perform exhaustive conformational sampling. In summary, PLANET achieved a decent performance in virtual screening as well as predicting protein-ligand binding affinity. This feature makes PLANET an attractive tool for drug discovery in the real world.

### Usage
1. Setup dependencies
```bash
conda env create -f planet.yaml
conda activate planet
```
2. Using PLANET <br>
We have created a demo folder which includes a protein file (adrb2.pdb), a crystal ligand file (adrb2_ligand.sdf) as well as molecules (mols.sdf) to be estimated. These files are  originally derived from DUD-E dataset and prepared as below: <br>
(1) The protein structure file (.pdb) are prepared using *prepwizard* in Maestro, including fixing broken residues and assign protonated states. Other structure preparation tools can also be applied. _NOTE:_ The most important is that $\alpha$-carbon of each reasidues must be correctly fixed in the .pdb file.
(2) The molecule files (mols.sdf) are prepared using *epik* in Maestro, including adding hydrogen atoms and ionized states. the adrb2_ligand.sdf file is only used for determining binding pocket (only need to be in .sdf format).
```bash
cd demo
python3.6 ../PLANET_run.py -p adrb2.pdb -l adrb2_ligand.sdf -m mols.sdf
```
3. Parameters
   - _-p or --protein_, protein structure file;
   - _-l or --ligand_, crystal ligand file for determining binding pocket, if specified, override the coordinate provided.
   - _-x or --center_x ; -y or --center_y ; -z or --center_z , coordinates to define the center of binding pocket, same as follows
   - _-m or --mol_file_, molecules to be esitmated in .sdf format
   - _--prefix_, if not specified, the default is "result", that is the outcome will be saved as "result.csv" and "result.sdf"
4. Output files <br>
We provide two output formats including .csv and .sdf

### Training PLANET
We provided the training scripts called "PLANET_train.py" and "PLANET_datautils.py". But the training data (i.e. PDBbind general set v.2020) are not included in this repository, which can be accessed through: http://pdbbind.org.cn/. <br>
As mentioned in our paper (in preparation), all structures in general set are prepared and a large number of decoy molecules are used for augmentation. This part of data are not provided to public till now. <br>
If anyone want to re-train the PLANET (maybe after the training data is released, at that time another folder called 'data' will be released, in which include the summary of training set, validation set and core set in .csv format), here is the protocol: <br>
suppose the absolute path to PDBbind general set is $DATASET, and all the scripts related to PLANET are in $PLANET. 
```bash
python3.6 $PLANET/process_PDBbind.py -d $DATASET -n $njobs
python3.6 $PLANET/PLANET_datautils.py -p $DATASET -d $PLANET/data/
python3.6 $PLANET/PLANET_train.py -t $PLANET/data/train.pkl  -v $PLANET/data/valid.pkl -te $PLANET/data/core.pkl -d .
```
