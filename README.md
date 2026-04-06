# **PLANET**

### PLANET: _**P**rotein-**L**igand **A**ffinity prediction **NET**work_

Predicting protein-ligand binding affinity is still a central issue in drug design. No wonder various deep learning models have been developed in recent years to tackle this issue in one aspect or another. So far most of them merely focus on reproducing the binding affinity of known binders (i.e. so-called “scoring power”).<br>
Here, we have developed a graph neural network model called PLANET (Protein-Ligand Affinity prediction NETwork). This model takes the graph-represented 3D structure of the binding pocket on the target protein and the 2D structural graph of the ligand molecule as inputs. PLANET was trained through a multi-objective process with three related tasks, i.e. deriving protein-ligand binding affinity, protein-ligand contact map, and intra-ligand distance matrix. <br>
As tested on the CASF-2016 benchmark, PLANET exhibited a comparable level of scoring power as some other machine learning models that rely on 3D protein-ligand complex structures as inputs. Besides, it exhibited notably better performance in virtual screening trials on the DUD-E and LIT-PCBA benchmarks. Compared to the popular conventional docking program GLIDE, PLANET took less than one percent of computation time to finish the same virtual screening job without a significant loss in accuracy because it did not need to perform exhaustive conformational sampling. In summary, PLANET achieved a decent performance in virtual screening as well as predicting protein-ligand binding affinity. This feature makes PLANET an attractive tool for drug discovery in the real world.

### Usage
1. Setup dependencies (requires [uv](https://docs.astral.sh/uv/))
```bash
uv sync
```
This installs PyTorch with CUDA 12.1 support by default. For CPU-only or a different CUDA version edit the index URL in `pyproject.toml` before running `uv sync`:
- CPU: `https://download.pytorch.org/whl/cpu`
- CUDA 12.4: `https://download.pytorch.org/whl/cu124`

2. Using PLANET <br>
We have created a demo folder which includes a protein file (adrb2.pdb), a crystal ligand file (adrb2_ligand.sdf) as well as molecules (mols.sdf) to be estimated. These files are  originally derived from DUD-E dataset and prepared as below: <br>
(1) The protein structure file (.pdb) are prepared using *prepwizard* in Maestro, including fixing broken residues and assign protonated states. Other structure preparation tools can also be applied. _NOTE:_ The most important is that $\alpha$-carbon of each reasidues must be correctly fixed in the .pdb file.
(2) The molecule files (mols.sdf) are prepared using *epik* in Maestro, including adding hydrogen atoms and ionized states. the adrb2_ligand.sdf file is only used for determining binding pocket (only need to be in .sdf format).
```bash
cd demo
uv run ../PLANET_run.py -p adrb2.pdb -l adrb2_ligand.sdf -m mols.sdf
```
3. Parameters
   - _-p or --protein_, protein structure file;
   - _-l or --ligand_, crystal ligand file for determining binding pocket, if specified, override the coordinate provided.
   - _-x or --center_x ; -y or --center_y ; -z or --center_z , coordinates to define the center of binding pocket, same as follows
   - _-m or --mol_file_, molecules to be esitmated in .sdf format
   - _--prefix_, if not specified, the default is "result", that is the outcome will be saved as "result.csv" and "result.sdf"
4. Output files <br>
We provide two output formats including .csv and .sdf

### Training PLANET on PDBbind
Training data (PDBbind general set v.2020) can be accessed at http://pdbbind.org.cn/. <br>
Suppose `$DATASET` is the absolute path to your PDBbind dataset directory, `$PK_JSON` is the path to your `PDBbind2020.json` file with pK values, and `$PLANET` is the repo root.

**Step 1 – preprocess structures (runs in parallel with `$NJOBS` workers):**
```bash
uv run $PLANET/process_PDBBind.py -d $DATASET -n $NJOBS -k $PK_JSON
```

**Step 2 – build train/valid/core pickle files:**
```bash
uv run $PLANET/PLANET_datautils.py -p $DATASET -d $PLANET/data/
```

**Step 3 – train:**
```bash
uv run $PLANET/PLANET_train.py \
    -t $PLANET/data/train.pkl \
    -v $PLANET/data/valid.pkl \
    -te $PLANET/data/core.pkl \
    -d .
```
