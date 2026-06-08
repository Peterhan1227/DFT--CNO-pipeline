# DFT–CNO Pipeline

A Python pipeline for computing Charge Neutral Orbitals (CNOs) from VASP wavefunctions, including density matrix construction, cube file export, and fatband analysis.

## Requirements

- Python 3.10 or newer
- `numpy`
- `matplotlib`
- `scipy`
- `ase`
- `VaspBandUnfolding` / `PyVaspWfc`

`VaspBandUnfolding` provides the `vaspwfc` module used by all scripts.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Peterhan1227/DFT--CNO-pipeline.git
cd DFT--CNO-pipeline
```

### 2. Create a conda environment

```bash
conda create -n vaspwfc-env python=3.10 -y
conda activate vaspwfc-env
```

### 3. Install dependencies

```bash
pip install numpy matplotlib scipy ase
pip install git+https://github.com/QijingZheng/VaspBandUnfolding.git
```

Or install `VaspBandUnfolding` from source:

```bash
pip install numpy matplotlib scipy ase
git clone https://github.com/QijingZheng/VaspBandUnfolding.git
cd VaspBandUnfolding
pip install .
```

## Data folder structure

Each material lives under `Density matrix cal/Data/<MaterialName>/` and must contain:

```
Data/
└── CoSn/               # material name matches MATERIAL in config.py
    ├── WAVECAR         # VASP wavefunction (not tracked by git)
    ├── WAVECAR_lm      # lm-decomposed wavefunction (not tracked by git)
    ├── POSCAR
    ├── KPOINTS
    ├── EIGENVAL
    ├── EIGENVAL_lm
    └── output/         # created automatically; contents not tracked by git
```

Add new materials by creating a subfolder with the same structure.

## Configuration

Open `Density matrix cal/config.py` and set the material and calculation type before running:

```python
MATERIAL = "CoSn"   # must match a subfolder name under Data/
LSORBIT  = True     # True for non-collinear (SOC) calculations, False otherwise
```

## How to run

Navigate into the scripts directory first:

```bash
cd 'Density matrix cal'
```

Then run the scripts in order:

```bash
python Wavecar_to_Coeff.py      # build density matrix and CNO data from WAVECAR
python export_cno_cubes.py      # export cube files for visualization
python cno_single_fatband.py    # generate CNO fatband analysis outputs
python plot_cno_eigenvalues.py  # plot CNO eigenvalue spectrum
python combine_cno_cubes.py     # combine cube files
```

## Notes

- All scripts import `from vaspwfc import vaspwfc`, so `VaspBandUnfolding` must be installed in the active environment before running.
- If `python` does not point to the correct environment, activate the conda environment first and rerun.
- Output files (`.npy`, `.cube`, `.png`, etc.) are written to `Data/<MaterialName>/output/` and are excluded from git tracking.
