# Requirements and Usage

## What this project needs

This project depends on:

- Python 3.10 or newer
- `numpy`
- `matplotlib`
- `scipy`
- `ase`
- `VaspBandUnfolding` / `PyVaspWfc`

`VaspBandUnfolding` provides the `vaspwfc` module used by the scripts in `Density matrix cal`.

## How to download the project

Clone the repository:

```bash
git clone https://github.com/Peterhan1227/DFT--CNO-pipeline.git
cd DFT--CNO-pipeline
```

## Recommended setup

Create and activate a fresh conda environment:

```bash
conda create -n vaspwfc-env python=3.10 -y
conda activate vaspwfc-env
```

## Install dependencies

Install the Python packages:

```bash
pip install numpy matplotlib scipy ase
pip install git+https://github.com/QijingZheng/VaspBandUnfolding.git
```
or

```bash
pip install numpy matplotlib scipy ase
git clone https://github.com/QijingZheng/VaspBandUnfolding.git
cd VaspBandUnfolding
pip install .
```


## How to run

first `cd '.\Density matrix cal\'` to get into the correct path.

Then simply use python to run the code. Example:

python Wavecar_to_Coeff.py
python export_cno_cubes.py
python cno_single_fatband.py
```

## Main scripts

- `Density matrix cal/Wavecar_to_Coeff.py`: builds the density matrix and CNO data from `WAVECAR`
- `Density matrix cal/export_cno_cubes.py`: exports cube files for visualization
- `Density matrix cal/cno_single_fatband.py`: generates CNO fatband analysis outputs

## Notes

- The scripts import `from vaspwfc import vaspwfc`, so `VaspBandUnfolding` must be installed in the active Python environment before running.
- If `python` does not point to the correct environment, activate the environment first and then rerun the script.
