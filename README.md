# 02456 Deep Learning

Group 48 project for 02456 Deep Learning course, for experimenting with AIS vessel trajectory modeling and small Informer/Transformer variants.

## How to start

**To recreate the results go to the `FinalProject.ipynb`**


## Repository Structure
- `dataloader.py` — utilities to load, preprocess and convert AIS CSV/parquet data.
- `informer_original_paper.py` — Implementation from original authors.
- `plotting.py` - diverse plotting functions for analysis.
- `tptrans.py`, `tpinform.py`, `temptpi.py` — model files (temptp contains 1 model using transformer and 1 using informer encoder).
- `utils.py` - small functions regarding handling of lat/lon and relative distances.
- `checkpoints/` — saved model checkpoints.
- `command_scripts/` — helper scripts for running jobs (e.g. `job.sh`).
- `logs/` and `important_performance_log_files/` — training and evaluation logs.
- `dataset/` — directory for the MMSI parquet files.
- `*.ipynb` — exploration and evaluation notebooks.
- `ablation_studies/` - the script and results to run the ablation studies

## Running on a cluster / batch system 
- The repo contains `*_job.sh` as submission scripts for the LSF cluster scheduler. Edit resource requests and paths then submit with your scheduler's submit command (e.g. `bsub < job.sh`).



