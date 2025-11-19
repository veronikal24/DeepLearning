# DeepLearning

Lightweight project for experimenting with AIS vessel trajectory modeling and small Informer/Transformer variants.

**Repository Structure**
- `dataloader.py` — utilities to load, preprocess and convert AIS CSV/parquet data.
- `informer_original_paper.py` — Implementation from original authors.
- `output_utils.py` - utilities to handle the *bsub* related log files.
- `plotting.py` - diverse plotting functions for analysis.
- `tptrans.py`, `tpinform.py`, `temptp.py` — model files (temptp contains 1 model using transformer and 1 using informer encoder).
- `utils.py` - small functions regarding handling of lat/lon and relative distances.
- `checkpoints/` — saved model checkpoints.
- `command_scripts/` — helper scripts for running jobs (e.g. `job.sh`).
- `logs/` and `important_performance_log_files/` — training and evaluation logs.
- `dataset/` — directory for the MMSI parquet files.
- `*.ipynb` — exploration and evaluation notebooks.


**Running on a cluster / batch system**
- The repo contains `*_job.sh` as submission scripts for the LSF cluster scheduler. Edit resource requests and paths then submit with your scheduler's submit command (e.g. `bsub < job.sh`).

**TODO / Improvements**
- Make model output filenames include parameter summaries (hyperparams, dataset name).
- Train Informer and Transformer variants using consistent training configurations for fair comparison.
- Add distance-based evaluation metrics (e.g., haversine distance between predicted and true positions).
- Move non-essential outputs (temporary files, error dumps) to `.gitignore`.


