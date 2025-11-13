# DeepLearning

Lightweight project for experimenting with AIS vessel trajectory modeling and small Informer/Transformer variants.

**Repository Structure**
- `dataloader.py` — utilities to load, preprocess and convert AIS CSV/parquet data.
- `informer_min.py` — compact Informer implementation used for experiments.
- `tpinform.py`, `tptrans.py` — experiment / training scripts (project-specific).
- `checkpoints/` — saved model checkpoints.
- `command_scripts/` — helper scripts for running jobs (e.g. `job.sh`).
- `logs/` and `important_performance_log_files/` — training and evaluation logs.
- `aisdk-.../` — raw AIS data folders used in preprocessing.
- Notebooks: `DataExplorationVeronika.ipynb`, `InformerPerformance.ipynb` — exploration and evaluation notebooks.


**Running on a cluster / batch system**
- The repo contains `job.sh` as an example submission script for an LSF-like scheduler. Edit resource requests and paths then submit with your scheduler's submit command (e.g. `bsub < job.sh`).

**TODO / Improvements**
- Make model output filenames include parameter summaries (hyperparams, dataset name).
- Train Informer and Transformer variants using consistent training configurations for fair comparison.
- Add distance-based evaluation metrics (e.g., haversine distance between predicted and true positions).
- Move non-essential outputs (temporary files, error dumps) to `.gitignore`.


