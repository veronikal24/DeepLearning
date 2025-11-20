#!/bin/bash
#BSUB -J TempTPI_Ablation
#BSUB -o ablation_studies/logs/results_TPT_%J.log
#BSUB -e ablation_studies/logs/errors_TPT_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4G]"
#BSUB -W 20:00
#BSUB -R "span[hosts=1]"
#BSUB -n 4

# Run this command first to make the script executable:
# chmod +x ablation_test.sh
# Then, when wanting to run on GPU, we need to run this job in the terminal with:
# bsub < ./ablation_test.sh

module load cuda
cd /zhome/0a/d/219948/
source venv/bin/activate
cd repos/DeepLearning/ablation_studies
python run.py --model TempTPI
