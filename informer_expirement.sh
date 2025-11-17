#!/bin/bash

#BSUB -J DL_INFORMEREXPER                  # Job name
#BSUB -o logs/output_informerexper_%J.log                # Output log file (including job ID %J)
#BSUB -e logs/error__informerexper%J.err                 # Error log file
#BSUB -q gpuv100                          # Queue name (gpu queue)
#BSUB -gpu "num=1:mode=exclusive_process"   # Request 1 GPU
#BSUB -R "rusage[mem=5G]"             # Request 5 GB of RAM
#BSUB -W 05:00                        # Time limit (5 hours)
#BSUB -R "span[hosts=1]"             # Number of CPUs
#BSUB -n 4                            # Number of CPU cores

# Run this command first to make the script executable:
# chmod +x job.sh
# Then, when wanting to run on GPU, we need to run this job in the terminal with:
# bsub < ./job.sh
# It uses some scheduling system which uses these bsub arguments above to assign ressources somehow :)

# Load the cuda module to make it accessible to pytorch
module load cuda

# changes directory into your home (change this to your own home path (which you can find with 'echo $HOME'))
#cd /zhome/b1/8/213657/
cd /zhome/63/7/219953/

# sources the python virtual environment (not sure if you need this, probably not if you installed everything in the base python)
source venv/bin/activate
# changes directory into the deep learning project folder
cd DeepLearning-1
# executes the script
python tpinform_experiment.py\
    --k 1500 \
    --epochs 400 \
    --ds_diff_in_seq 20 \
    --ds_window_total 420 \
    --ds_window_pred 120 \
    --ds_stride 15 \
    --training_batchsize 64 \
    --training_lr 5e-4
