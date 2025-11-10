#!/bin/bash

#BSUB -J DL_Boats                  # Job name
#BSUB -o output_%J.log                # Output log file (including job ID %J)
#BSUB -e error_%J.err                 # Error log file
#BSUB -q c02516                          # Queue name (gpu queue)
#BSUB -gpu "num=1:mode=exclusive_process"   # Request 1 GPU
#BSUB -R "rusage[mem=10G]"             # Request 8 GB of RAM
#BSUB -W 01:00                        # Time limit (1 hour)
#BSUB -R "span[hosts=1]"             # Number of CPUs
#BSUB -n 4                            # Number of CPU cores

# When we want to execute the script on GPU, we need to run this job in the terminal with:
# ./job.sh
# It uses some scheduling system which uses these bsub arguments above to assign ressources somehow :)

# Load the cuda module to make it accessible to pytorch
module load cuda

# change directory into your home (i think you have to change the part on your machine (to find your home path, just type 'echo $HOME'))
cd /zhome/b1/8/213657/
# sources the python virtual environment (not sure if you need this, probably not if you installed everything in the base python)
source venv/bin/activate
# change directory into the deep learning project folder
cd DeepLearning
# execute the script
python main.py
