#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1  # number of gpus per node
#SBATCH -time 00:05:00

#SBATCH --mail-type=begin        # send mail when job begins
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --partition interactive_gpu_p
#SBATCH --qos interactive_gpu

#SBATCH -c 6 <- don't use more than 50% (25% for icb-gpusrv0[1-2])
          # of GPU queue node cores unless you request entire node
#SBATCH --mem=15G <- dont use more than 50% (25% for icb-gpusrv0[1-2])
          # of GPU queue node memory unless you request entire node

#SBATCH --nice=10000 <- manual priority (should always be set to 10000),
          # always include this line

source $HOME/.bashrc

# do stuff
conda activate jointformer
python experiments/joint_training/test.py