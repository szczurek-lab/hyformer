#!/bin/bash

#SBATCH --job-name=extract_targets   
#SBATCH --array=0-11                 
#SBATCH --output=array_job_%A_%a.out  
#SBATCH --error=array_job_%A_%a.err   

#SBATCH -p cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                  
#SBATCH --cpus-per-task=32                                   
#SBATCH --mem=8G                                            
#SBATCH --time=24:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=1000                                        
###

# Define the arrays
splits=("train" "val" "test")
targets=("plogp" "qed" "guacamol_mpo" "physchem")

# Create a list of combinations
combinations=()
for split in "${splits[@]}"; do
  for target in "${targets[@]}"; do
    combinations+=("$split,$target")
  done
done

# Get the specific combination for this job
combination=${combinations[$SLURM_ARRAY_TASK_ID]}
split=$(echo $combination | cut -d',' -f1)
target=$(echo $combination | cut -d',' -f2)

# Modify the split name if needed
if [ "$split" == "train" ] && [ "$target" == "guacamol_mpo" ]; then
  split="train/10000/seed_0"
fi

# Run the script
echo "Extracting $target for $split..."
python3 experiments/data/extract_targets.py \
    --target $target \
    --data_path /lustre/groups/aih/jointformer/data/datasets/guacamol/$split/smiles.txt \
    --output /lustre/groups/aih/jointformer/data/datasets/guacamol/$split/$target.npy \
    --n_workers $SLURM_CPUS_PER_TASK
