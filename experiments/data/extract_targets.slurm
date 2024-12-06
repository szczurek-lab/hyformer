#!/bin/bash
             
#SBATCH --job-name=preprocess_physchem_unimol
#SBATCH --output logs/%x/%j.log                              
#SBATCH --error logs/%x/%j.err   

#SBATCH --array=0-1    

#SBATCH -p cpu_p
#SBATCH --qos cpu_normal

#SBATCH --nodes=1                                           
#SBATCH --ntasks-per-node=1                                  
#SBATCH --cpus-per-task=32                                   
#SBATCH --mem=80G                                            
#SBATCH --time=72:00:00                                    

#SBATCH --mail-type=begin                                   
#SBATCH --mail-type=end                                     
#SBATCH --mail-type=fail                                    
#SBATCH --mail-user=adam.izdebski@helmholtz-munich.de

#SBATCH --nice=1000                                        
###

# Define the arrays
splits=("test" "train")
targets=("physchem")

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

source $HOME/.bashrc

conda deactivate
conda activate jointformer

# Run the script
echo "Extracting $target for $split..."
python3 experiments/data/extract_targets.py \
    --target $target \
    --data_path /lustre/groups/aih/jointformer/data/data/unimol/truncate_only/smiles_${split}.npy \
    --output /lustre/groups/aih/jointformer/data/data/unimol/truncate_only/smiles_${target}_${split}.npy \
    --n_workers $SLURM_CPUS_PER_TASK
