<<<<<<< HEAD
python -u train_sourceonly.py
=======
#!/bin/bash

#SBATCH --job-name=gta5
#SBATCH --gres=gpu:1
#SBATCH -o slurm.out
#SBATCH --time=14-0  # 10 hours

. /data/shinahyung/anaconda3/etc/profile.d/conda.sh
conda activate torch38gpu

python -u train_sourceonly.py
>>>>>>> target supervised
