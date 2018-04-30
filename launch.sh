#!/bin/bash
#SBATCH --job-name=cmp-stack
#SBATCH --mail-user=georgia.stuart@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=28
#SBATCH --output=outfile.out
#SBATCH --partition=Math
#SBATCH --account=gks090020
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4096

source /etc/profile.d/modules.sh
module load mpich/mpich3.2
export PATH="/petastore/ganymede/home/gks090020/miniconda3/envs/mcmc/bin:$PATH"
export PATH="/petastore/ganymede/home/gks090020/cmp-stack:$PATH"


mpirun -np 28 python main.py