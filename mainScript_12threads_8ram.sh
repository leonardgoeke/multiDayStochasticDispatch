#!/bin/bash --login
#SBATCH --array=10,11
#SBATCH --time=120:00:00
#SBATCH --job-name=results/run_%j
#SBATCH --output=results/run_%j.out
#SBATCH --error=results/run_%j.err

module add julia/1.10.3
module add gurobi/12.0.1

sbatch --nodes=1 --ntasks=1 --mem-per-cpu=8G --time=7200 --cpus-per-task=12 --ntasks-per-node=1 --wrap "julia -t \$SLURM_CPUS_PER_TASK mainScript.jl  $SLURM_ARRAY_TASK_ID"
