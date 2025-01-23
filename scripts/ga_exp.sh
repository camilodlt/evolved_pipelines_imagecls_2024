#!/bin/bash 
#SBATCH --job-name=MAGE_GA
#SBATCH --time=24:00:00
##SBATCH --partition=exlusive
#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --threads-per-core=2
#SBATCH --hint=multithread 
#SBATCH --mem-per-cpu=2G

# CD
cd /tmpdir/$USER/dev/GA_PCAM

# LOAD MODULES
module purge
module load conda/4.9.2
# module load python/3.8.5
module load julia/1.10.5
module load intelmpi chdb/1.0
conda activate mage

export JULIA_DEPOT_PATH=/tmpdir/$USER/.julia
export UTCGP_PYTHON=~/.conda/envs/mage/bin/python
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE=~/.conda/envs/mage/bin/python
export OPENBLAS_NUM_THREADS=1

# export UTCGP_CONSTRAINED=yes
# export UTCGP_MIN_INT=-10000
# export UTCGP_MAX_INT=10000
# export UTCGP_MIN_FLOAT=-10000
# export UTCGP_MAX_FLOAT=10000
# export UTCGP_SMALL_ARRAY=100
# export UTCGP_BIG_ARRAY=1000

SEED=$1
TMP=/tmpdir/$USER/dev/GA_PCAM/metrics_ga/tmp

srun $(placement 1 64 --hyper --mode=compact) julia --project --threads=64 --startup-file=no src/ga.jl --mutation_rate 1 --n_nodes 10 --n_elite 6 --n_new 14 --tour_size 3 --n_samples 100 --acc_weight 0.5188 --n_repetitions 3 --budget 30000000 --seed $SEED 1> $TMP/$SEED.stdout 2> $TMP/$SEED.stderr

