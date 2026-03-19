#!/bin/bash

#SBATCH --job-name=moscal2.0
#SBATCH --output=moscal%j.out
#SBATCH -q normal
#SBATCH -p douwj
#SBATCH -n 1
#SBATCH --cpus-per-task=8  # Request 24 CPUs for the task

export OMP_NUM_THREADS=8  # You can adjust this if using OpenMP elsewhere

BIN=/storage/douwenjieLab/biruihao/mkct/99-production/apps/moscal2.0/build/code/bose_quad_2.out

${BIN}
