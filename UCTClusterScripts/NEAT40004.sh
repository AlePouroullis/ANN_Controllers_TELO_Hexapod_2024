#!/bin/sh
#SBATCH --account=compsci
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=24
#SBATCH --time=130:00:00
#SBATCH --job-name="40000 4"
#SBATCH --mail-user=sctmic015@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

module load python/anaconda-python-3.7

/home/sctmic015/HonoursProject

python3 mapElitesNEAT.py 40000 4



