#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=0
#SBATCH --partition=GPU
#SBATCH --job-name=Resshift
#SBATCH --error=logs/error.log
#SBATCH --output=logs/result.log
source ../.env-torch/bin/activate
# DISABLE_TELEMETRY=YES TOKENIZERS_PARALLELISM=true srun python unet_refiner.py fit --config config/unet_refiner_track1.yaml
DISABLE_TELEMETRY=YES TOKENIZERS_PARALLELISM=true srun python diffusion_resshift_unet.py fit --config config/diffusion_resshift_imagenet_unet.yaml
# DISABLE_TELEMETRY=YES TOKENIZERS_PARALLELISM=true python inference.py