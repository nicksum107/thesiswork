srun \
  --account=pvl \
  --job-name=test \
  --mail-user=nsum@princeton.edu \
  --mail-type=ALL \
  --ntasks=1 \
  --cpus-per-task=2 \
  --mem=32768 \
  --gres=gpu:1 \
  --time=00-04:00:00 \
  --pty bash