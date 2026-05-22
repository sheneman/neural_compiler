#!/bin/bash
#SBATCH --job-name=neural_compiler_bench
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/slurm_%j.log
#SBATCH --error=benchmarks/slurm_%j.log

cd /path/to/neural_compiler

source .venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export BENCH_PROGRESS_FILE="$PWD/benchmarks/progress.log"
> "$BENCH_PROGRESS_FILE"

echo "=== Environment ==="
hostname
date
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
python3 -c "import torch; print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')" 2>/dev/null
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

echo "=== Running tests ==="
python3 -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
echo ""

echo "=== Large program benchmarks (VRAM-aware batch sizing) ==="

mkdir -p benchmarks/figures_rtx4090

stdbuf -oL -eL python3 -m benchmarks.run_benchmarks \
    --large-only \
    -o benchmarks/results_rtx4090_large.csv 2>&1

echo ""
echo "=== Generating plots ==="
python3 -m benchmarks.plot_benchmarks benchmarks/results_rtx4090_large.csv \
    --output-dir benchmarks/figures_rtx4090

echo ""
echo "=== Done ==="
date
