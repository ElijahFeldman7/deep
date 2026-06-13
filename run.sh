#!/bin/bash
#SBATCH --job-name=deepseek_turing_custom
#SBATCH --partition=gpu
#SBATCH --nodelist=unicron       
#SBATCH --gres=gpu:turing:6      # Request all 6 GPUs
#SBATCH --cpus-per-task=16       
#SBATCH --mem=32G                
#SBATCH --time=48:00:00          
#SBATCH --output=ner_pipeline_%j.out
#SBATCH --error=ner_pipeline_%j.err

# 1. Activate your environment
source /tmp/venv/bin/activate

# 2. Start the custom-built llama-server in the background
# (We added --split-mode layer and --flash-attn off to prevent graph input inflation)
echo "Starting custom llama-server with DeepSeek-V4-Flash..."
./llama.cpp/build/bin/llama-server \
    --model /csl/users/2028efeldman/model/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf \
    --n_gpu_layers 99 \
    --ctx_size 32768 \
    --fit off \
    --flash-attn off \
    --split-mode layer \
    --port 8000 &
SERVER_PID=$!

# 3. Wait properly using the /health endpoint until it returns HTTP 200
echo "Waiting for the 165GB model to fully load into VRAM..."
while [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)" -ne 200 ]; do
    sleep 15
done
echo "Custom V4 server is fully loaded and ready on port 8000!"

# 4. Run your python extraction script
python run.py \
    --input dataset.csv \
    --output dataset_extracted.csv \
    --cache cache.json \
    --url http://localhost:8000/v1 \
    --model /csl/users/2028efeldman/model/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf
    --concurrency 4
# 5. Clean up
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null
