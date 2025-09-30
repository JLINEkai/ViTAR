CUDA_VISIBLE_DEVICES=0,1 vllm serve ./checkpoint/ViTAR/actor/huggingface \
--port 8002 --gpu-memory-utilization 0.8 \
--max-model-len 32768 \
--tensor-parallel-size 2 \
--allowed-local-media-path ./images 
