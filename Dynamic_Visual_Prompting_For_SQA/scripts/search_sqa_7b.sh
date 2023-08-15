CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 --master_port 11111 train.py \
    --llm_model 7B\
    --llama_model_path ../data/weights/ \
    --data_path ../data/alpaca_data.json \
    --max_seq_len 512 \
    --batch_size 8 \
    --accum_iter 2 \
    --epochs 2 \
    --warmup_epochs 1 \
    --blr 1e-2 \
    --weight_decay 0.04 \
    --output_dir ./LaVIN-7B/\
    --adapter_type attn\
    --adapter_dim 8\
    --adapter_scale 1\
    --n_prompt 7 \
    --prompt_format QCM-ALE \
    --temperature 10.\
    --visual_adapter_type router \
    --kab_app