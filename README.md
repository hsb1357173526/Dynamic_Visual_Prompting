# Dynamic Visual Prompting

![avatar](./method.png)
---

## Environment Setup
```bash
pip install -r requirements.txt
```

## Dataset Preparation
See [`DATA.md`](./DATA.md)



## KAB-APP
```bash
# ---------- For BERT in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_bert_vqa per_gpu_batchsize=256

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_bert_gqa per_gpu_batchsize=256

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_bert_snli_ve per_gpu_batchsize=256

# ---------- For T5 in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_t5_vqa per_gpu_batchsize=256

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_t5_gqa per_gpu_batchsize=256

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_search_t5_snli_ve per_gpu_batchsize=256
```
* <font color='red'>**Note**</font>: For different PLMs or VL tasks, KAB-APP will finally print out the searched DVP placement `K`, and you need to record `K` for subsequent training.



## Finetuning
```bash
# ---------- For BERT in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_vqa per_gpu_batchsize=256 insert_layer=K

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_gqa per_gpu_batchsize=256 insert_layer=K

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_snli_ve per_gpu_batchsize=256 insert_layer=K

# ---------- For T5 in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_vqa per_gpu_batchsize=256 insert_layer=K

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_gqa per_gpu_batchsize=256 insert_layer=K

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_snli_ve per_gpu_batchsize=256 insert_layer=K
```
* Here, we use the searched result `K` of KAB-APP for inserting.

## Parameter-efficient transfer learning
```bash
# ---------- For BERT in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_vqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=1e-3

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_gqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=5e-4

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_snli_ve per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=5e-4

# ---------- For T5 in  VL tasks ----------
# VQA2.0
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_vqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=3e-4

# GQA
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_gqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=3e-4

# SNLI-VE
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_snli_ve per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=3e-4
```

## Evaluation
```bash
# ---------- For BERT in  VL tasks e.g. VQA2.0----------
# Evaluate fine-funed BERT
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_vqa per_gpu_batchsize=256 insert_layer=K test_only=True precision=32 load_path=<checkpoint path>

# Evaluate BERT with adapter
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_bert_vqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=1e-3 test_only=True precision=32 load_path=<checkpoint path>

# ---------- For T5 in  VL tasks e.g. VQA2.0----------
# Evaluate fine-funed T5
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_vqa per_gpu_batchsize=256 insert_layer=K test_only=True precision=32 load_path=<checkpoint path>

# Evaluate T5 with adapter
CUDA_VISIBLE_DEVICES=0 python run.py with data_root=./datasets num_gpus=1 num_nodes=1 dvp_adaption_t5_vqa per_gpu_batchsize=256 insert_layer=K use_adapter=True learning_rate=3e-4 test_only=True precision=32 load_path=<checkpoint path>
```
* For VQA2.0, the above script will generate `result/vqa_submit_***.json`, you can upload it to eval.ai (https://eval.ai/web/challenges/challenge-page/830/overview) evaluation server to get test-dev score.