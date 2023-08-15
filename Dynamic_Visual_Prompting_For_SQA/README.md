# Dynamic Visual Prompting for SQA
[![Python](https://img.shields.io/badge/python-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-%237732a8)

This is the implementation of LLaMA-DVP<sub>*adp*</sub> in ScienceQA of "Adapting Pre-trained Language Models to Vision-Language Tasks via Dynamic Visual Prompting". We follow [LaVIN](https://luogen1996.github.io/lavin/) and put DVP in the image description section of the prompt. 

## Preparation
* We follow the repositories of LaVIN [here](https://github.com/luogen1996/LaVIN/blob/main/README.md) to prepare environments, model weights and datasets.

* The file structure should look like:
```bash
Dynamic_Visual_Prompting_For_SQA/
  |-- lavin
  |-- scripts
  |-- train.py
  |-- eval.py
  ......
data/
  |-- problems.json
  |-- pid_splits.json
  |-- captions.json
  |-- alpaca_data.json
  |-- images
      |-- train          # ScienceQA train image
      |-- val            # ScienceQA val image
      |-- test          # ScienceQA test image
  |-- weights
      |-- tokenizer.model
          |--7B
              |-- params.json
              |-- consolidated.00.pth
      |-- ......
```


# KAB-APP for LLaMA-DVP<sub>*adp*</sub>
```bash
bash ./scripts/search_sqa_7b.sh
```
* After KAB-APP, you should remember the searched insertion layer **K** and use it for fine-tuning.



# Fine-Tuning for LLaMA-DVP<sub>*adp*</sub>
* Before fine-tuning, you need to change the `--insert_layer` of `./scripts/finetuning_sqa_7b.sh` to **K**.
```bash
bash ./scripts/finetuning_sqa_7b.sh
```