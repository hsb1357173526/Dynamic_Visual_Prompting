# Dataset Preparation

---
* We utilize three datsets: Visual Question Answering v2 (VQAv2), Grounding Question Answering (GQA), Stanford Natural Language Inference - Visual Entailment (SNLI-VE).
* We will provide download url, please download the datasets by yourself.
* We use `pyarrow` to serialize the datasets, conversion scripts are located in `dvp/utils/write_*.py`.
* Please organize the datasets as follows and run `make_arrow` functions to convert the dataset to pyarrow binary file.


## VQAv2
* Download url: https://visualqa.org/download.html.
* Download COCO [2014 train images](http://images.cocodataset.org/zips/train2014.zip), [2014 val images](http://images.cocodataset.org/zips/val2014.zip), [2015 test images](http://images.cocodataset.org/zips/test2015.zip), annotations ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip)), and questions ([train](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip), [val](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip), [test](https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip)).


    ./datasets
    ├── train2014            
    │   ├── COCO_train2014_000000000009.jpg                
    |   └── ...
    ├── val2014              
    |   ├── COCO_val2014_000000000042.jpg
    |   └── ...  
    ├── test2015              
    |   ├── COCO_test2015_000000000001.jpg
    |   └── ...         
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    ├── v2_OpenEnded_mscoco_val2014_questions.json
    ├── v2_OpenEnded_mscoco_test2015_questions.json
    ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
    ├── v2_mscoco_train2014_annotations.json
    └── v2_mscoco_val2014_annotations.json

```python
from dvp.utils.write_vqa import make_arrow
make_arrow('./datasets','./datasets')
```

## GQA
* Download url: https://cs.stanford.edu/people/dorarad/gqa/download.html.
* Download [GQA questions](https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip) and [GQA images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip).


    ./datasets
    ├── images          
    │   ├── 1.jpg                  
    │   ├── 2.jpg
    │   └── ...
    ├── train_balanced_questions.json
    ├── train_balanced_questions.json
    └── testdev_balanced_questions.json


```python
from dvp.utils.write_gqa import make_arrow
make_arrow('./datasets', './datasets')
```

## SNLI-VE
* Download url: https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/snli_ve_data/snli_ve_data.zip.
* Follow by [OFA repository](https://github.com/OFA-Sys/OFA/blob/main/datasets.md), we use its ```.tsv``` format of SNLI-VE.


    ./datasets
    ├── snli_ve_train.tsv
    ├── snli_ve_dev.tsv
    └── snli_ve_test.tsv

```python
from dvp.utils.write_snli_ve import make_arrow
make_arrow('./datasets', './datasets')
```