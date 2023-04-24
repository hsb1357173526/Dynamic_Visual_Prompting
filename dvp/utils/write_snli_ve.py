import json
import pandas as pd
import pyarrow as pa
import os
import base64

from tqdm import tqdm
from collections import defaultdict

def process(iden, data):
    texts = [d[3] for d in data]
    labels = [d[5] for d in data]
    img = base64.b64decode(data[0][2])
    return [img, texts, labels, iden]


def make_arrow(root, dataset_root):
    train_data = pd.read_csv(f"{root}/snli_ve_train.tsv",sep='\t',header=None)
    train_data = train_data[train_data[3].notnull()]
    train_data.index = range(len(train_data))

    dev_data = pd.read_csv(f"{root}/snli_ve_dev.tsv", sep='\t', header=None)
    dev_data = dev_data[dev_data[3].notnull()]
    dev_data.index = range(len(dev_data))

    test_data = pd.read_csv(f"{root}/snli_ve_test.tsv", sep='\t', header=None)
    test_data = test_data[test_data[3].notnull()]
    test_data.index = range(len(test_data))

    splits = ["train","dev","test"]
    datas = [train_data,dev_data,test_data]

    annotations = dict()

    for split, data in zip(splits, datas):
        _annot = defaultdict(list)
        for i in range(len(data)):
            _annot[data.loc[i][1]].append(data.loc[i])
        annotations[split] = _annot

    for split in splits:
        bs = [
            process(iden, data) for iden, data in tqdm(annotations[split].items())
        ]


        dataframe = pd.DataFrame(
            bs, columns=["image","questions", "answers", "identifier"],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/snli_ve_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)