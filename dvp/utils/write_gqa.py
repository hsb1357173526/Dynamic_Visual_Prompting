import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from .glossary import normalize_word
from .ans_punct import prep_ans




def path2rest(path, split, annotations, label2ans):
    iid = path.split("/")[-1][:-4]

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    questions = [qa[0] for qa in qas]
    answers = [qa[1] for qa in qas]
    answer_labels = ([a["labels"] for a in answers])
    answer_scores = ([a["scores"] for a in answers])
    answers = ([[label2ans[str(l)] for l in al] for al in answer_labels])

    return [binary, questions, answers, answer_labels, answer_scores, iid, qids, split]



def ans_stat(json_file):
    ans_to_ix,ix_to_ans = json.load(open(json_file, 'r'))[:2]
    return ans_to_ix,ix_to_ans


def make_arrow(root, dataset_root):
    with open(f"{root}/train_balanced_questions.json", "r") as fp:
        questions_train = json.load(fp)
    with open(f"{root}/testdev_balanced_questions.json", "r") as fp:
        questions_test_dev = json.load(fp)
    with open(f"{root}/val_balanced_questions.json", "r") as fp:
        questions_test_val = json.load(fp)

    '''
    "image","questions","answers","answer_labels","answer_scores","image_id","question_id","split",
    
    train_df = []
    for id,train_info in questions_train.items():
        question_id = id
        split = 'train'
        question = train_info['question']
        image_id = train_info['imageId']
        path = './images/' + image_id + '.jpg'
        answer = train_info['answer']
        answer = prep_ans(answer)
        answer_label = ans_to_ix[answer]
        train_df.append(path2rest(split,question_id,question,path,image_id,answer,answer_label))
    '''
    ans2label,label2ans = ans_stat('./dicts.json')
    annotations = dict()

    for split, questions in zip(
        ["train","test-dev",'val'],
        [
            questions_train,
            questions_test_dev,
            questions_test_val,
        ],
    ):
        _annot = defaultdict(dict)
        for q_id, q in questions.items():
            _annot[q["imageId"]][q_id] = [q["question"]]
            answer = prep_ans(q['answer'])
            labels = []
            scores = []
            if answer in ans2label.keys():
                labels.append(ans2label[answer])
                scores.append(1.0)
            _annot[q["imageId"]][q_id].append({"labels": labels, "scores": scores,})

        annotations[split] = _annot

    '''
    all_major_answers = list()

    for split, annots in zip(
        ["train", "val"], [annotations_train2014, annotations_val2014],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            all_major_answers.append(q["multiple_choice_answer"])

    all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    ans2label = {k: i for i, k in enumerate(counter.keys())}
    label2ans = list(counter.keys())
    '''


    for split in ["train", "test-dev",'val']:
        filtered_annot = dict()
        for ik, iv in annotations[split].items():
            new_q = dict()
            for qk, qv in iv.items():
                if len(qv[1]["labels"]) != 0:
                    new_q[qk] = qv
            if len(new_q) != 0:
                filtered_annot[ik] = new_q
        annotations[split] = filtered_annot

    for split in [
        "train",
        "test-dev",
        'val',
    ]:
        annot = annotations[split]
        paths = list(glob(f"{root}/images/*.jpg"))
        random.shuffle(paths)
        annot_paths = [
            path
            for path in paths
            if path.split("/")[-1][:-4] in annot
        ]

        '''
        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")           
        '''

        print(
            len(paths), len(annot_paths), len(annot),
        )

        bs = [
            path2rest(path, split, annotations, label2ans) for path in tqdm(annot_paths)
        ]

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/gqa_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
