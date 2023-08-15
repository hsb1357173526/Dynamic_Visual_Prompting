# coding=utf-8
# Copyright 2022 Gen Luo. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import  json, re,random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from util.base_prompt import *
import torch
from lavin import Tokenizer
import copy

class ScienceQADataSet(Data.Dataset):
    def __init__(self, args,split,model_path,max_words=512,max_image_feats=1):
        super(ScienceQADataSet, self).__init__()
        self.args = args
        # --------------------------
        # ---- Raw data loading ---
        # --------------------------
        self.problems = json.load(open(os.path.join(args.data_root, 'problems.json')))
        pid_splits = json.load(open(os.path.join(args.data_root, 'pid_splits.json')))
        captions = json.load(open(args.caption_file))["captions"]
        self.image_path=os.path.join(args.data_root,'images',split)
        self.tokenizer = Tokenizer(model_path=model_path + '/tokenizer.model')
        self.max_words = max_words
        self.max_image_feats=max_image_feats
        self.split=split
        for qid in self.problems:
            self.problems[qid]['caption'] = captions[qid] if qid in captions else ""

        self.qids = pid_splits['%s' % (split)]

        print(f"number of problems in split {split}: {len(self.qids)}\n")

        self.transforms=transforms.Compose([transforms.Resize((224, 224), interpolation=Image.BICUBIC),transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])

    def tokenize(self,prompt,answer):
        example=prompt+answer
        # print(prompt)
        prompt=torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.int64)
        example = torch.tensor(self.tokenizer.encode(example, bos=True, eos=True), dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[:self.max_words]
        labels = copy.deepcopy(example)
        labels[:len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        # 确定每个prompt的长度（除掉bos_token）
        prompt_len = len(prompt) - 1
        return example, labels, example_mask,label_mask, prompt_len


    def __getitem__(self, idx):

        prompt_question,prompt_answer= build_prompt(self.problems,self.qids[idx],self.args)

        if self.problems[self.qids[idx]]['image'] is not None:
            image = Image.open(os.path.join(self.image_path, self.qids[idx], 'image.png')).convert('RGB')
            image = self.transforms(image)
            indicator=1
        else:
            image=torch.Tensor(torch.zeros(3,224,224).float())
            indicator=0

        example, labels, example_mask, label_mask, prompt_len = self.tokenize(prompt_question,prompt_answer)

        return example, labels, example_mask, image, indicator, prompt_len

    def __len__(self):
        return len(self.qids)

    def shuffle_list(self, list):
        random.shuffle(list)






