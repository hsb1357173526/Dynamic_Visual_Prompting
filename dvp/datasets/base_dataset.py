import random
import torch
import io
import pyarrow as pa
import os

from PIL import Image
from dvp.transforms import keys_to_transforms
from transformers import BertTokenizer, T5Tokenizer, LlamaTokenizer

class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        transform_keys: list,
        image_size: int,
        tokenizer: str,
        names: list,
        text_column_name: str = "",
        remove_duplicate=True,
        max_text_len=16,
    ):


        assert len(transform_keys) >= 1
        super().__init__()

        self.transforms = keys_to_transforms(transform_keys, size=image_size)
        self.text_column_name = text_column_name
        self.names = names
        self.max_text_len = max_text_len
        self.data_dir = data_dir

        if 'bert' in tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer, do_lower_case=True)
            self.use_model = 'bert'

        elif 't5' in tokenizer:
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer, do_lower_case=False)
            self.use_model = 't5'

        elif 'llama' in tokenizer:
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.tokenizer.pad_token = '[PAD]'
            self.use_model = 'llama'


        else:
            raise NotImplementedError("no found tokenizer")


        if len(names) != 0:
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        # index_mapper 取出所有问题-图像对，定位第几张图和这张图的第几个问题
        if text_column_name != "" :
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)

    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.index_mapper)

    def get_raw_image(self, index, image_key="image"):
        index, caption_index = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes)

    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }


    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]
        if  self.use_model == 'llama':
            text = text + '</s>'
            return {
                "text": text,
                "img_index": index,
                "cap_index": caption_index,
                "raw_index": raw_index,
            }
        else:
            text_encoding = self.tokenizer(text,
                                           padding="max_length",
                                           truncation=True,
                                           max_length=self.max_text_len)

            return {
                "text": (text,text_encoding),
                "img_index": index,
                "cap_index": caption_index,
                "raw_index": raw_index,
            }


    def collate(self, batch):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]


        for img_key in img_keys:
            new_images = torch.zeros(batch_size, 3, 224, 224)
            for i in range(len(new_images)):
                new_images[i] = dict_batch[img_key][i][0]
            dict_batch[img_key] = new_images

        if self.use_model == 'llama':
            texts_encoding = self.tokenizer(dict_batch['text'],padding=True,return_tensors='pt')
            text_input_ids = texts_encoding['input_ids'].long()
            text_attention_mask = texts_encoding['attention_mask'].long()
            dict_batch['text_input_ids'] = text_input_ids
            dict_batch['text_attention_mask'] = text_attention_mask

        else:
            text_input_ids = torch.zeros(batch_size, self.max_text_len).long()
            text_attention_mask = torch.zeros(batch_size, self.max_text_len)
            for i in range(batch_size):
                text_input_ids[i] = torch.LongTensor(dict_batch['text'][i][1]['input_ids'])
                text_attention_mask[i] = torch.LongTensor(dict_batch['text'][i][1]['attention_mask'])
            dict_batch['text_input_ids'] = text_input_ids
            dict_batch['text_attention_mask'] = text_attention_mask

            if self.use_model == 'bert':
                text_token_type_ids = torch.zeros(batch_size, self.max_text_len).long()
                for i in range(batch_size):
                    text_token_type_ids[i] = torch.LongTensor(dict_batch['text'][i][1]['token_type_ids'])
                dict_batch['text_token_type_ids'] = text_token_type_ids

            dict_batch['text'] = [d[0] for d in dict_batch['text']]

        return dict_batch
