from .base_dataset import BaseDataset

class SNLIVEDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.ans2label = {
            'contradiction':0,
            'neutral':1,
            'entailment':2,
        }
        if split == "train":
            names = ["snli_ve_train"]
        elif split == "val":
            names = ["snli_ve_dev", "snli_ve_test"]
        elif split == "test":
            names = ["snli_ve_dev", "snli_ve_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]
        index, question_index = self.index_mapper[index]
        answers = self.table["answers"][index][question_index].as_py()
        answers = self.ans2label[answers]
        return {
            "image": image_tensor,
            "text": text,
            "answers": answers,
            "table_name": self.table_names[index],
        }
