from torch.utils.data import Dataset, DataLoader
import torch
import json
import pickle
from random import choice
from tqdm import tqdm

from data_helper import convert_semanticID_to_text

class PassageIdsDataset(Dataset):
    """The Dataset class for training (step.1 psg -> id)
       This class contains pure psg-id pairs data for training.
    """
    def __init__(self,
                 data_path="data_concerned/datasets/nq/passage_id_pairs_train.json",
                 old2new_mapper_path="data_concerned/datasets/nq/old2new_id_mapper_train_st5-base_k10_c100.pkl") -> None:
        super().__init__()
        with open(data_path, "r") as f1:
            psg_id_pairs = json.load(f1)
        with open(old2new_mapper_path, "rb") as f2:
            old2new_mapper = pickle.load(f2)
        self.passages = []
        self.ids = []

        for single_data in tqdm(psg_id_pairs):
            self.ids.append(old2new_mapper[single_data["passage_id"]])
            self.passages.append(single_data["text"])

    def __getitem__(self, index):
        return self.ids[index], self.passages[index]
    
    def __len__(self):
        return len(self.ids)
        




class PassageGeneratedQueryDataset(Dataset):
    """The Dataset class for training (step.2 query -> id)
       This class contains both passage and query for seq2seq semantic
       id predicting training and psg contrastive training. This class
       will generate data in the sequence like:
       [old_id, semantic_id, passage, generated_query]

    Arguments:
        data_path: the path of file storing the processed psg_id_gq.json.
        mode: whether "train" or "valid".
        train_ratio: the ratio of trainset in the whole dataset.
        id_mapper_path: the path of file containing the old2new id mapper.
    """
    def __init__(self,
                 data_path="data_concerned/datasets/nq/passage_id_gq_for_train.json",
                 mode=None,
                 train_ratio=0.9,
                 id_mapper_path="data_concerned/datasets/nq/ID_MAPPER_trn_st5-base_k10_c100.pkl") -> None:
        super().__init__()
        
        try:
            assert mode in ["train", "valid"]
            self.mode = mode
        except AssertionError:
            print("The dataset mode is not provided, it is going to be set in train mode")
            self.mode = "train"

        with open(data_path, "r") as f:
            all_data = json.load(f)
        
        self.all_old_ids = []
        self.all_semantic_ids = []
        self.all_passages = []
        self.all_generated_querys = []

        # load old2new id mapper
        with open(id_mapper_path, "rb") as f_mapper:
            id_mapper = pickle.load(f_mapper)
        
        print("start reading passage-id-gq data...")
        for data_i in all_data:
            self.all_old_ids.append(data_i["passage_id"])
            self.all_semantic_ids.append(convert_semanticID_to_text(id_mapper[data_i["passage_id"]]))
            self.all_passages.append(data_i["text"])
            self.all_generated_querys.append(choice(data_i["generated_querys"]))

        self.tv_split = int(len(self.all_old_ids) * train_ratio)

    def __getitem__(self, index):
        
        if self.mode == "train":
            pass
        elif self.mode == "valid":
            index += self.tv_split
        
        return self.all_old_ids[index],      \
                self.all_semantic_ids[index], \
                self.all_passages[index],     \
                self.all_generated_querys[index]
    
    def __len__(self):
        if self.mode == "train":
            return self.tv_split
        elif self.mode == "valid":
            return len(self.all_old_ids) - self.tv_split
    
    # @staticmethod
    # def collate_fn(batch):


if __name__ == "__main__":
    ppd = PassageIdsDataset()
    print("done")