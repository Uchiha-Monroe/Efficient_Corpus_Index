

import os

os.environ["OMP_NUM_THREADS"] = '4'

from typing import List
import pickle
import argparse
from tqdm import tqdm, trange

import json


import numpy as np
import pandas as pd

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.cluster import KMeans, MiniBatchKMeans

import torch
import torch.nn as nn

from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def text2emb_by_sentence_transformer(
        text: str,
        model
) -> np.array:
    """Generate sentence representation by sentence_transformers library
    """
    return model.encode(text)


def text2emb_by_hftransformer(texts: str,
                              model: nn.Module,
                              tokenizer,
                              pooling: str = 'mean') -> torch.Tensor:
    """Convert text to embeddings, using the specified model

    :texts: text needs to be embedded
    :model: model used to embed texts, from Huggingface Transformers lib by default
    :tokenizer: tokenizer used by model, usually from Transformers lib
    :pooling: pooling strategy
    """
    if not pooling in ["cls", "mean", "max", "last2mean"]:
        print("wrong pooling strategy is provided, using mean pooling strategy by default.")
        pooling = "mean"

    if pooling == "mean":
        inputs = tokenizer(texts, return_tensors="pt")
        outputs = model(**inputs)
        return outputs.last_hidden_state.detach().cpu().numpy().mean()


def prepro_nq(save_to_path: bool = False):
    """Collect unique passages from the Natural Question dataset with
        'question-answer-context' format.
       Convert nq-dev.json and nq-train.json -> passage_id_pairs_all.json

    the format dataset nq-dev.json and nq-train.json is:
    
    List[
        {
            "dataset": "nq_dev_psgs_w100",
            "question": ...,
            "answers":[...],
            "positive_ctxs":[
                {
                    "title":...,
                    "text":...,
                    "score":...,
                    "title_score":...,
                    "passage_id":...
                },
                ...
            ],
            "negative_ctxs":[
                {
                    "title": str,
                    "text: str,
                    "score": float,
                    "title_score": float,
                    "passage_id": str
                },
                ...
            ],
            "hard_negative_ctxs":[
                {
                    "title": str,
                    "text: str,
                    "score": float,
                    "title_score": float,
                    "passage_id": str
                }
            ]
        },
        {
            ...
        }
    ]

    the generated data is passage_id_pairs_all:
    
    List[
      {"passage_id": ...,
       "text": ...},
      {"passage_id": ...,
       "text": ...},
       ...
    ]
    
    """
    trn_path = "data_concerned/datasets/nq/nq-train.json"
    dev_path = "data_concerned/datasets/nq/nq-dev.json"

    all_passage_id = set()
    passage_id_pairs_all = []

    def _pro_fn(raw_data):
        for data in tqdm(raw_data):
            for pos_data in data["positive_ctxs"]:
                if pos_data["passage_id"] not in all_passage_id:
                    all_passage_id.add(pos_data["passage_id"])
                    cur_data_pair = {"passage_id": pos_data["passage_id"], "text": pos_data["text"]}
                    passage_id_pairs_all.append(cur_data_pair)
            for neg_data in data["negative_ctxs"]:
                if neg_data["passage_id"] not in all_passage_id:
                    all_passage_id.add(neg_data["passage_id"])
                    cur_data_pair = {"passage_id": neg_data["passage_id"], "text": neg_data["text"]}
                    passage_id_pairs_all.append(cur_data_pair)
            for hard_neg_data in data["hard_negative_ctxs"]:
                if hard_neg_data["passage_id"] not in all_passage_id:
                    all_passage_id.add(hard_neg_data["passage_id"])
                    cur_data_pair = {"passage_id": hard_neg_data["passage_id"], "text": hard_neg_data["text"]}
                    passage_id_pairs_all.append(cur_data_pair)

    print('start preprocessing dev set...')
    with open(dev_path, "r") as fd:
        dev_data = json.load(fd)
    _pro_fn(dev_data)

    print('start preprocessing train set...')
    with open(trn_path, 'r') as ft:
        trn_data = json.load(ft)
    _pro_fn(trn_data)

    print(f'done. totally {len(passage_id_pairs_all)} unique passages.')

    if save_to_path:
        with open("data_concerned/datasets/nq/passage_id_pairs_all.json", "w") as f:
            json.dump(passage_id_pairs_all, f, indent=4)


def passage_embedder(
        data_path,
        split:str,
        embed_fn,
        model_path,
        model_name,
        save_dir='data_concerned/datasets/nq/',
        runtime_whitening=False,
        is_parallelization=None,
        bsz=128
):
    """Convert passage_id_pairs.json -> passage_embeddings.npy

    Args:
        data_path(str): path of passage_id_pairs.json-like files
        split(str): indicate the which part of the data is used. e.g. "train", "test", "all"
        embed_fn(callable): function transforming str to numpy array
        model_path(str): path of emb model
        model_name(str): name of emb model
        save_dir(str): path of output file
        runtime_whitening(bool): whether using runtime-whitening strategy or not
    """
    print('reading data...')
    with open(data_path, 'r') as f:
        passages_ids_pair = json.load(f)

    all_psg_emb = []
    model = SentenceTransformer(model_path).to('cuda')

    print('start converting passages to embeddings...')
    if is_parallelization:
        passages_list = [entry["text"] for entry in passages_ids_pair]
        new_ind_list = list(range(0, len(passages_ids_pair), bsz))
        for ind in tqdm(new_ind_list):
            sentences = passages_list[ind: ind+bsz]
            embs = embed_fn(sentences, model)
            all_psg_emb.append(embs)
    else:
        for ind, psg_id in tqdm(enumerate(passages_ids_pair), total=len(passages_ids_pair)):
            emb = embed_fn(psg_id["text"], model)
            all_psg_emb.append(emb)

            # compute mu and sigma at each step iteratively
            if runtime_whitening:
                n = len(passages_ids_pair)
                if ind == 0:
                    mu = emb
                    sigma = np.zeros((emb.shape[0], emb.shape[0]))
                else:
                    mu = (n / (n + 1)) * mu + (1 / (n + 1)) * emb
                    sigma = (n / (n + 1)) * (sigma + mu.T @ mu) + (1 / (n + 1)) * (emb.T @ emb) - (mu.T @ mu)

    all_psg_emb_np = np.vstack(all_psg_emb)

    if runtime_whitening:
        all_psg_emb_np = (all_psg_emb_np + mu) @ sigma

    save_path = save_dir + 'passage_embeddings2_' + split + '_' + model_name + '.npy'
    with open(save_path, 'wb') as f:
        np.save(f, all_psg_emb_np)


def compute_kernel_bias_from_allemb(embs: np.ndarray, n_components=256):
    """conpute kernel and bias of whitening sentence representation
    
    Args:
    embs(numpy.array) [num_samples, embedding_size]: the embedding 
        representations of all passages
    n_components(int): choose the most important n_components of the 
        original representation, aka dimensionality reduction
    
    return:
      W: kernel
      -mu: bias

      y = (x + bias).dot(kernel)
    """
    print("start SVD decomposing...")
    mu = embs.mean(axis=0, keepdims=True)
    cov = np.cov(embs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))

    return W[:, :n_components], -mu


def matrix_whitener(matrix: np.ndarray):
    """convert a matrix to the whitened version.

    :matrix [num_samples, embedding_size]:
    """
    kernel, bias = compute_kernel_bias_from_allemb(matrix)
    return (matrix + bias) @ kernel


def extract_old_ids(
        passage_id_file_path="data_concerned/datasets/nq/passage_id_pairs_all.json",
        save2disk=False,
        train_test_ratio=0.9
):
    """This function extracts the old passage_id with the sequence of
       *passage_id_pairs.json, and can split the whole ids into train
       and test disposition with the given train_test_ratio.

    Args:
        train_test_ratio(float): ratio of train set to all data. Note
        that this value needs to be consistent with other functions
        possessing the same argument.
    
    """
    with open(passage_id_file_path, 'r') as f:
        psg_id_data = json.load(f)

    all_old_ids = []
    for psg_id in tqdm(psg_id_data):
        all_old_ids.append(psg_id["passage_id"])

    split_divd = int(len(all_old_ids) * train_test_ratio)
    old_ids_list_trn = all_old_ids[: split_divd]
    old_ids_list_test = all_old_ids[split_divd:]

    if save2disk:
        with open("data_concerned/datasets/nq/old_ids_list_all.pkl", "wb") as f:
            pickle.dump(all_old_ids, f)
        with open("data_concerned/datasets/nq/old_ids_list_train.pkl", "wb") as f:
            pickle.dump(old_ids_list_trn, f)
        with open("data_concerned/datasets/nq/old_ids_list_test.pkl", "wb") as f:
            pickle.dump(old_ids_list_test, f)


def divide_psg_id_pairs(
        passage_id_file_path="data_concerned/datasets/nq/passage_id_pairs_all.json",
        save_dir="data_concerned/datasets/nq/",
        train_test_ratio=0.9
):
    """Divide the whole all passage_id_pairs.json file to train and test disposition

    Args:
        train_test_ratio(float): ratio of train set to all data 
    """
    with open(passage_id_file_path, 'r') as f:
        psg_id_data = json.load(f)

    divd = int(len(psg_id_data) * train_test_ratio)
    psg_id_data_trn = psg_id_data[: divd]
    psg_id_data_test = psg_id_data[divd:]

    with open(save_dir + "passage_id_pairs_train.json", "w") as f:
        json.dump(psg_id_data_trn, f)
    with open(save_dir + "passage_id_pairs_test.json", "w") as f:
        json.dump(psg_id_data_test, f)


def generate_query_from_doc(
        model,
        tokenizer,
        doc_text,
        input_max_length=300,
        output_max_length=50,
        top_p=0.95,
        num_generated_query=3,
        device=DEVICE
):
    """Generate a few querys according to the input document. Used for data augmentation
    """
    input_ids = tokenizer(doc_text,
                          max_length=input_max_length,
                          truncation=True,
                          padding=True,
                          return_tensors="pt")["input_ids"].to(device)
    outputs = model.generate(input_ids=input_ids,
                             max_length=output_max_length,
                             do_sample=True,
                             top_p=top_p,
                             num_return_sequences=num_generated_query)
    generated_querys = []
    for i in range(outputs.shape[0]):
        generated_querys.append(tokenizer.decode(outputs[i], skip_special_tokens=True))
    return generated_querys


def generate_querys_from_stream(
        passage_file_path="data_concerned/datasets/nq/passage_id_pairs_train.json",
        gq_model_path="ptm/doc2query-msmarco-t5-small-v1",
        device=DEVICE,
        bsz=32
):
    d2q_model = T5ForConditionalGeneration.from_pretrained(gq_model_path).to(device)
    d2q_tokenizer = T5Tokenizer.from_pretrained(gq_model_path)

    psg_id_gq_triple_list = []
    all_generated_queries = []
    with open(passage_file_path, "r") as f_gq:
        psg_id_pairs_list = json.load(f_gq)
        new_ind = list(range(0, len(psg_id_pairs_list), bsz))
        
        all_text_list = []
        print(f'caching all text...')
        for data in tqdm(psg_id_pairs_list):
            all_text_list.append(data["text"])

        for ind in tqdm(new_ind):
            ind_data = psg_id_pairs_list[ind: ind+bsz]
            generated_queries = generate_query_from_doc(
                model=d2q_model,
                tokenizer=d2q_tokenizer,
                doc_text=all_text_list[ind: ind+bsz]
            )
            all_generated_queries.extend(generated_queries)
            psg_id_gq_triple_list.extend(ind_data)
        
        for i, s_data in enumerate(psg_id_gq_triple_list):
            s_data["generated_queries"] = all_generated_queries[i: i+3] # 3 is the num of generated queries


        # for data in tqdm(psg_id_pairs_list):
        #     new_data = {}
        #     new_data["passage_id"] = data["passage_id"]
        #     new_data["text"] = data["text"]
        #     new_data["generated_querys"] = generate_query_from_doc(
        #         model=d2q_model,
        #         tokenizer=d2q_tokenizer,
        #         doc_text=data["text"]
        #     )
        #     psg_id_gq_triple_list.append(new_data)
        with open("data_concerned/datasets/nq/passage_id_gq_train.json", "w") as fw:
            json.dump(psg_id_gq_triple_list, fw, indent=4)


def convert_semanticID_to_text(hie_id: List[int]) -> str:
    """Convert the generated hierarchical semantic id to text for training
       eg. [2, 5, 9, 5, 52] --> "2 5 9 5 52"
    
    Arguments:
      :hie_id - List[int]: the hierarchical semantic id for a single passage
    return:
      :str_id - str: the id with str format 
    """
    str_id_cand = []
    for sem_i in hie_id:
        str_id_cand.append(str(sem_i))
    return " ".join(str_id_cand)


def generate_data_for_training_from_original_nq(
        original_train_data: str="data_concerned/datasets/nq/nq-train.json",
        generated_train_data: str=None,
        output_dir: str=None,
        old2new: str=None,
        new2old: str=None,
        split: str="train"
):
    """(Deprecated)
    Generate data for structured semantic ids generation training.
    Since there are two different formats of data, namely the original
    data from the NQ dataset and the data generate by data augmentation.
    
    Args:
        original_train_data(str): path of original Natural Questions
        dataset
        generated_train_data(str): path of generated querys (data
        augmentation) dataset
        output_dir(str): path to save the organized dataset
        old2new(str): path of the old2new_id_mapper
        new2old(str): path of the new2old_id_mapper
    """
    with open(old2new, "rb") as f:
        old2new_id_mapper = pickle.load(f)

    # construct data from original NQ dataset
    organized_data = []
    with open(original_train_data, "r") as f:
        raw_data = json.load(f)
    for data in tqdm(raw_data):
        temp_data = {
            "question": data["question"],
            "positive_new_ids": [],
            "negative_new_ids": [],
            "hard_negative_new_ids": []
        }
        for pos_data in data["positive_ctxs"]:
            if not pos_data["passage_id"] in old2new_id_mapper:
                continue
            temp_data["positive_new_ids"].append(old2new_id_mapper[pos_data["passage_id"]])
        for neg_data in data["negative_ctxs"]:
            if not neg_data["passage_id"] in old2new_id_mapper:
                continue
            temp_data["negative_new_ids"].append(old2new_id_mapper[neg_data["passage_id"]])
        for hard_neg_data in data["hard_negative_ctxs"]:
            if not hard_neg_data["passage_id"] in old2new_id_mapper:
                continue
            temp_data["hard_negative_new_ids"].append(old2new_id_mapper[hard_neg_data["passage_id"]])
        organized_data.append(temp_data)
    
    # save the organized training data to disk
    save_path = "data_concerned/datasets/nq/" + "organized_nq_" + split + ".pkl"
    with open(save_path, "wb") as f:
        pickle.dump(organized_data, f)
    

def organize_data4training_from_pairs(
        path_passage_id_pairs,
        if_use_data_augmentation,
        old2new: str=None,
        split: str="train"
):
    """(Deprecated)
    Organize data for training (passage -> id)
    """
    with open(old2new, "rb") as f:
        old2new_id_mapper = pickle.load(f)
    
    organized_data = []
    with open(path_passage_id_pairs, 'r') as f:
        psg_id_pairs = json.load(f)
    for data in tqdm(psg_id_pairs):
        pass



if __name__ == "__main__":

    ###### test zone ######
    
    # print('done')
    ###### test zone ###### 

    # step 1. preprocess the question-answer-context data into unique passages and save to disk
    # prepro_nq(True)
    # divide_psg_id_pairs()
    # extract_old_ids(save2disk=True)

    # step 2. convert the passage from text to embeding
    # passages_path = 'data_concerned/datasets/nq/passage_id_pairs_train.json'
    # embedder_path = 'ptm/sentence-t5-base'
    # passage_embedder(
    #     data_path=passages_path,
    #     split="train",
    #     embed_fn=text2emb_by_sentence_transformer,
    #     model_path=embedder_path,
    #     model_name='sentence-t5-base',
    #     runtime_whitening=False,
    #     is_parallelization=True
    # )

    # step 3. compute semantic sturcture id for each psg
    def create_structured_semantic_id(
            psg_emb_path: str="data_concerned/datasets/nq/passage_embeddings_train_sentence-t5-base.npy",
            k: int=10,
            c: int=100,
            emb_dim: int=768,
            cluster_bsz: int=int(1e3),
    ):
        """Create the structured semantic ids from data representation
        
        Args:
            psg_emb_path(str): the path of data representation. It shoule
                be loaded as a numpy.array format
            k(int): the number of clusters in clustering process
            c(int): the maximum number of leaf cluster
            emb_dim(int): dim of data representation
            cluster_bsz(int): batch size of KMeans clustering
        """
        
        # load passage_embedding npy file
        with open(psg_emb_path, 'rb') as f:
            X = np.load(f)

        kmeans = KMeans(
            n_clusters=k,
            max_iter=500,
            n_init=100,
            init='k-means++',
            tol=1e-6,
            verbose=20
        )

        mini_kmeans = MiniBatchKMeans(
            n_clusters=k,
            max_iter=300,
            n_init=100,
            init='k-means++',
            batch_size=cluster_bsz,
            reassignment_ratio=0.01,
            max_no_improvement=50,
            tol=1e-7,
            verbose=1
        )

        # use a list to store generated structured semantic ids
        semantic_id_list = []

        def classify_recursion(x_data_pos):
            if x_data_pos.shape[0] <= c:
                if x_data_pos.shape[0] == 1:
                    return
                for idx, pos in enumerate(x_data_pos):
                    semantic_id_list[pos].append(idx)
                return

            temp_data = np.zeros((x_data_pos.shape[0], emb_dim))
            for idx, pos in enumerate(x_data_pos):
                temp_data[idx, :] = X[pos]

            if x_data_pos.shape[0] >= cluster_bsz:
                pred = mini_kmeans.fit_predict(temp_data)
            else:
                pred = kmeans.fit_predict(temp_data)

            for i in range(k):
                pos_lists = []
                for id_, class_ in enumerate(pred):
                    if class_ == i:
                        pos_lists.append(x_data_pos[id_])
                        semantic_id_list[x_data_pos[id_]].append(i)
                classify_recursion(np.array(pos_lists))
            return

        print('Start First Clustering')
        pred = kmeans.fit_predict(X)
        print(pred.shape)  # int 0-9 for each vector
        print(kmeans.n_iter_)

        for class_ in pred:
            semantic_id_list.append([class_])

        print('Start Recursively Clustering...')
        for i in range(k):
            print(i, "th cluster")
            pos_lists = []
            for id_, class_ in enumerate(pred):
                if class_ == i:
                    pos_lists.append(id_)
            classify_recursion(np.array(pos_lists))
        print('Complete!')
        
        # This process is very time-consuming, so it will be better to save the
        # intermediate result to disk
        with open("data_concerned/datasets/nq/new_ids_list_train.pkl", "wb") as f:
            pickle.dump(semantic_id_list, f)

        old2new_id_mapper = {}
        new2old_id_mapper = {}
        
        # load old id
        with open("data_concerned/datasets/nq/old_ids_list_train.pkl", "rb") as f:
            old_id_list = pickle.load(f)
        # load new id
        with open("data_concerned/datasets/nq/new_ids_list_train.pkl", "rb") as f:
            new_id_list = pickle.load(f)

        # generate old2new_id_mapper
        for i in range(len(old_id_list)):
            old2new_id_mapper[old_id_list[i]] = convert_semanticID_to_text(new_id_list[i])
        # generate new2old_id_mapper
        for old, new in old2new_id_mapper.items():
            new2old_id_mapper[new] = old

        # save to disk
        with open(f"data_concerned/datasets/nq/old2new_id_mapper_train_st5-base_k{k}_c{c}.pkl", "wb") as f:
            pickle.dump(old2new_id_mapper, f)
        with open(f"data_concerned/datasets/nq/new2old_id_mapper_train_st5-base_k{k}_c{c}.pkl", "wb") as f:
            pickle.dump(new2old_id_mapper, f)

    # create_structured_semantic_id()

    # step 4. Generate querys from documents
    # generate_querys_from_stream()

    # step 5. organize data for training
    # generate_data_for_training_from_original_nq(
    #     old2new='data_concerned/datasets/nq/old2new_id_mapper_train_st5-base_k10_c100.pkl',
    #     new2old='data_concerned/datasets/nq/new2old_id_mapper_train_st5-base_k10_c100.pkl'
    # )

    print('done')
