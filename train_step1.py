import argparse
from collections import deque
import os

from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import random
from dataset_proc import PassageIdsDataset


seed = 42


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="experiment_outputs/")
    parser.add_argument('--backbone_model_path', type=str, default="ptm/flan-t5-small")
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--lr', type=int, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=36)
    parser.add_argument('--num_devset', type=int, default=500)
    parser.add_argument('--topK', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_gpu', default=2)
    return parser.parse_args()

args_parser = argument_parser()


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


torch.manual_seed(args_parser.seed)
random.seed(args_parser.seed)
torch.cuda.manual_seed_all(args_parser.seed)


local_rank = int(os.environ["LOCAL_RANK"])
init_ddp(local_rank)

trn = PassageIdsDataset()
trn_sampler = DistributedSampler(trn, shuffle=False)
trnloader = DataLoader(
    dataset=trn,
    batch_size=args_parser.batch_size,
    # shuffle=False,
    num_workers=2,
    pin_memory=True,
    sampler=trn_sampler
)

dev_sampler = DistributedSampler(trn, shuffle=True)
devloader = DataLoader(
    dataset=trn,
    # shuffle=True,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
    sampler=dev_sampler
)

tst = PassageIdsDataset(
    data_path="data_concerned/datasets/nq/passage_id_pairs_test.json",
    old2new_mapper_path="data_concerned/datasets/nq/old2new_id_mapper_test_st5-base_k10_c100.pkl"
)
tstloader = DataLoader(dataset=tst, batch_size=args_parser.batch_size)


t5_model = T5ForConditionalGeneration.from_pretrained(args_parser.backbone_model_path).cuda()
# DataParallel
t5_model = nn.parallel.DistributedDataParallel(t5_model, device_ids=[local_rank])
t5_tokenizer = AutoTokenizer.from_pretrained(args_parser.backbone_model_path)


optimizer = torch.optim.Adam(t5_model.parameters(), lr=args_parser.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=1, gamma=0.9, verbose=True)


moving_avg_losses = deque(maxlen=40)
trn_step = 0
num_steps_per_epoch = len(trn) // (args_parser.batch_size * args_parser.num_gpu) + 1
num_steps_overall = args_parser.num_epoch * num_steps_per_epoch
for i_epoch in range(1, args_parser.num_epoch + 1):
    t5_model.train()
    for data in trnloader:
        
        trn_step += 1
        optimizer.zero_grad()
        ids, psgs = data
        inputs = t5_tokenizer(psgs, max_length=512, padding=True, truncation=True, return_tensors="pt").input_ids.cuda()
        targets = t5_tokenizer(ids, max_length=50, padding=True, truncation=True, return_tensors="pt").input_ids.cuda()
        outputs = t5_model(input_ids=inputs, labels=targets)
        loss = torch.mean(outputs.loss)
        
        temp_loss = loss.item()
        moving_avg_losses.append(temp_loss)
        avg_loss = sum(moving_avg_losses) / len(moving_avg_losses)
        if local_rank == 0 and trn_step % 30 == 0:
            print(f"""Epoch:{i_epoch} temploss:{temp_loss:.04} avgloss:{avg_loss:.04} step No.{trn_step - num_steps_per_epoch*(i_epoch-1)}/{num_steps_per_epoch}(current epoch) No.{trn_step}/{num_steps_overall}(whole epoch)""")
        
        loss.backward()
        optimizer.step()

        # evaluate every 1000 steps
        if trn_step % 3000 == 0 and local_rank == 0:
            t5_model.eval()
            print('start evaluating...')
            all_labels = []
            all_predicts = []
            print('generating...')
            iter_dev = iter(devloader)
            for dev_step in trange(args_parser.num_devset):
                dev_data = next(iter_dev)
                ids, psgs = dev_data
                input_ids = t5_tokenizer.encode(psgs[0], return_tensors='pt').cuda()
                output_ids = t5_model.module.generate(inputs=input_ids, num_return_sequences=args_parser.topK, num_beams=args_parser.topK)
                cur_predict = []
                for i in range(output_ids.shape[0]):
                    output_text = t5_tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    cur_predict.append(output_text)
                all_predicts.append(cur_predict)
                all_labels.append(ids[0])
            print('calculating...')
            hits = 0
            for j in range(len(all_labels)):
                if all_labels[j] in set(all_predicts[j]):
                    hits += 1
            hits_score = hits / len(all_labels)
            print(f"hit@{args_parser.topK}: {hits_score*100:.04}%")
            t5_model.train()


    scheduler.step()



print("done")