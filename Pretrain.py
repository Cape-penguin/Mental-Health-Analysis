import os
import json
import yaml
import time
import random
import numpy as np
import datetime
import logging
import argparse
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss

import scipy.io as sio
from sklearn.metrics import accuracy_score, f1_score

from src.loss.ntxentloss import NTXentLoss
from src.optim import create_optimizer
from src.scheduler import create_scheduler
from src.model.main_model import MyModel
from src.model.IndexSorter import IndexSorter, mini_batch_level_shuffle
from src.dataset import utils
from src.dataset.post_dataset import l2i
from src.dataset.data_handler import create_dataset, create_loader, create_sampler, create_fixed_sampler


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_config(path):
        with open(path, 'r', encoding="utf-8") as f:
            return yaml.safe_load(f)

def cluster_labels_dist(cluster_labels):
    label_counts = Counter(cluster_labels)
    for label, count in sorted(label_counts.items()):
        logging.info(f"Cluster {label}: {count} samples")

def train(model, data_loader, criterion, optimizer, epoch, device, scheduler, config):
    max_epoch = config['epoch']
    model.train()
    logging.info(f'[{epoch}/{max_epoch}] Set model to training mode.')
    logging.info(f'device:{device}')
    
    # init queue
    num_steps=len(data_loader)
    index_sorter=IndexSorter(device, num_steps, config)
    
    ce = CrossEntropyLoss()
    
    # tokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer'])
    
    # load vocabs
    d = {}
    with open(config['keywords'], 'r', encoding='utf-8') as f:
        word_dict=json.load(f)
    for l, i in l2i.items():
        d[i]=torch.tensor(word_dict[l], dtype=torch.int32)

    total_cluster_labels = []
    metric_acc, metric_loss = [], []
    for i, batch in enumerate(tqdm(data_loader, total=len(data_loader), ascii=" =", leave=False)):
        text=batch['text']
        label=batch['label'].to(device) # (64, )
        idx=batch['index'].to(device)

        optimizer.zero_grad()
        
        text_input_i=tokenizer(batch['text'], truncation=True, padding="max_length", 
                                  max_length=128, return_tensors="pt").to(device)

        text_input_j=tokenizer(batch['text'], truncation=True, padding="max_length", 
                                  max_length=128, return_tensors="pt").to(device)

        text_input_j['input_ids']=utils.mask(text_input_j['input_ids'], label, d, tokenizer, 
                                model.text_encoder.config.vocab_size, device, )
        
        text_input=[text_input_i, text_input_j]

        logits, text_feat_i, text_feat_j, loss_mlm = model(text_input)
        
        loss = ce(logits, label) + criterion(text_feat_i, text_feat_j) + loss_mlm 
        
        loss.backward()
        optimizer.step()
        
        # step 1. collecting
        index_sorter.collecting(text_feat_j, idx, label)
        
        # step 2. inserting
        index_sorter.inserting()
        
        # metrics
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true = label.to('cpu').numpy()
        y_pred = preds.to('cpu').numpy()
        metric_loss.append(loss.to('cpu').item())
        metric_acc.append(accuracy_score(y_true, y_pred))
    
    acc = sum(metric_acc) / len(metric_acc)
    loss = sum(metric_loss) / len(metric_loss)
    logging.info(f'[{epoch}/{max_epoch}] (Train) acc:{100*acc:.4f}%, loss:{loss:.6f}')
  
    # step 3. sorting & grouping
    t_q=index_sorter.G_feat_set
    l_q=index_sorter.G_label_set
    i_q=index_sorter.G_index_set
    G_index_set=index_sorter.grouping(t_q, l_q, i_q)
    
    # step 4. mini-batch level shuffle
    G_index_set=mini_batch_level_shuffle(G_index_set, 64)
    
    # temporary logging
    logging.info(f'G_index_set: {G_index_set[:8]}')
    
    return G_index_set, acc, loss
        
@torch.inference_mode()
def validate(model, data_loader, epoch, device, config):
    print('evaluate')

    model.eval()

    # tokenizer
    tokenizer=AutoTokenizer.from_pretrained(config['tokenizer'])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15  # 15% masking
    )
    
    metric_acc = []
    Y_TRUE, Y_PRED = [], []
    for i, batch in enumerate(tqdm(data_loader, total=len(data_loader), ascii=" =", leave=False)):
        text=batch['text']
        label=batch['label'].to(device) # (64, )
        idx=batch['index']

        text_input_i=tokenizer(batch['text'], truncation=True, padding="max_length", 
                                  max_length=512, return_tensors="pt").to(device)

        text_input_j=[tokenizer(text, truncation=True, padding="max_length", 
                                max_length=512, ) for text in batch['text']]

        text_input_j=data_collator(text_input_j).to(device)

        text_input=[text_input_i, text_input_j]

        logits, text_feat_i, text_feat_j, loss_mlm = model(text_input)
        
        # metrics
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        y_true = label.to('cpu').numpy()
        y_pred = preds.to('cpu').numpy()
        metric_acc.append(accuracy_score(y_true, y_pred))
        # macro f1 score
        Y_TRUE.extend(y_true)
        Y_PRED.extend(y_pred)
    
    acc = sum(metric_acc) / len(metric_acc)
    macro = f1_score(Y_TRUE, Y_PRED, average='macro')

    return acc, macro

def pretrain(args, config):
    # fix the seed 
    seed = config['seed'] + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    device="cuda" if torch.cuda.is_available() else "cpu"
    print('device: {}'.format(device))

    start_epoch = 0
    max_epoch = config['epoch']
    
    # dataset
    datasets = create_dataset(config)
    
    # sampler
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    # model
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
    model = MyModel(tokenizer=tokenizer, config=config, device=device)
    model = model.to(device)

    # objective function
    criterion = NTXentLoss(temperature=1.0)
    
    # optimizer & scheduler
    arg_opt=utils.AttrDict(config['optimizer'])
    optimizer=create_optimizer(arg_opt, model)
    arg_sche=utils.AttrDict(config['scheduler'])
    scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    # dev dataset
    val_loader=create_loader([datasets[1]],[None],batch_size=config['batch_size'],shuffles=[False],num_workers=config['num_workers'],collate_fns=config['collate_fns'],pin_memories=config['pin_memory'],drop_lasts=config['drop_last'])[0]

    # resume checkpoint
    if args.resume:
        print(f'resume train from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['lr_scheduler'])
        start_epoch = ckpt['epoch']+1
        next_index_set = sio.loadmat(
            os.path.join(args.indices_pth, f"total_indices_{start_epoch-1}.mat"),
        )['indices'].squeeze()
    
    print("Start training...")
    for epoch in range(start_epoch, max_epoch):
        
        if epoch == 0:
            samplers=create_sampler(datasets, [True], num_tasks, global_rank)
            logging.info("Creating Train DataLoader")
            data_loader=create_loader([datasets[0]],[None],batch_size=config['batch_size'],shuffles=config['shuffle'],num_workers=config['num_workers'],collate_fns=config['collate_fns'],pin_memories=config['pin_memory'],drop_lasts=config['drop_last'])[0]
        else:
            index_file = ...
            prev_index_set=next_index_set
            samplers=create_fixed_sampler(num_tasks, global_rank, prev_index_set)
            logging.info("Creating Train DataLoader")
            data_loader=create_loader([datasets[0]],samplers=samplers,batch_size=config['batch_size'],shuffles=config['shuffle'],num_workers=config['num_workers'],collate_fns=config['collate_fns'],pin_memories=config['pin_memory'],drop_lasts=config['drop_last'])[0]

        # train  
        next_index_set, acc, loss = train(model, data_loader, criterion, optimizer, epoch, device, scheduler, config)
        next_index_set = np.array(next_index_set)
        save_path=os.path.join(args.indices_pth, f'total_indices_{epoch}.mat')
        sio.savemat(save_path, {'indices' : next_index_set})
        
        # evaluate
        acc_val, macro_f1 = validate(model, val_loader, epoch, device, config)
    
        print(f'[{epoch+1}/{max_epoch}] (train) acc:{100 * acc:.2f} loss:{loss:.4f} (dev) acc:{100 * acc_val:.2f} macro f1:{macro_f1:.4f}')

        # save model
        if (epoch+1)%10==0:
            save_obj = {
                'model' : model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.checkpoint, 'ckpt-%02d.pth' % (epoch+1)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain.yaml')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--output_dir', default='./output/Pretrain')
    parser.add_argument('--checkpoint', default='./output/Pretrain/ckpt')
    parser.add_argument('--indices_pth', default='./output/Pretrain/indices')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.mkdir(args.checkpoint)
        os.mkdir(args.indices_pth)
        
    config = load_config(args.config)
    start_time = time.time()   
    logging.basicConfig(
        format='(%(asctime)s) %(levelname)s:%(message)s',
        datefmt='%m/%d %I:%M:%S %p', 
        level=logging.INFO,
        filename=os.path.join(args.output_dir, '.logging')
    )
    print("Start training...")

    pretrain(args, config)

    print(f"process finished. time elapsed: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")