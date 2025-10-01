import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Optional

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

from src.dataset.post_dataset import post_dataset
from src.dataset.sampler_for_grit import Sampler_for_GRIT

def create_sampler(datasets, shuffles, num_tasks, global_rank):
    print("Creating samplers")
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = DistributedSampler(dataset,num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     

def create_fixed_sampler(num_tasks, global_rank, index_set):
    print("Creating a fixed sampler")
    samplers = []
    sampler = Sampler_for_GRIT(pre_indices=index_set,num_replicas=num_tasks, rank=global_rank)
    samplers.append(sampler)
    return samplers   

def create_dataset(config: Optional[dict] = None):
    print('Creating datasets')

    df = pd.read_csv(config['train_file'], encoding='utf-8')

    train_df, dev_df = train_test_split(df, test_size=0.5, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(dev_df, test_size=0.5, random_state=42, shuffle=True)

    # pd.Series to list
    train_df = train_df.apply(lambda row: [row['post'], row['illness']], axis=1,).tolist() 
    val_df = val_df.apply(lambda row: [row['post'], row['illness']], axis=1,).tolist()
    train_dataset = post_dataset(train_df)
    val_dataset = post_dataset(val_df)

    print(f"train dataset size = {len(train_dataset)}, dev dataset size = {len(val_dataset)}")

    return [train_dataset, val_dataset]

def create_loader(datasets, samplers, batch_size, shuffles, num_workers, collate_fns, pin_memories, drop_lasts):
    print(f"Creating DataLoaders")
    loaders = []
    for idx, (dataset,sampler,bs,shuffle,n_worker,collate_fn,pin_memory,drop_last) in enumerate(zip(datasets,samplers,batch_size,shuffles,num_workers,collate_fns,pin_memories,drop_lasts)):
        shuffle = (sampler is None)
        loader = DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=n_worker,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )              
        loaders.append(loader)

    return loaders