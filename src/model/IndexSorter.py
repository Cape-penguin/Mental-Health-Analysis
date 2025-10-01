import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from typing import Optional
from sklearn.cluster import KMeans

def show_label_dist(cluster_labels, n_clusters):
    logging.info(f'Show label distribution. n_cluster={n_clusters}')
    label_counts = Counter(cluster_labels)
    for label, count in sorted(label_counts.items()):
        logging.info(f"\t[{label}] {count} samples")

class IndexSorter():
    # Index Sorter in Clusters
    def __init__(self, device, num_steps, config:Optional[dict]):

        self.device=device
        self.embed_dim=int(config['embed_dim'])
        self.batch_size=int(config['batch_size'][0])
        self.queue_size=int(config['queue_size'])
        self.total_sample_len=num_steps*self.batch_size
        
        # For simplicity
        assert self.queue_size % self.batch_size == 0
        
        # global queue
        self.G_index_set=torch.zeros(self.total_sample_len, dtype=torch.int32).to(device)
        self.G_label_set=torch.zeros(self.total_sample_len, dtype=torch.int32).to(device)
        self.G_feat_set=torch.zeros((self.total_sample_len, self.embed_dim), dtype=torch.float32).to(device)
        self.Cluster_label_set=torch.zeros(self.total_sample_len, dtype=torch.int32).to(device)
        self.G_idx=0
        
        # queue        
        self.idx_queue=torch.zeros(self.queue_size, dtype=torch.int32).to(device)
        self.label_queue=torch.zeros(self.queue_size, dtype=torch.int32).to(device)
        self.feat_queue=torch.zeros((self.queue_size, self.embed_dim), dtype=torch.float32).to(device)
        self.queue_ptr=torch.zeros(1, dtype=torch.int32)
        
        # temporary logging
        logging.info('Cluster Index Sorter configuration.')
        logging.info(f'device={device}, num_steps={num_steps}')
        logging.info(f'total_sample_len={self.total_sample_len}')
        logging.info(f'G_feat_set.size()={self.G_feat_set.size()}')
        logging.info(f'G_label_set.size()={self.G_label_set.size()}')
        logging.info(f'G_index_set.size()={self.G_index_set.size()}')
        
    @torch.no_grad()
    def inserting(self,):
        # enqueue
        self.G_index_set[self.G_idx:self.G_idx+self.queue_size]=self.idx_queue
        self.G_label_set[self.G_idx:self.G_idx+self.queue_size]=self.label_queue
        self.G_feat_set[self.G_idx:self.G_idx+self.queue_size]=self.feat_queue
        self.G_idx=(self.G_idx+self.queue_size) % self.total_sample_len
        
    @torch.no_grad()
    def collecting(self, feat, idx, lbl):
        ptr = int(self.queue_ptr)
        # replace the keys at ptr (dequeue and enqueue)
        self.idx_queue[ptr:ptr + self.batch_size]=idx.detach()
        self.label_queue[ptr:ptr + self.batch_size]=lbl.detach()
        self.feat_queue[ptr:ptr + self.batch_size]=feat.detach()
        ptr = (ptr + self.batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def clustering(self, n_clusters=12):
        # initiate k-means cluster
        n_clusters=n_clusters
        kmeans=KMeans(n_clusters=n_clusters, random_state=42)
        
        feats=self.G_feat_set.to('cpu').numpy()
        kmeans.fit(feats)
        cluster_labels=kmeans.labels_
        # cluster label distribution
        show_label_dist(cluster_labels, n_clusters)

        self.Cluster_label_set=torch.tensor(cluster_labels, dtype=torch.int32)

    @torch.no_grad()
    def grouping(self, text_sub_queue, label_sub_queue, index_sub_queue, temp=1.0):
        
        label_sub_queue=label_sub_queue.view(-1, 1)
        index_sub_queue=index_sub_queue.view(-1, 1)
        
        sim_t2t = F.softmax(text_sub_queue.detach() @ text_sub_queue.detach().t() / temp,dim=1)
        sim_t2t = sim_t2t * (label_sub_queue.detach() != label_sub_queue.detach().t())
        sim_t2t.fill_diagonal_(-1)
        
        bs=text_sub_queue.size()[0]
        I_index_set=[]
        start = torch.randint(low=0, high=int(bs-1),size=(1,))[0]
        start = start.to(text_sub_queue.device)
        prev_t_idx=start
        
        for _ in range(bs):
            next_t=torch.topk(sim_t2t[prev_t_idx], 1)
            I_index_set.append(index_sub_queue[prev_t_idx].item())
            sim_t2t[prev_t_idx,:]=-1
            sim_t2t[:,prev_t_idx]=-1
            prev_t_idx=next_t.indices[0]
        
        return I_index_set            
    
@torch.no_grad()
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
            yield lst[i:i + n]

@torch.no_grad()
def mini_batch_level_shuffle(index_set, batch_size):
    divided_G_index_set = list(chunks(index_set,batch_size))
    total_chunk_size = len(divided_G_index_set)
    chunk_arr = np.arange(total_chunk_size)
    random.shuffle(chunk_arr)
    shuffled_G_index_set=[]
    for ind in chunk_arr:
        shuffled_G_index_set += divided_G_index_set[ind]

    return shuffled_G_index_set