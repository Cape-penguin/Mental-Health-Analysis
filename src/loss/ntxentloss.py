import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, zis, zjs):
        
        batch_size = zis.shape[0]

        # 1. Normalize
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # 2. Concat [2B, D]
        z = torch.cat([zis, zjs], dim=0)

        # 3. Similarity matrix [2B, 2B]
        sim = torch.matmul(z, z.T) / self.temperature

        # 4. Mask self-similarity
        mask = torch.eye(2 * batch_size, device=zis.device).bool()
        sim.masked_fill_(mask, -1e9)  

        # 5. Positive pairs (i vs i+B, and i+B vs i)
        positives = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)

        labels = positives  

        # 6. Cross-entropy loss
        loss = F.cross_entropy(sim, labels)

        return loss