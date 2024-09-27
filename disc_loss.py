import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeLoss(nn.Module):

    def get_similarity_scores(self, text_feats, image_feats):
        cosine_sims = text_feats @ image_feats.t() # (B x B) sim matrix
        targets = cosine_sims.diag(0).unsqueeze(1) # targets are the image itself --> extract the main diagnol (B x 1)
        cosine_sims.fill_diagonal_(float("-inf"))  # mask targets.
        cosine_sims = torch.cat([targets, cosine_sims], dim=1) # B x (1+B)

        return cosine_sims

    def forward(self, text_feats, img_feats):
        """
        rand_25: winoground (AND)
        rand_50: 1 retrieval correct (OR)
        """
        sims = self.get_similarity_scores(text_feats, img_feats) # Sim matrix of size [B | B X B]. 1st column = retrieval targets
        labels = torch.zeros(sims.shape[0]).long().to(img_feats.device)
        acc_1 = (sims.argmax(dim=1) == labels).detach().float()        
        rand_50 = acc_1.mean().item()
        rand_25 = 1 if rand_50 ==1 else 0
        
        return rand_50, rand_25, acc_1
