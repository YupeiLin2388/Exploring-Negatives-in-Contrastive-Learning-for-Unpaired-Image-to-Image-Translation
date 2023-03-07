from packaging import version
import torch
from torch import nn




class PatchNCELoss(nn.Module):
    def __init__(self, opt,patch):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.patch_nums = patch

        

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)
            #elif self.opt.choose_patch !=0:
        l_neg, _ = l_neg.sort(dim=0, descending=True)
        l_neg = l_neg[:, :self.opt.choose_patch]
                #print(l_neg[-3:,])
                #zero_indexs = l_neg[:,0].reshape(-1)
                #zero_indexs = zero_indexs.tolist()
                #indexs =  zero_indexs.index(-10)
                #n = l_neg.shape[0]-indexs
                #l_neg[indexs:] = l_neg[:n]

        
        
        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        #print(out[0])
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        #print("ranknce_loss:",loss,loss.shape)
        print(loss)
        #loss +=self.opt.ib_aphla*ib_loss
        #print("total_loss",loss)
        return loss
