```python
import torch

class GRPO:
    
    def __init__(self,eps,clip,beta):
        self.eps =eps
        self.clip = clip
        self.beta = beta
    
    def mask_mean(self,loss,mask,dim=-1):
        return (loss*mask).sum(dim=dim)/mask.sum(dim=dim) # 点乘 和 tensor 按维度求和

    
    def group_advantages(self,rewards):
        mean=torch.mean(rewards)
        std = torch.std(rewards,unbiased=False)# unbiased=False， 真实这组数的尺度，而不是总体估计
        advantages = (rewards-mean)/(std+self.eps)
        return advantages.detach()# 梯度截断
    
    def grpo_loss(self,old_logps, new_logps,ref_logps,advantages):
        ref_logr = new_logps - ref_logps # logr
        kl_score = (torch.exp(ref_logr)-ref_logr-1)*self.beta #  KL penalty: e^logr-logr-1
        ratio = torch.exp(new_logps - old_logps) # r   exp 的作用: 把“对数概率差”还原成“概率的比例”
        surr1 = ratio*advantages # r*A
        surr2 = torch.clamp(ratio,1-self.clip,1+self.clip)*advantages # clip(r)*A
        loss = - torch.mean(torch.min(surr1,surr2)-kl_score) # 最大化,
        return loss # loss
```