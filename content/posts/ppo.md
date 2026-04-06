```python
import torch
from torch import nn
from torch.nn import functional as F

class PPO:
    def __init__(self,clip,gamma=0.99,lam=0.95):
        self.clip = clip
        self.gamma = gamma
        self.lam = lam
    
    def mask_mean(self,loss,mask,dim=-1):
        return (loss*mask).sum(dim=dim)/mask.sum(dim=dim)
    
    def advantage_estimate(self,rewards,values):
        seq_len = values.shape[1] # [batch_size, seq_len]
        advantages = torch.zeros_like(rewards)
        gae=0
        for i in range(seq_len-1,-1,-1):
            next_value = values[:,i+1] if i<seq_len-1 else 0.0
            # rewards[:,i] 对应 r_t
            # next_value 对应 V(s_{t+1})
            # values[:,i] 对应 V(s_t)
            delta = rewards[:,i]+self.gamma*next_value-values[:,i] # TD error = r_t + V(s_{t+1}) - V(s_t)
            gae = delta+self.lam*self.gamma*gae # A_t​=δt​+γλ*A_t+1​
            advantages[:,i]=gae # 给 batch 中所有 sequence 的第 i 个 token 赋 advantage
        returns =advantages+values # Target Value Q(s,a)=A(s,a)+V(s)
        return advantages,returns
    
    def policy_loss(self,new_probs,old_probs,advantages,act_mask):
        ratio = torch.exp(new_probs-old_probs)
        surr1=ratio*advantages
        surr2=torch.clamp(ratio,1-self.clip,1+self.clip)*advantages
        loss=-torch.min(surr1,surr2)
        return self.mask_mean(loss,act_mask) # act_mask 是 action mask（动作mask）
    
    def value_loss(self,new_values,returns,act_mask):
        loss = (new_values-returns)**2 # 更新预期
        return self.mask_mean(loss,act_mask)
```