## MultiHeadSelfAttention
```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,nums_head=8,hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.head_dim = hidden_dim//nums_head
        self.q_proj = nn.Linear(hidden_dim,hidden_dim)
        self.k_proj = nn.Linear(hidden_dim,hidden_dim)
        self.v_proj = nn.Linear(hidden_dim,hidden_dim)
        self.o_proj = nn.Linear(hidden_dim,hidden_dim)
        self.dropout=nn.Dropout(0.1)

    def forward(self,x,attn_mask=None):
        batch_size,seq_len,self.hidden_dim = x.size()
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        # split
        q_state = q.view(batch_size,seq_len,self.nums_head,self.head_dim).transpose(1,2)
        k_state = k.view(batch_size,seq_len,self.nums_head,self.head_dim).transpose(1,2)
        v_state = v.view(batch_size,seq_len,self.nums_head,self.head_dim).transpose(1,2)
        
        # q@k^t/sqrt(d)
        attn_score = q_state@k_state.transpose(-1,-2)/math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask ==0,-1e9)
        attn_weight = torch.softmax(attn_score,dim=-1)
        attn_weight = self.dropout(attn_weight)
        output_mid = attn_weight@v_state
        # concat
        output_mid = output_mid.transpose(-1,-2).contiguous()
        output_mid = output_mid.view(batch_size,seq_len,-1)
        output= self.o_proj(output_mid)
        return output

x= torch.rand(3,2,512)
attn_mask = torch.tensor([
    [0,1],
    [0,0],
    [1,0],
]).unsqueeze(1).unsqueeze(2).expand(3,8,2,2)

net = MultiHeadSelfAttention(8,512)
net(x,attn_mask).shape  
    
```