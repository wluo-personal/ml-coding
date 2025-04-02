"""
Multi-head attention + KV cache + RoPE

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_linear = nn.Linear(hidden_size, hidden_size)
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        self.o_linear = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, hidden_state, causal_mask=None, past_key_value=None, use_cache=True):
        batch_size = hidden_state.size(0)

        # calculate KQ, K, V
        query = self.q_linear(hidden_state)
        key = self.k_linear(hidden_state)
        value = self.v_linear(hidden_state)

        # split - multi head

        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # we can add RoPE infomation here (after query, key linear transform and before attention calculation)
        # ROPE only applies to new query and key

        # check cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # (batch, num_head, seq_length, hidden_size)
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        # save new cache
        new_past_key_value = (key, value) if use_cache else None

        # calculate attention score
        attention_scores = torch.matmul(query, key.transpose(-1,-2)) / torch.sqrt(self.head_dim)

        # causal mask if needed
        if causal_mask is not None:
            attention_scores += causal_mask * -1e9

        # calculate attention output
        attention_probs = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_probs, value)

        # combine multi-head
        output = output.transpose(1,2).contiguous().view(batch_size,-1, self.hidden_size)
        output = self.o_linear(output)
        return (output, new_past_key_value) if use_cache else output



