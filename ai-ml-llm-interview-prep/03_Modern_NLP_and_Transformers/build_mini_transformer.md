# Coding a Mini Transformer in PyTorch

In a highly technical MLE interview, you might be asked to implement the core Multi-Head Self-Attention block from scratch. 

---

## The PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model       # e.g., 512
        self.num_heads = num_heads   # e.g., 8
        self.d_k = d_model // num_heads # e.g., 64 (Dimension per head)
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        seq_len = q.size(1)
        
        # 1. Linear projections and reshape for multi-head
        # Original shape: (batch_size, seq_len, d_model)
        # Reshaped: (batch_size, seq_len, num_heads, d_k)
        # Transposed: (batch_size, num_heads, seq_len, d_k) -> Crucial for matmul
        
        Q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Calculate Attention Scores (Dot Product)
        # Q * K^T. We transpose the last two dimensions of K.
        # Output shape: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 3. Scale by sqrt(d_k) to prevent vanishing gradients in softmax
        scores = scores / math.sqrt(self.d_k)
        
        # 4. Apply Mask (Optional - e.g., for Decoder autoregressive generation)
        if mask is not None:
            # Mask should contain -1e9 for positions we want to ignore (future tokens)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5. Softmax to get probability distribution
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 6. Multiply weights by Values
        # Output shape: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attention_weights, V)
        
        # 7. Concatenate the heads back together
        # Transpose back: (batch_size, seq_len, num_heads, d_k)
        # Contiguous() is required before view() after transpose
        # Reshape to original: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 8. Final linear projection
        return self.out_linear(output)

# --- Sanity Check ---
# batch=2, seq_length=10, d_model=512
dummy_input = torch.rand(2, 10, 512)
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Self-attention means Q, K, and V all come from the same input
output = mha(dummy_input, dummy_input, dummy_input)
print(f"Input shape: {dummy_input.shape}") 
print(f"Output shape: {output.shape}") # Should be exactly the same
```

## Interview Focus Points
1.  **The Reshape and Transpose:** Explain *why* we reshape to `(batch, num_heads, seq_len, d_k)`. We do this so PyTorch's `matmul` treats the batch and heads as independent parallel operations, multiplying the `seq_len x d_k` matrices together instantly.
2.  **The Masking Logic:** Explain that setting future tokens to `-infinity` means that after Softmax, their attention weight becomes exactly `0.0`.
3.  **The Scaling Factor:** If you forget `/ math.sqrt(self.d_k)`, it's an instant red flag. Explain it stabilizes gradients.