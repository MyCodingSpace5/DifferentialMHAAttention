class XDifferentialAttention(nn.Module):
    def __init__(self, q, k, v, embedding_size=512, num_heads=16, head_dim=64, vocab_size=5000):
        super().__init__()
        self.num_heads = num_heads
        self.W_k = nn.ParameterList([nn.Parameter(torch.empty(embedding_size, head_dim * 2)) for _ in range(self.num_heads)])
        self.W_v = nn.ParameterList([nn.Parameter(torch.empty(embedding_size, head_dim)) for _ in range(self.num_heads)])
        self.W_q = nn.ParameterList(
            [nn.Parameter(torch.empty(embedding_size, head_dim)) for _ in range(self.num_heads)])
        self.W_o = nn.Parameter(torch.empty(self.num_heads * self.head_dim, embedding_size))
        for i in range(self.num_heads):
            nn.init.xavier_uniform(self.W_k[i])
            nn.init.xaiver_uniform(self.W_v[i])
            nn.init.xaiver_uniform(self.W_q[i])
        nn.init.xavier_uniform_(self.W_o)
        self.head_dim = head_dim
        self.prediction_layer = nn.Linear(self.num_heads * self.head_dim, vocab_size)
        self.lambd1 = nn.Parameter(torch.tensor(0.12))
        self.lambd2 = nn.Parameter(torch.tensor(0.24))
        self.lambd3 = nn.Parameter(torch.tensor(0.82))
        self.lambd4 = nn.Parameter(torch.tensor(0.55))
        self.lambd_init = 0.272
        self.lambd = torch.exp(self.lambd1 * self.lambd2) - torch.exp(self.lambd3 * self.lambd4) + self.lambd_init
    def attention(self, x, i):
        q = x @ self.W_q[i]
        k = x @ self.W_k[i]
        v = x @ self.W_v[i]
        s = 1 / torch.sqrt(self.head_dim)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        a1 = q1 @ k1.transpose(-1, -2) * s
        a2 = q2 @ k2.transpose(-1, -2) * s
        attn_weights = (torch.softmax(a1, dim=-1) - self.lambd * torch.softmax(a2, dim=-1))
        attn_weights @ v
    def mha(self, x):
        heads = [self.attention(x, i) for i in range(self.num_heads)]
        O = torch.stack(heads, dim=1) # (num_heads, batch_size, num_latents, head_dim)
        head_dim = O.size(3)
        O = O.permute(1, 0, 3, 2) # (batch_size, num_heads, head_dim, num_latents)
        O = O.flatten(start_dim=1, end_dim=2) # (batch_size, num_heads * head_dim, num_latents)
        norm = nn.GroupNorm(self.num_heads, self.num_heads * head_dim)
        O = group_norm(O) # same shape
        O = O.unflatten(1, (self.num_heads, head_dim)) # (batch_size, num_heads, head_dim, num_latents)
        O = O.permute(1, 0, 3, 2) # (num_heads, batch_size, num_latents, head_dim)
        O = O * (1 - self.lambd_init)
        O = O.permute(1, 2, 0, 3) # (batch_size, num_latents, num_heads, head_dim)
        O = O.flatten(start_dim=2) # (batch_size, num_latents, num_heads * head_dim)
        return O @ self.W_o
    def forward(self, x):
        x = self.mha(x)
        x = self.prediction_layer(x)
        self.lambd = torch.exp(self.lambd1 * self.lambd2) - torch.exp(self.lambd3 * self.lambd4) + self.lambd_init
        return x
