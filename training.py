import torch
import torch.nn as nn
import mmap
import random
import pickle
import argparse
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
def parse_args():
    parser = argparse.ArgumentParser(description = "This is a demo program")
    parser.add_argument('-batch_size',type = int,required = True,help = 'Please provide a batch size')
    return parser.parse_args()
args = parse_args()
print(f'batch size: {args.batch_size}')
from torch.nn import functional as F
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(device)
chars = ""
with open('vocab.txt','r', encoding='utf-8') as f:
    text= f.read()
chars = sorted(list(set(text)))
print(chars)
batch_size = args.batch_size
block_size = 256
vocab_size = len(chars)
max_inters = 55000
learning_rate = 1e-4
eval_inters = 250
dropout = 0.2
n_embed = 512
n_layer = 8
n_head = 8

string_to_int = { ch: i for i, ch in enumerate(chars)}
int_to_string = { i: ch for i, ch in enumerate(chars)}
encode = lambda s : [string_to_int[c] for c in s]
decode = lambda l : '' .join([int_to_string [i] for i in l])
data = torch.tensor(encode(text), dtype = torch.long)
print(data[:100])

n = int(0.8 * len(data))
train_data = data[:n]
val_data = data[n:]
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for spilt in ['train', 'val']:
        losses = torch.zeros(eval_inters)
        for k in range (eval_inters):
            X, Y = get_batch(spilt)
            logits , loss = m(X,Y)
            losses[k] = loss.item()
            out[spilt] = losses.mean()
    m.train()
    return out
class Head(nn.Module):
    " " " one head of self-attention " " " 
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embed,head_size,bias = False)
        self.query = nn.Linear(n_embed,head_size,bias = False)
        self.value = nn.Linear(n_embed,head_size,bias = False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        #input of size (batch(B),time-step(T),channels(C)
        #output of size (batch(B),time-step(T), head_size(h)
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        #Compute attention scores
        weight = q @ k.transpose(-2,-1) * k.shape[-1] ** -0.5
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        #perform weighted aggreation
        v = self.value(x)
        out = weight @ v
        return out
    
                            


class MultiHeadAttention(nn.Module):
    " " "  multiple heads of self-attention in parallel" " " 
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj =nn.Linear(head_size * num_heads, n_embed)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedFoward(nn.Module):
    " " " a simple linear layer followed by non-linearity" " " 
    def __init__(self,n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    " " " Transformer block:comunication followed by computation " " "
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head,head_size)
        self.ffwd = FeedFoward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    def forward(self, x):
        y = self.sa(x)
        x= self.ln1( x + y)
        y = self.ffwd(x)
        x= self.ln2( x + y)
        return x 
        


class GPTLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head = n_head) for _ in range (n_layer)])

        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed,vocab_size)

        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
               torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
        
        
    def forward(self,index,targets = None):
        logits = self.token_embedding_table(index)


        #idx  and targets are both (B,T) tensor of integers
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.postion_embedding_table(torch.arange(T, device=device)) #(T,C)
        x = tok_emb + pos_emb #(B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,C,vocab_size)
        

        
        if targets is None:
            loss = None
        else:
        
            B,T,C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits,targets)
        return logits , loss
    def generate(self,index, max_new_tokens):
        #index is (B * T) arrayof indices in the current context
        for _ in range (max_new_tokens):
            index_cond = index[:, -block_size:]
            #get the predictions
            logits , loss = self.forward(index_cond)
            #focus only on last time step
            logits = logits[:, -1, :] # Becomes (B, C)
            #apply softmax to get possibities/probaities
            probs = F.softmax(logits, dim = -1) #(B,C)
            #sample from distribution
            index_next = torch.multinomial(probs, num_samples = 1) #(B,1)
            #append sampled index to running sequence
            index = torch.cat((index,index_next), dim = 1) #(B, T +1)
        return index
model = GPTLanguageModel(vocab_size)
m = model. to(device)

    
x = train_data[:block_size]
y = train_data[1:block_size + 1]

#memory map for using small snippets of text from a single file at a time
def random_chunk(spilt):
    filename = "train_split.txt" if spilt == "train" else "val_spilt.txt"
    with open(filename, 'rb') as f:
       with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #Determine filesize and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)
            #Seek  the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size -1)
            #Decode the block to a string, ignoring any invalid byte sequence
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r',' ')

            # Train and test spilts
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data
    


def get_batch(split):
    data = random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i +1:i + block_size+ 1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y
x,y = get_batch('train')
print(x)
print()
print(y)

# Create Pytorch Optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)
for inter in range(max_inters):
    
    if inter % eval_inters == 0:
        losses = estimate_loss()
        print(f'step {inter }, losses{losses}')
    
    #sample batch of data
    xb, yb = get_batch('train')
    logits, loss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    #eval. loss
print(loss.item())
    
with open("model-01.pkl","wb") as f:
    pickle.dump(m,f)

print("Loading Model")
with open('model-01.pkl', 'rb') as f:
    m = pickle.load(f)
    model = m.to(device)
print("loaded")

