import torch
from BigramLanguageModel import BigramLanguageModel
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/')

# Read the file -- entire dataset
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    #print("Length of the dataset is: ", len(text))
    return text

# Create the vocabulary
def create_vocabulary(text):
    vocabulary = sorted(set(text))
    #print("Length of the vocabulary is: ", len(vocabulary))
    return vocabulary
    
def encode_vocabulary(vocabulary):
    stoi = {ch: i for i,ch in enumerate(vocabulary)}
    encode = lambda s: [stoi[ch] for ch in s]
    return encode

def decode_vocaulary(vocabulary):
    itos = {i: ch for i, ch in enumerate(vocabulary)}
    decode = lambda a: ''.join(itos[i] for i in a)
    return decode

# Tokenize the text -- create the train and test dataset
def tokenize_text(vocabulary, text):
    encode = encode_vocabulary(vocabulary)
    decode = decode_vocaulary(vocabulary)

    data = torch.tensor(encode(text))
    train_percent = 0.9
    train_dataset = data[:int(0.9*len(data))]
    validation_dataset = data[int(0.9*len(data)):]

    #print(data[:1000])

    #print("Encoding of the text is: ", encode(text))
    #print("Decoding of the text is: ", decode(encode(text)))
    return train_dataset, validation_dataset

def get_batches(train_data, block_size):
    x = train_data[:block_size]
    y = train_data[1:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        #print(f"When the input context {context} then target is {target}")

def get_batched_blocks(xb, yb, batch_size, block_size):
    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            #print(f"When the input context is {context} then the target is {target}")


def get_optimizer(m):
    optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
    return optimizer

def main():
    block_size = 256
    max_iter = 1000
    eval_iters = 200
    eval_interval = 500
    #n_heads = 6
    n_heads = 1
    n_layers = 6

    device = "cuda" if torch.cuda.is_available() else "cpu"
      
    file_path = './input.txt'
    text = read_file(file_path)
    vocabulary = create_vocabulary(text)
    decode = decode_vocaulary(vocabulary)
    train_data, val_data = tokenize_text(vocabulary, text)
    get_batches(train_data, block_size)

    torch.manual_seed(1337)
    batch_size = 64 # number of batches to do in parallel

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    xb, yb = get_batch("train")
    get_batched_blocks(xb, yb, batch_size, block_size)

    m = BigramLanguageModel(len(vocabulary), n_heads, n_layers)
    m = m.to(device)
    optimizer = get_optimizer(m)

    # There is no gradient for the function -- helps in efficient use of memory
    # Run the model 'k' times and determine the loss value -- then take the mean value
    @torch.no_grad()
    def estimate_loss():
        out = {}
        # Eventhough there is no difference between the modes it is good to mention it
        # Since, there can be some layers that behave differently under different circumstances
        # Here, the code is in evaluation mode
        m.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = m(X, device, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        # Here the model is in training mode
        m.train()
        return out

    
    # Train the network
    #batch_size= 32
    for iter in tqdm(range(max_iter)):
        # every once in a while validate the model
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(losses)
            print(f"step{iter}: train loss is {losses['train']:.4f}, val loss is {losses['val']:.4f} ")

        xb, yb = get_batch("train")
        logits, loss = m(xb, device, yb)

        optimizer.zero_grad()
        loss.backward()

        for name, params in m.named_parameters():
            if params.grad is not None:
                writer.add_histogram(name, params.grad, iter)
    
        optimizer.step()

    print(loss.item())

    #logits, loss = m(xb, yb)
    #print(logits.shape)
    #print(loss)
    print(decode(m.generate(idx= torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500, device=device)[0].tolist()))

if __name__ == "__main__":
    main()
