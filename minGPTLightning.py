"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

# Ported from minGPT library https://github.com/karpathy/minGPT

import math
import logging
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl

logger = logging.getLogger(__name__)

import wandb


import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

def sampleModel(contextString, model, data, numChars, temperature=1.0, top_k=10):
    x = torch.tensor([data.stoi[s] for s in contextString], dtype=torch.long)[None,...].to(model.device)
    y = sample(model, x, numChars, temperature=temperature, sample=True, top_k=top_k)[0]
    completion = ''.join([data.itos[int(i)] for i in y])
    return completion


import math
from torch.utils.data import Dataset, DataLoader, random_split

# This code is tweaked from play_char in minGPT
class CharDataModule(pl.LightningDataModule):
    def __init__(self, text_path, block_size, train_weight, valid_weight, test_weight, batch_size):
        super().__init__()
        f = open(text_path, "r")
        self.text = f.read()
        f.close()
        self.block_size = block_size
        self.batch_size = batch_size
        self.train_weight = train_weight
        self.valid_weight = valid_weight
        self.test_weight = test_weight
        
    # optional, only called once and one 1 GPU
    # we will use to this to create the character to index mapping
    def prepare_data(self):
        chars = sorted(list(set(self.text)))
        data_size, vocab_size = len(self.text), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.vocab_size = vocab_size
        total_weight = self.train_weight + self.valid_weight + self.test_weight
        data_len = len(self.text)
        self.train_len = int((float(self.train_weight)/total_weight)*data_len)
        self.valid_len = int((float(self.valid_weight)/total_weight)*data_len)
        self.test_len  = int((float(self.test_weight)/total_weight)*data_len)
        self.train_data = self.text[:self.train_len]
        self.valid_data = self.text[self.train_len:self.train_len+self.valid_len]
        self.test_data  = self.text[self.train_len+self.valid_len:]

    # called on each GPU seperately and accepts stage to define if we are at fit or test step
    def setup(self, stage=None):
        # we set up only relevant datasets when stage is specified
        if stage == 'fit' or stage is None:
            self.train_dataset = CharDataset(self.train_data, self.stoi, self.itos, self.block_size, self.batch_size)
            self.valid_dataset = CharDataset(self.valid_data, self.stoi, self.itos, self.block_size, self.batch_size)
        if stage == 'test' or stage is None:
            self.test_dataset = CharDataset(self.test_data, self.stoi, self.itos, self.block_size, self.batch_size)

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # TODO: Add a test thing that does one character at a time prediction
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class CharDataset(Dataset):

    def __init__(self, data, stoi, itos, block_size, batch_size=128):
        self.data = data
        self.stoi = stoi
        self.itos = itos
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = len(self.stoi)
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

class AdaptableBatchedCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batchedPrLoss = BatchedCrossEntropyLoss()
        self.batchedIndexLoss = BatchedIndexCrossEntropyLoss()
    
    def forward(self, y, targets, rollupLosses=True):
        if targets.dtype == torch.int64: # fitting to desired word indices
            if len(targets.shape) == 1: # if single batch, expand out
                targets = targets.view((1, targets.shape[0]))
            loss = self.batchedIndexLoss(y, targets, rollupLosses=rollupLosses)
        else: # fitting to word prs
            if len(targets.shape) == 2: # if single batch, expand out
                targets = targets.view((1, targets.shape[0], targets.shape[1]))
            loss = self.batchedPrLoss(y, targets, rollupLosses=rollupLosses)
        return loss
        
class BatchedIndexCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, target, inputsAreLog=False, rollupLosses=True):
        '''
        torch.gather(input, dim, index) does the following
        out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
        out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
        out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

        y is [b,L,vocabSize]
        goals is [b,L]
        we want
        out[bi,l] = y[bi,l,goals[bi,l]]
        but that doesn't fit the above pattern.
        To fix this, we can just do
        out[bi,l,k] = y[bi,l,goals[bi,l,k]]
        where k is only ever 0
        so we need to add that axis to goals
        '''
        b,L = target.shape
        values = torch.gather(y, 2, target.view((b,L,1)))
        # Now make it look like b,L
        values = values.view((b,L))
        # Actual pr for those values is 1.0, so
        # -target*x.log()-(1.0-target)*(1.0-x).log()
        # turns into
        if not inputsAreLog:
            res = -values.clamp(0.00001, 0.99999)
            res.log_()
        else:
            res = values
        # this gives us one loss per (batch, word), usually they just want a single loss value, so this can roll them up if you want
        if rollupLosses: return res.mean()
        else: return res

class BatchedCrossEntropyLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, target, inputsAreLog=False, rollupLosses=True):
        # TODO: Add log support for this loss
        # (1-y).log() if y is already taken log?
        # we can do (1-exp(y)).log() but that doesn't seem ideal
        vals = -target*y.log()-(1.0-target)*(1.0-y).log()
        # sum along not batch axis
        res = vals.sum(axis=2)
        if rollupLosses: return res.mean()
        else: return res
        # -target[i]*log(x[i])-(1-target[i])*log(1-x[i])


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    weight_decay = 0.1 # only applied on matmul weights
    betas = (0.9, 0.95)
    learning_rate = 3e-4
    print_every = 100
    save_every = 2000
    ckpt_path = None
    
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(pl.LightningModule):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(pl.LightningModule):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, data, trainer):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        
        self.config = config
        
        self.hparams.lr = config.learning_rate
        
        # log hyperparameters (saves to self.hparams, which is logged to wandb as the config)
        self.save_hyperparameters()

        self.trainer = trainer
        self.data = data
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        
        print("configure optimizers with lr" + str(self.hparams.lr))
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.lr, betas=self.config.betas)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.lr*10, steps_per_epoch=len(self.data.train_dataset), epochs=10)
        return [optimizer], [scheduler]

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        # convert logits from [b,L,dim] -> [b*L,dim]
        # convert targets from [b,L] -> [b*L]
        # that way cross entropy is happy and can deal with indices
        # cross entropy does the log internally so this is what we want
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ys.view(-1))
        return logits, loss

        # if we are given some desired targets also calculate the loss
        #loss = None
        #if targets is not None:
        #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        #return logits, loss
    
    # takes a batch and computes the loss; backprop goes through it
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)

        # logging metrics we calculated by hand
        # Here's the docs for reference https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html#log
        # this takes a name and value, and under the hood it uses wandb.log
        self.log('train/loss', loss, on_epoch=False, on_step=True) # if you do on_step=False (by default this is true) then it'll only do epoch wise averaging outputs, see test_step below
    
        if self.global_step % self.config.print_every == 0:
            sample = sampleModel("hi\n", self, self.data, 200, temperature=1.0, top_k=10)
            self.logger.experiment.log(
            {"train/sample": wandb.Html(sample.replace("\n", "<br/>")),
             "global_step": self.global_step})
        
        if self.config.ckpt_path is not None and self.global_step % self.config.save_every == 0:
            os.makedirs(self.config.ckpt_path, exist_ok=True)
            self.trainer.save_checkpoint(self.config.ckpt_path + "/" + str(self.global_step) + ".ckpt")
        
        return loss
        
    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        
        
    # save the model after we are done with testing, we will use ONNX format (https://onnx.ai/) cause it lets us use nice things like the neutron model viewer in W&B (https://github.com/lutzroeder/netron)
    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        #wandb.save(model_filename)
        pass
    
    # return the logits so they can be used by validation_epoch_end
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        
        return logits
    
    # example of how to log the logits as a histogram
    def validation_epoch_end(self, validation_step_outputs):
        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})