from torch import nn
import torch
import math

class TransformerConfig:
    """ base GPT config, params common to all GPT versions """
    embedPdrop = 0.1
    residPdrop = 0.1
    attnPdrop = 0.1
    # Soft RELU Params
    weightLess=0.5
    offset=0.5
    train=True

    def __init__(self, numHeads, vocabSize, embeddingDim, posEmbeddingDim, keyDim, valueDim, hiddenSize, numLayers, seqLen, **kwargs):
        self.numHeads = numHeads
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.posEmbeddingDim = posEmbeddingDim
        self.keyDim = keyDim
        self.valueDim = valueDim
        self.hiddenSize = hiddenSize
        self.numLayers = numLayers
        self.seqLen = seqLen
        for k,v in kwargs.items():
            setattr(self, k, v)


import pytorch_lightning as pl
    


class SoftRELULayer(pl.LightningModule):
    def __init__(self, weightLess, offset, maxMag=4.0):
        super().__init__()
        self.weightLess = weightLess
        self.offset = offset
        self.maxMag = maxMag
    
    def forward(self, x):
        biggerThan = torch.max(torch.tensor([0.0]).to(device=x.device), x)
        lessThan = torch.min(torch.tensor([0.0]).to(device=x.device), x)
        res = biggerThan + lessThan*self.weightLess - self.offset
        if self.maxMag is not None:
            res.clamp_max_(self.maxMag)
        return res
    

# TODO: see if batch norm works for transformers

class AdaptableBatchedCrossEntropyLoss(HelpfulModule):
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
        

class BatchedIndexCrossEntropyLoss(HelpfulModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, target, rollupLosses=True):
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
        res = -values.clamp(0.00001, 0.99999).log()
        # this gives us one loss per (batch, word), usually they just want a single loss value, so this can roll them up if you want
        if rollupLosses: return res.mean()
        else: return res

class BatchedCrossEntropyLoss(HelpfulModule):
    def __init__(self):
        super().__init__()
    
    def forward(self, y, target, rollupLosses=True):
        vals = -target*y.log()-(1.0-target)*(1.0-y).log()
        # sum along not batch axis
        res = vals.sum(axis=2)
        if rollupLosses: return res.mean()
        else: return res
        # -target[i]*log(x[i])-(1-target[i])*log(1-x[i])

class LayerNorm(HelpfulModule):
    def __init__(self, eps=0.01):
        super().__init__()
        self.eps = eps
        self.multiplicitiveWeight = nn.Parameter(torch.tensor(1.0))
        self.additiveWeight = nn.Parameter(torch.tensor(0.0))
        self.nBatches = 0
    
    def forward(self, x):
        mu = x.mean((1,2,3), keepdim=True)
        var = x.var((1,2,3), keepdim=True) # TODO: add correction based on batch size
        normalizedOutput = (x-mu)/torch.max(var, torch.tensor(self.eps).to(device=x.device))
        return normalizedOutput*self.multiplicitiveWeight+self.additiveWeight

class SequentialDenseLayer(HelpfulModule):
    def __init__(self, name, inputDim, hiddenDim, outputDim, nLayers, act, einsumStr=None):
        super().__init__()
        self.name, self.inputDim, self.hiddenDim, self.outputDim, self.nLayers, self.act = name, inputDim, hiddenDim, outputDim, nLayers, act
        self.einsumStr = einsumStr
        projectInto = DenseLayer(name + "_" + "project", inputDim, hiddenDim,einsumStr=einsumStr)
        projectOut = DenseLayer(name + "_" + "projectOut", hiddenDim, outputDim,einsumStr=einsumStr)
        allLayers = [projectInto] + [DenseLayer(name + "_" + str(i), inputDim=hiddenDim, outputDim=hiddenDim, act=act,einsumStr=einsumStr) for i in range(nLayers)] + [projectOut]
        self.layers = nn.Sequential(*allLayers)
    
    def forward(self, x):
        return self.layers(x)
    
class DenseLayer(HelpfulModule):
    def __init__(self, name, inputDim, outputDim, act=None, einsumStr=None):
        super().__init__()
        self.name = name
        self.inputDim, self.outputDim = inputDim, outputDim
        self.weight = nn.Parameter(torch.normal(0, 1.0/math.sqrt(inputDim), [inputDim, outputDim])) # this is because dotting two vectors of mean zero std 1.0 gets output of mean zero std sqrt(inputDim), so we multiply to fix that
        self.bias = nn.Parameter(torch.normal(0, 1.0, [outputDim]))
        self.act = act
        self.einsumStr = einsumStr
    
    def forward(self, x, einsumStr=None):
        if einsumStr is None: einsumStr = self.einsumStr
        #print(self.name, "x", x, "weights", self.weights, "biases", self.biases)
        if einsumStr is None:
            res = (x@self.weight + self.bias) 
        else:
            res = (torch.einsum(einsumStr, x, self.weight)+self.bias)
        res.div_(math.sqrt(2.0)) # adding two things of mean 0 std 1 requires dividing by math.sqrt(2.0) to make output mean 0.0 std 1.0
        if self.act is None:
            return res
        else:
            return self.act(res)
    
class EmbeddingLayer(HelpfulModule):
    def __init__(self, vocabSize, embeddingDim):
        super().__init__()
        self.vocabSize, self.embeddingDim = vocabSize, embeddingDim
        # Todo: what is good initialization for embeddings?
        self.embeddings = nn.Parameter(torch.normal(0, 1, [vocabSize, embeddingDim]))
    # Inputs should be dimension [batchSize] and they should be integers
    def forward(self, x):
        return self.embeddings[x]
    
class Transformer(HelpfulModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n, k, v, m = config.numHeads, config.keyDim, config.valueDim, config.hiddenSize
        d = config.embeddingDim+config.posEmbeddingDim
        self.n, self.d, self.k, self.v, self.m = n,d,k,v,m
        self.embedding = EmbeddingLayer(config.vocabSize, config.embeddingDim)
        self.embedDropout = nn.Dropout(config.embedPdrop)
        self.posEmbeddings = nn.Parameter(torch.normal(0, 1, [config.seqLen, config.posEmbeddingDim]))
        self.encodingLayers = nn.Sequential(*[TransformerBlock(n,d,k,v,m,config,layerNum=i) for i in range(config.numLayers)])
        self.finalProjection = DenseLayer("FinalProj", n*d, config.embeddingDim)
        self.finalProjection2 = DenseLayer("FinalProj2", n*d, config.vocabSize)
        self.softmax = nn.Softmax(dim=2)
        self.lossFunc = AdaptableBatchedCrossEntropyLoss()
        # TODO: positional encodings
    
    
    def forward(self, x, targets=None, rollupLosses=True):
        # x is of size [b,L], word integer indices
        if len(x.shape) == 1: # make everythingn work for batch size 1
            x = x.view((1,x.shape[0]))
        b, L = x.shape
        
        n,d = self.n, self.d
        # embeddings need to go from [b,L,embeddingDim] to [b,L,n,embeddingDim]
        embs = self.embedding(x).view((b,L,1,self.config.embeddingDim)).expand((b, L, n, self.config.embeddingDim))
        if self.config.train:
            embs = self.embedDropout(embs)
        # positional embeddings are the same for every batch, so they need to go from [L,embeddingDim] to [b,L,n,embeddingDim]
        posEmbs = self.posEmbeddings[torch.arange(L)].view((1,L,1,self.config.posEmbeddingDim)).expand((b,L,n,self.config.posEmbeddingDim))
        if self.config.train:
            posEmbs = self.embedDropout(posEmbs)
        embeddings = torch.cat([embs, posEmbs], axis=3)
        #embeddings = embs + posEmbs
        # now it's ready to go through the embeddings
        forwardPass = self.encodingLayers(embeddings)
        #print(forwardPass[0,0,0])
        # It's currently dim [b,L,n,d], we need to make it [b,L,vocabSize]
        # For now I will just flatten and then project, so first make it [b,L,n*d]
        flattenedOutputs = forwardPass.reshape((b,L,n*d))
        # project to [b,L,vocabSize]
        
        # embeddings are [vocabSize, embeddingDim]
        # final embedding is [batch, numWords, embeddingDim]
        # we want output of dim [batch, numWords, vocabSize]
        # Also use softmax to convert to prs
        #finalEmbedding = self.finalProjection(flattenedOutputs, "bli,iv->blv")
        #wordPrs = self.softmax(torch.einsum("ve,bne->bnv", self.embedding.embeddings, finalEmbedding))
        wordPrs = self.softmax(self.finalProjection2(flattenedOutputs, "bli,iv->blv"))
        loss = None
        if targets is not None:
            loss = self.lossFunc(wordPrs, targets, rollupLosses=rollupLosses)
        
        return wordPrs, loss
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        no_decay = set()
        decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                else:
                    no_decay.add(fpn) # embeddings should not be decayed
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
        


# Transformer variant where it creates a list of weights of dim contextSize, then uses those to average over keys made by the last contextSize tokens
# each attention layer goes [b,L,n,d] -> [b,L-contextSize,n,d], which means that you need to input at least L>=contextSize*nlayers
# complexity is L*contextSize, this removes the need for any positional embeddings, and allows information to trickle a distance of contextSize*nLayers (possibly further with some later strats I have where you take the layers and output from the current layer)
class LookAroundTransformer(HelpfulModule):
    def __init__(self, config, numHeads, vocabSize, embeddingDim, nLayers, hiddenDimBefore, lookaroundDim, hiddenDimAfter, nLayersBefore, nLayersAfter, contextSize, hasAct=True, normalizeDim=True,addOther=False,  **kwargs):
        super().__init__()
        self.numHeads, self.vocabSize, self.embeddingDim = numHeads, vocabSize, embeddingDim
        self.n, self.d, self.nLayers = numHeads,embeddingDim,nLayers
        def makeLayer(i):
            return LookAroundAttention(config=config, n=self.n, d=self.d, hiddenDimBefore=hiddenDimBefore, lookaroundDim=lookaroundDim,hiddenDimAfter=hiddenDimAfter,nLayersBefore=nLayersBefore,nLayersAfter=nLayersAfter,contextSize=contextSize,layerNum=i,**kwargs)
        self.attentionLayers = torch.nn.Sequential(*[makeLayer(i) for i in range(nLayers)])
        self.embedding = EmbeddingLayer(vocabSize=vocabSize, embeddingDim=embeddingDim) # TODO: Figure out how to do embeddings with any size of vocab
        if hasAct:
            self.finalProj = DenseLayer("finalProj", self.d*self.n, self.embeddingDim, act=SoftRELULayer(**kwargs))
            self.finalProj2 = DenseLayer("finalProj2", self.d*self.n, self.vocabSize, act=SoftRELULayer(**kwargs))
        else:
            self.finalProj = DenseLayer("finalProj", self.d*self.n, self.embeddingDim)
            self.finalProj2 = DenseLayer("finalProj2", self.d*self.n, self.vocabSize)
        self.softmax = torch.nn.Softmax(dim=2)
        self.lossFunc = AdaptableBatchedCrossEntropyLoss()
        self.embedDropout = nn.Dropout(config.embd_pdrop)
        self.config = config
        self.normalizeDim = normalizeDim
        self.addOther = addOther
    
    def configure_optimizers(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        no_decay = set()
        decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight'):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                else:
                    no_decay.add(fpn) # embeddings should not be decayed
        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    
    def forward(self, x, targets=None):
        b,L = x.shape
        n,d = self.n, self.d
        embeddings = self.embedding(x) # [b,L] -> [b,L,d]
        if self.config.train:
            embeddings = self.embedDropout(embeddings)
        inputsToAttention = embeddings.view((b,L,1,d)).expand((b,L,n,d)) # make it look like [b,L,n,d] for the attention heads
        attentionOutputs = self.attentionLayers(inputsToAttention) # output will be [b,L-contextSize*nLayers,n,d]
        b,Lnew,n,d = attentionOutputs.shape # Lnew = L - contextSize*nLayers
        flattenedOutputs = attentionOutputs.reshape((b,Lnew,n*d))
        finalEmbedding = self.finalProj(flattenedOutputs, "blk,kv->blv") # [b,Lnew,n*d]x[n*d,embeddingDim] -> [b,Lnew,embeddingDim]
        finalProj2 = self.finalProj2(flattenedOutputs, "blk,kv->blv")
        # It's important we normalize the magnitude of the things before dot product (see this one cool paper)
        if self.normalizeDim:
            wordPrs = torch.einsum("ve,bne->bnv", normalizeDim(self.embedding.embeddings, dim=1), normalizeDim(finalEmbedding, dim=2))
        else:
            wordPrs = torch.einsum("ve,bne->bnv", self.embedding.embeddings, finalEmbedding)
        if self.addOther:
            wordPrs = wordPrs + finalProj2
        wordPrs = self.softmax(wordPrs)
        loss = None
        if targets is not None:
            remainingTargets = targets[:,-Lnew:] # we can only measure the ones that had sufficient context
            loss = self.lossFunc(wordPrs, remainingTargets)
        return wordPrs, loss
        
# ensures sum of rows is 1.0.
# if any rows are all zeros, it sets them to uniform distr (all values are 1.0/length of row)
# assumes input values are between 0.0 and 1.0, probably generated by torch.rand
def normalizeDim(mat, dim):
    rowSums = mat.sum(axis=dim, keepdim=True)
    matNew = mat.masked_fill((rowSums==0).expand(mat.shape), 1.0) # any rows that are all zero get set to uniform distr to prevent divide by zero and ensure good probabilities
    rowSums = matNew.sum(axis=dim, keepdim=True)
    matNew.div_(rowSums)
    return matNew

class LookAroundAttention(HelpfulModule):
    def __init__(self, config, n, d, hiddenDimBefore, lookaroundDim, hiddenDimAfter, nLayersBefore, nLayersAfter, contextSize, layerNum=0, **kwargs):
        super().__init__()
        self.act = SoftRELULayer(**kwargs)
        self.n, self.d, self.hiddenDimBefore, self.lookaroundDim, self.hiddenDimAfter, self.nLayersBefore, self.nLayersAfter, self.contextSize, self.layerNum = n, d, hiddenDimBefore, lookaroundDim, hiddenDimAfter, nLayersBefore, nLayersAfter, contextSize, layerNum
        #self.projectToLookaround = SequentialDenseLayer("projectToLookaround_" + str(layerNum), inputDim=d, hiddenDim=hiddenDimBefore, outputDim=hiddenDimBefore, nLayers=nLayersBefore, act=self.act, einsumStr="blnd,dh->blnh")
        self.projectToLookaroundKey = DenseLayer("projectToLookaroundKey_" + str(layerNum), hiddenDimBefore, lookaroundDim, act=self.act)
        self.projectToLookaroundWeights = DenseLayer("projectToLookaroundWeights_" + str(layerNum), hiddenDimBefore, contextSize, act=self.act)
        #self.projectFromLookaround = SequentialDenseLayer("projectAfterLookaround_" + str(layerNum), inputDim=lookaroundDim, hiddenDim=hiddenDimAfter, outputDim=d, nLayers=nLayersAfter, act=self.act, einsumStr="blnv,vd->blnd")
        self.softmax = torch.nn.Softmax(dim=3)
        self.attentionDropout = nn.Dropout(config.attn_pdrop)
        self.config = config # todo: add keys
    
    def forward(self, x, intermediateResults=False, efficientMethod=True):
        # x is [b,L,n,d], output is [b,L-contextSize,n,d]
        b,L,n,d = x.shape
        LAfter = L-self.contextSize+1
        # we are going to output [b,L-contextSize,n,d]
        # first, project to lookaround vector. Each vector of size d is dotted with something of size d to be turned into something of size k
        #lookaroundVector = self.projectToLookaround(x) # [b,L,n,d] -> [b,L,n,hiddenDimBefore] # "blnd,dh->blnh"
        lookaroundKeys = self.projectToLookaroundKey(x, "blnh,hv->blnv") # [b,L,n,hiddenDimBefore] -> [b,L,n,lookaroundDim]
        # we can only apply lookbehind to things that have contextSize or more things to look at (we start from -LAfter which has a + 1 because one of the things they look at is themselves
        lookaroundWeights = self.projectToLookaroundWeights(x[:,-LAfter:], "blnh,hw->blnw") # [b,LAfter,n,hiddenDimBefore] -> [b,L-contextSize,n,contextSize]
        # apply softmax so they become weights from 0 to 1 that sum to 1
        actualWeights = self.softmax(lookaroundWeights) # [b,L-contextSize,n,contextSize]
        lookaroundValues = []
        # to apply this in one large matrix, we need to somehow do:
        # lookaroundKeys[:,i:i+self.contextSize]
        # apply lookaround
        
        if efficientMethod:
            sampleIndices = torch.stack([torch.arange(i,i+self.contextSize) for i in range(LAfter)]).long().to(x.device)
            wordContexts = lookaroundKeys[:,sampleIndices] # [b,LAfter,contextSize,n,d]
            lookaroundValue = torch.einsum("blcnd,blnc->blnd", wordContexts, actualWeights) # [b,LAfter,contextSize,n,lookaroundDim]x[b,LAfter,n,contextSize]->[b,LAfter,n,lookaroundDim]
            #print("lookaroundValue1", lookaroundValue, lookaroundValue.shape)
        else:
            for i in range(LAfter):
                wordWeights = actualWeights[:,i] # [b,n,contextSize]
                # we need to take [b,L,n,lookaroundDim] and get the relevant vectors we will be using
                context = lookaroundKeys[:,i:i+self.contextSize] # [b,contextSize,n,lookaroundDim]
                # dot the weights along the axis: this does a weighted sum of context vectors
                lookaroundValues.append(torch.einsum("bcnd,bnc->bnd", context, wordWeights).view((b,1,n,self.lookaroundDim))) # [b,contextSize,n,lookaroundDim] x [b,n,contextSize] -> [b,n,lookaroundDim]
            # stack all the vectors we found
            lookaroundValue = torch.cat(lookaroundValues, dim=1) # [b,L-contextSize,n,lookaroundDim]
            #print("lookaroundValue2", lookaroundValue, lookaroundValue.shape)
        #res = self.projectFromLookaround(lookaroundValue) # "blnv,vd->blnd"
        res = lookaroundValue+x[:,-LAfter:] # do resnet
        if self.config.train:
            res = self.attentionDropout(res)
        if intermediateResults:
            return lookaroundVector, lookaroundKeys, lookaroundWeights, actualWeights, lookaroundValues, lookaroundValue, res
        else:
            return res
        

class MultiHeadSelfAttention(HelpfulModule):
    def __init__(self, n, d, k, v, config, layerNum=0, doNormalize=True, efficientMethod=True):
        super().__init__()
        self.n, self.d, self.kDim, self.vDim = n,d,k,v
        self.config = config
        self.doNormalize = doNormalize
        self.efficientMethod = efficientMethod
        # Todo: compute initialization scaling factors
        # TODO: What about more things than just QKV? Like four or five or something
        self.Q = DenseLayer("Q" + str(layerNum), d, k)
        self.K = DenseLayer("K" + str(layerNum), d, k)
        self.V = DenseLayer("V" + str(layerNum), d, v)
        self.Wch = DenseLayer("Wch" + str(layerNum), v,d)
        self.softmax = torch.nn.Softmax(dim=1)
        self.softmaxAlltogether = torch.nn.Softmax(dim=3)
        self.attentionDropout = nn.Dropout(config.attnPdrop)
    def forward(self, x):
        # x is [b,L,n,d]
        # b is batch size
        # L is sentence length
        # n is num heads
        # d is embedding dimension
        # we need to use Q, K, V to make a qi, ki, vi for each word
        # because we dot qi and kj, they need to be same dim, call this k
        # vi can be any dim, call this v
        # we need [b,L,n,d] -> [b,L,n,k] for qi and ki
        # we need [b,L,n,d] -> [b,L,n,v] for vi
        # [b,L,n,k]
        
        b,L,n,d,kDim,vDim = x.shape[0], x.shape[1], self.n, self.d, self.kDim, self.vDim
        
        q = self.Q(x, "blnd,dk->blnk")
        k = self.K(x, "blnd,dk->blnk")
        v = self.V(x, "blnd,dv->blnv")
        # each of these dot a row of dim d by a column of dim d, so we need to divide by sqrt(d) to ensure output is mean 0 std 1
        #print("q", q[:,0,0,0].mean(), q[:,0,0,0].std())
        #print("k", k[:,0,0,0].mean(), k[:,0,0,0].std())
        #print("v", v[:,0,0,0].mean(), v[:,0,0,0].std())
        # Normally people just do a massive matrix, but that is quadratic in terms of L, and very wasteful with memory
        # Instead, we will do a loop over each word and do this for each word.
        # It's still quadratic in terms of L for time complexity (and slightly slower than giant matrix, because we are in python), but now linear in terms of space complexity, which is important for GPU space
        
        # simpler way:
        if self.efficientMethod:
            # need to dot each pair of q and k

            # q[i,j] is a vector of size k
            # k[i,j] is a vector of size k
            # for every pair of (vector from q, vector from k), we need to get an output by taking their dot product
            # normally if you have two matrices A and B of size NxM and MxK,
            # when you multiply them, you can think of the output matrix's value in the (i,j)th position as the ith row in A dot jth column in B
            # (thus it is every pair of row from first and column from second)
            # in einsum, torch.einsum("ij,jk->ik", A, B)
            # If instead A and B are of size NxM and NxM and you want to do every pair of rows, you can just do
            # torch.einsum("ij,kj->ik") # transpose second matrices indices so it takes rows instead of columns
            # we have an additional batch and head index at the front, so include that
            # this will output something of dim [b,L,n,L]
            # value [i,j,k,l] is batch i, head k, vector j dot with vector l 
            dotQueryKey = torch.einsum("binj, bknj->bink", q, k)/math.sqrt(kDim)
            # we need to softmax along axis 3, this will be [b,L,n,L] -> [n,L]
            #print("scores 0 fast:", dotQueryKey[0,0,:,:].shape, dotQueryKey[0,0,:,:])
            queryPrs = self.softmaxAlltogether(dotQueryKey)
            # now dot query weights with vectors to get [b,L,n,v] 
            u = torch.einsum("binj,bjnk->bink", queryPrs, v)
        else:
        

            inds = torch.tensor(range(L))
            us = []
            for i in range(L):
                # q is [b,L,n,k]
                # expand it so it looks as k so we can do dot product
                qi = q[:,i,:,:].view((b,1,n,kDim)).expand((b,L,n,kDim))
                # dot product is component wise product and then sum, so just do that
                # scores is now [b,L,n]
                scores = (qi*k).sum(axis=3) 
                #print("ayy", scores.shape, (b,L,n))
                #if i == 0: print("scores 0 slow", scores[0].shape, scores[0])
                #print("scores bad", scores[:,0,0].mean(), scores[:,0,0].std())
                if self.doNormalize:
                    scores.div_(math.sqrt(kDim+0.0)) # also divide by sqrt(k), this ensures outputs are mean 0 std 1 if values of Q and K are mean 0 std 1
                #print("scores good", scores[:,0,0].mean(), scores[:,0,0].std())
                scores[:,inds>i,:] = np.NINF # mask out words after current word
                scores = self.softmax(scores)
                # scores is [b,L,n], we need to make it look like [b,L,n,1] so we can expand it along last axis 
                scores = scores.view((b,L,n,1)).expand((b,L,n,vDim))
                ui = (scores*v).sum(axis=1)
                #print((scores*v).shape, (b,L,n,vDim), ui.shape, (b,n,vDim))
                #print("ui bad", ui[:,0,0].mean(), ui[:,0,0].std())
                # in general, for a weighted sum of uncorrelated variables, we have
                # var(sum_i s_i*x_i) = sum_i s_i^2*var(x_i)
                # if we assume all x_i are initially std=1.0 (so var(x_i) = 1.0^2=1.0), we get
                # var(sum_i s_i*x_i) = sum_i s_i^2
                # Since we want var(sum_i s_i*x_i) = 1.0, we need to multiply by a constant, and if we do
                # var((sum_i s_i*x_i)/sqrt(sum_i s_i^2))
                # = var(sum_i s_i*x_i)/(sum_i s_i^2)
                # = (sum_i s_i^2)/(sum_i s_i^2)
                # = 1.0
                if self.doNormalize:
                    ui = ui/(scores.pow(2.0).sum(axis=1).sqrt())
                #print("ui good", ui[:,0,0].mean(), ui[:,0,0].std())
                us.append(ui.view((b,1,n,vDim)))
            u = torch.cat(us, dim=1)
        # u is [b,L,n,vDim]
        # we want [b,L,n,d]
        res = self.Wch(u, "blnv,vd->blnd") # this computation dots rows of dim v by columns of dim v, so we need to divide by sqrt(v) to ensure output is mean 0 std 1
        if self.config.train:
            res = self.attentionDropout(res)
        return res

    
class TransformerBlock(HelpfulModule):
    def __init__(self, n, d, k, v, m, config, layerNum=0, doNormalize=True):
        super().__init__()
        # input x is [b,n,d]
        # b is batchSize
        # n is number of heads
        # d is embedding dimension
        # k is key size
        # m is hidden layer size
        self.n, self.d, self.k, self.m = n, d, k, m
        self.config = config
        self.W1 = DenseLayer("W1_" + str(layerNum), d,m)
        self.W2 = DenseLayer("W2_" + str(layerNum), m,d)
        self.attention = MultiHeadSelfAttention(n,d,k,v, config, doNormalize=doNormalize)
        self.layerNorm1 = LayerNorm()
        self.layerNorm2 = LayerNorm()
        self.doNormalize = doNormalize
        self.RELU = SoftRELULayer(config.weightLess, config.offset)
        
    def forward(self, x):
        attentionOut = self.attention(x)
        ui = self.layerNorm1(x+attentionOut) # todo: check to see if layer norm inside res net block is doing weird stuff, since we have a second res net thing below not attached
        # [d,m]x[b,L,n,d] -> [b,L,n,m]
        # this dot products rows of size d by columns of size d, so we need to divide by sqrt(d) to get mean 0 std 1
        denseOutput = self.RELU(self.W1(ui, "blnd,dm->blnm"))
        # this dot products rows of size m by columns of size m, so we need to divide by sqrt(m) to get mean 0 std 1
        projectedBack = self.W2(denseOutput, "blnm,md->blnd")
        return self.layerNorm2(ui+projectedBack)
        
    