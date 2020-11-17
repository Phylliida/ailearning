# really dumb primality test, from https://stackoverflow.com/a/8019354/2924421
def prime(a):
    return not (a < 2 or any(a % x == 0 for x in range(2, int(a ** 0.5) + 1)))

def toArr(n, base, numDigits):
    strRepr = formatBase(n, base, numDigits)
    return [toInt(c) for c in strRepr]

def formatBase(n, base, numDigits):
    res = np.base_repr(n, base)
    return "0"*(numDigits-len(res))+res

def toInt(c):
    if '0' <= c <= '9': return int(c)
    elif 'A' <= c <= 'Z': return 10+ord(c)-ord('A')
    elif 'a' <= c <= 'z': return 10+26+ord(c)-ord('a')

# https://stackoverflow.com/a/28666223/2924421
def numberToBase(n, b, numDigits):
    if n == 0:
        return [0]*numDigits
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    res = digits[::-1]
    # append zeros to front as needed
    return [0]*(numDigits-len(res))+res
    
    
from torch.utils.data import Dataset
import numpy as np
import torch
import automataBattle
import random
import math
from importlib import reload
reload(automataBattle)

global debug
debug=False

def dPrint(*args, **kwargs):
    global debug
    if debug: print(*args, **kwargs)
# creates a function you can call with an array and the number of things to sample. It'll sample using argmax
def makeModelSampler(model, device):
    if hasattr(model, "block_size"):
        contextSize = model.block_size
    else:
        contextSize = model.module.block_size
    def sampleModel(arr, numToSample):
        def toArr(arr):
            if type(arr) is torch.tensor: # if tensor, convert to list
                arr = arr.cpu().numpy().flatten()
            return list(arr)
        def sampleOnce(arr):
            # pad with zeros if needed
            posOfOutput = -1 # if we have a full context, the output symbol is the last symbol we output
            if len(arr) < contextSize:
                posOfOutput = len(arr)-1 # we don't have a full context, we need to retreive the symbol after the provided context
                inputs = arr + [0]*(contextSize-len(arr))
            elif len(arr) > contextSize: # too many, just use the last bit for context
                inputs = arr[-contextSize:]
            else:
                inputs = arr
            inputTensor = torch.tensor(inputs).long().view((1, len(inputs))).to(device)
            dPrint("inputTensor:", inputTensor)
            outputs = model(inputTensor)[0].detach().cpu()
            dPrint("outputs", outputs)
            prs = torch.nn.Softmax(dim=2)(outputs)
            dPrint("prs", prs)
            outputSymbols = prs.argmax(axis=2).flatten().numpy()
            dPrint("ouput symbols", outputSymbols)
            dPrint("pos of output", posOfOutput)
            return list(outputSymbols)[posOfOutput]
        curArr = toArr(arr)
        sampledOutputs = []
        for i in range(numToSample):
            sampledOutput = sampleOnce(curArr)
            curArr.append(sampledOutput)
            sampledOutputs.append(sampledOutput)
        return sampledOutputs
    return sampleModel


class AutomataDataset(Dataset):
    def __init__(self, automata, label, symbols, split, sequenceLen):
        self.automata = automata
        self.label = label
        self.split = split # train/test
        self.vocab_size = len(automata.symbols)**2
        # -1 because we are predicting next token
        self.block_size = sequenceLen
        self.nSymbols = len(automata.symbols)
        
        self.stoi = dict([(s, i) for (i, s) in enumerate(symbols)])
        self.itos = dict([(i, s) for (i, s) in enumerate(symbols)])
        self.sequenceLen = sequenceLen
        
        self.numSymbols = len(self.stoi)
        possibleInputs = self.numSymbols**sequenceLen
        
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(possibleInputs)
        num_test = min(int(possibleInputs*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        
        '''
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        '''


    def __len__(self):
        return len(self.ixes)

    def __str__(self):
        return str(self.label)
    
    def testModelOnDataI(self, idx, model, device):
        sampleModel = makeModelSampler(model, device)
        inputSymbols = numberToBase(self.ixes[idx], self.numSymbols, self.sequenceLen)
        X, automataY = self.generate(self.sequenceLen, inputSymbols)
        desiredX, desiredY = self[idx]
        i = 0
        dPrint("originalInputSymbols", inputSymbols)
        dPrint("X", X, "Y", automataY)
        dPrint("self[idx]", self[idx])
        inputSymbols = [X[i]+automataY[i]*self.nSymbols]
        modelOutputs = []
        while True:
            dPrint("inputSymbols", inputSymbols)
            yPredict = sampleModel(inputSymbols, 1)[0]
            modelOutputs.append(yPredict)
            dPrint("yPredict", yPredict, "modelOutputs", modelOutputs)
            i += 1
            if len(modelOutputs) >= len(automataY)-1: break # -1 because first symbol of automata outputs is always produced by the initial state before processing anything
            dPrint("next x:", X[i])
            yPredict = yPredict % self.nSymbols # things >= nSymbols are invalid outputs, so loop around
            nextInput = X[i]+yPredict*self.nSymbols
            inputSymbols.append(nextInput)
        return X, automataY, list(desiredY.cpu().numpy()), inputSymbols, modelOutputs
    
    def runTest(self, model, device, showFailures=False):
        correct = 0
        incorrect = 0
        for i in range(len(self)):
            X, automataY, Y, inputSymbols, modelOutputs = self.testModelOnDataI(i, model, device)
            if Y != modelOutputs:
                incorrect += 1
                if showFailures:
                    print("X", X, "automataY", automataY, "desiredY", Y, "modelInputs", inputSymbols, "modelOutputs", modelOutputs)
            else:
                correct += 1
        return (correct/float(correct+incorrect))
                
    
    # Actually produces [sequenceLen] for X and [sequenceLen+1] for Y, because the first value of Y is the symbol emitted by the initialState before processing any strings
    def generate(self, sequenceLen, inputSymbols):
        X, Y = self.automata.generate(sequenceLen, lambda i: self.itos[inputSymbols[i]])
        X = [self.stoi[x] for x in X]
        Y = [self.stoi[y] for y in Y]
        return X, Y
        
    
    def __getitem__(self, idx):
        inputSymbols = numberToBase(self.ixes[idx], self.numSymbols, self.sequenceLen)
        X, Y = self.generate(self.sequenceLen, inputSymbols)
        x = torch.tensor(X)
        y = torch.tensor(Y) # predict the output of the Automata
        previous = y[:-1] # of length sequenceLen, because first item is the symbol emitted by initialState
        outputs = y[1:] # Todo: look into encoding multiple things ("tuple encodings") instead of this gross thing
        # we need to encode the previous output as part of our input string, otherwise it only sees the input string and has no hope of knowing what's happening
        # technically the loss should provide the necessary info, but I find that this helps significantly since otherwise it has to do everything with positional encodings which doesn't really work well for arbitrary lengths (TODO - examine why this is the case)
        xOutput = x+previous*self.nSymbols # for now, we just encode a tuple (input, prevOutput) as input+prevOutput*vocabSize
        yOutput = outputs
        return xOutput, yOutput
        
        '''
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028" 
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
        '''


class FastLearnAutomataDataset(Dataset):
    def __init__(self, nStates, nSymbols, split, sequenceLen, numSequences):
        self.nStates = nStates
        self.nSymbols = nSymbols
        self.split = split # train/test
        self.vocab_size = nSymbols*nSymbols
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = sequenceLen
        
        self.sequenceLen, self.numSequences = sequenceLen, numSequences
        
        '''
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        '''


    def __len__(self):
        return self.numSequences

    def __getitem__(self, idx):
        a = automataBattle.Automata(nStates=self.nStates, symbols=range(self.nSymbols), randomConnect=True)
        a.minimize()
        while a.complexity() != self.nStates:
            a = automataBattle.Automata(nStates=self.nStates, symbols=range(self.nSymbols), randomConnect=True)
            a.minimize()
        X, Y = a.generate(self.sequenceLen+1, lambda: random.choice(range(self.nSymbols)))
        x = torch.tensor(X)
        y = torch.tensor(Y) # predict the output of the Automata
        previous = y[:-1]
        shiftedForwadInputsOne = x[1:]
        outputs = y[1:] # Todo: look into encoding multiple things ("tuple encodings") instead of this gross thing
        xOutput = shiftedForwadInputsOne+previous*self.nSymbols
        yOutput = outputs
        return xOutput, yOutput
        
        '''
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028" 
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
        '''


class FactoringDataset(Dataset):
    """ tweaked from code from mingpt
    Returns factoring problems of up to some number of digits in the inputs. Recall
    that all GPT cares about are sequences of integers, and completing them according to
    patterns in the data. Therefore, we have to somehow encode addition problems
    as a sequence of integers.
    
    The sum of two n-digit numbers gives a third up to (n+1)-digit number. So our
    encoding will simply be the n-digit first number, n-digit second number, 
    and (n+1)-digit result, all simply concatenated together. Because each addition
    problem is so structured, there is no need to bother the model with encoding
    +, =, or other tokens. Each possible sequence has the same length, and simply
    contains the raw digits of the addition problem.
    
    As a few examples, the 2-digit problems:
    - 85 + 50 = 135 becomes the sequence [8, 5, 5, 0, 1, 3, 5]
    - 6 + 39 = 45 becomes the sequence [0, 6, 3, 9, 0, 4, 5]
    etc.
    
    We will also only train GPT on the final (n+1)-digits because the first
    two n-digits are always assumed to be given. So when we give GPT an exam later,
    we will e.g. feed it the sequence [0, 6, 3, 9], which encodes that we'd like
    to add 6 + 39, and hope that the model completes the integer sequence with [0, 4, 5]
    in 3 sequential steps.
    
    fun exercise: does it help if the result is asked to be produced in reverse order?
    """

    def __init__(self, ndigit, base, split):
        self.split = split # train/test
        self.ndigit = ndigit
        self.base = base
        self.vocab_size = base # base possible digits 0..base
        # *2 because num digits might be multiplied by two for product output, but then -1 because very last digit doesn't plug back
        self.block_size = ndigit + ndigit + ndigit*2 - 1
        self.primes = [i for i in range(base**self.ndigit) if prime(i)]
        
        # split up all addition problems into either training data or test data
        num = len(self.primes)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def __len__(self):
        return self.ixes.size

    def asStr(self, idx, base):
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = len(self.primes)
        a = self.primes[idx // nd]
        b = self.primes[idx %  nd]
        c = a*b
        sortedThings = [a,b] # sort them so it's reliable for the network
        sortedThings.sort()
        a,b = sortedThings
        render = f'{formatBase(c, base, self.ndigit*2)}{formatBase(a, base, self.ndigit)}{formatBase(b, base, self.ndigit)}' # e.g. 15=3*5= becomes "150305" 
        return render
    
    def __getitem__(self, idx):
        render = self.asStr(idx, self.base)
        dix = [toInt(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
    
# From MinGPT Repo
class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
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
    
    
    

# ensures sum of rows is 1.0.
# if any rows are all zeros, it sets them to uniform distr (all values are 1.0/length of row)
# assumes input values are between 0.0 and 1.0, probably generated by torch.rand
def normalizeRows(mat):
    rowSums = mat.sum(axis=1, keepdim=True)
    mat[(rowSums==0).flatten()] = 1.0 # any rows that are all zero get set to uniform distr to prevent divide by zero and ensure good probabilities
    rowSums = mat.sum(axis=1, keepdim=True)
    return mat/rowSums

def sampleRow(row):
    return torch.multinomial(row.flatten(), 1, replacement=True)
def sampleRowDeterministic(row):
    return row.flatten().argmax()

class HMM(object):
    def __init__(self, nNodes, nSymbols, randomizeFirstState=True):
        self.nNodes, self.nSymbols, self.randomizeFirstState = nNodes, nSymbols, randomizeFirstState
        # self.transitionMatrix[i,j] is pr of going from node i to node j
        # thus, [i,0] + [i,1] + ... + [i,n-1] = 1
        # so each row needs to sum to 1
        self.transitionMatrix = normalizeRows(torch.rand([nNodes, nNodes]))
        # self.emitMatrix[i,j] is pr of node i emitting symbol j
        # thus, [i,0] + [i,1] + ...  [i,n-1] = 1
        # so each row needs to sum to 1
        self.emitMatrix = normalizeRows(torch.rand([nNodes, nSymbols]))
        # Todo: initial distr on initial nodes
        
    def generate(self, nTokens):
        if self.randomizeFirstState:
            curState = torch.randint(0, self.nNodes, [1])
        else:
            curState = torch.tensor([0])
        result = []
        for i in range(nTokens):
            result.append(sampleRowDeterministic(self.emitMatrix[curState]))
            curState = sampleRowDeterministic(self.transitionMatrix[curState])
        return torch.tensor(result)

        
    
        
    def test(self):
        # Check that rows are roughly summing to 1.0
        assert(torch.allclose(self.transitionMatrix.sum(axis=1), torch.ones([self.nSymbols]), 0.001))

class HMMDataset(Dataset):
    def __init__(self, hmm, split, sequenceLen, numSequences):
        self.hmm = hmm
        self.split = split # train/test
        self.ndigit = ndigit
        self.vocab_size = hmm.nSymbols
        # +1 due to potential carry overflow, but then -1 because very last digit doesn't plug back
        self.block_size = sequenceLen
        
        self.sequenceLen, self.numSequences = sequenceLen, numSequences
        
        '''
        # split up all addition problems into either training data or test data
        num = (10**self.ndigit)**2 # total number of possible combinations
        r = np.random.RandomState(1337) # make deterministic
        perm = r.permutation(num)
        num_test = min(int(num*0.2), 1000) # 20% of the whole dataset, or only up to 1000
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]
        '''


    def __len__(self):
        return self.numSequences

    def __getitem__(self, idx):
        
        data = self.hmm.generate(self.sequenceLen)
        x = data[:-1]
        y = data[1:] # predict the next token in the sequence
        return x, y
        
        '''
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx]
        nd = 10**self.ndigit
        a = idx // nd
        b = idx %  nd
        c = a + b
        render = f'%0{self.ndigit}d%0{self.ndigit}d%0{self.ndigit+1}d' % (a,b,c) # e.g. 03+25=28 becomes "0325028" 
        dix = [int(s) for s in render] # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long) # predict the next token in the sequence
        y[:self.ndigit*2-1] = -100 # we will only train in the output locations. -100 will mask loss to zero
        return x, y
        '''
