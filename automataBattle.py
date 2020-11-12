
import random
import graphviz
import numpy as np
from numba import jit
import networkx as nx


class State(object):
  def __init__(self, index, outputSymbol):
    self.index = index
    self.outputSymbol = outputSymbol
    self.connections = {}
    self.connectionsInto = {}
  def connect(self, symbol, state):
    if symbol in self.connections:
      self.removeConnection(symbol, self.connections[symbol])
    self.connections[symbol] = state
    if not symbol in state.connectionsInto:
      state.connectionsInto[symbol] = set()
    state.connectionsInto[symbol].add(self)

  def removeConnection(self, symbol, state):
    if symbol in self.connections and self.connections[symbol] == state:
      self.connections.pop(symbol)
    if symbol in state.connectionsInto and self in state.connectionsInto[symbol]:
      state.connectionsInto[symbol].remove(self)


# see https://www.youtube.com/watch?v=0XaGAkY09Wc
def minimizeDFA(symbols, states, initialState):
  # Get the set of things that can be reached from the start state
  reachable = set()
  toProcess = [initialState]
  while len(toProcess) > 0:
    cur = toProcess.pop(0)
    if not cur.index in reachable:
        reachable.add(cur.index)
        for sym in symbols:
          nextThing = cur.connections[sym]
          if not nextThing.index in reachable:
            toProcess.append(nextThing)
  

    
  # two states are called zero equivalent if they output the same symbol
  curEquivalences = []
  for sym in symbols:
    symGroup = []
    for si in reachable:
      s = states[si]
      if s.outputSymbol == sym:
        symGroup.append(s)
    if len(symGroup) > 0:
      curEquivalences.append(symGroup)
  for i, eq in enumerate(curEquivalences):
    for s in eq:
      s.eqGroup = i
  
  madeChange = True
  while madeChange:
    madeChange = False
    newEquivalences = []
    #print([[str(x.index) for x in box] for box in curEquivalences])
    for g in curEquivalences:
      boxes = [[g[0]]]
      for e in g[1:]:
        boxToMergeInto = None
        for box in boxes:
          goodBox = True
          for sym in symbols:
            if box[0].connections[sym].eqGroup != e.connections[sym].eqGroup:
              goodBox = False
              break
          if goodBox:
            boxToMergeInto = box
            break
        if boxToMergeInto is None:
          boxes.append([e])
        else:
          boxToMergeInto.append(e)
      newEquivalences += boxes
      if len(boxes) > 1: madeChange = True
    curEquivalences = newEquivalences
    for i, eq in enumerate(curEquivalences):
      for s in eq:
        s.eqGroup = i
  boxes = curEquivalences
  resStates = []
  for i, box in enumerate(boxes):
    resStates.append(State(i, box[0].outputSymbol))
  for i in range(len(boxes)):
    exampleNode = boxes[i][0]
    for sym in symbols:
      outputBoxI = exampleNode.connections[sym].eqGroup
      resStates[i].connect(sym, resStates[outputBoxI])
  return resStates, resStates[initialState.eqGroup]


class Automata(object):
  '''
  Initial state is always the first one
  '''
  def __init__(self, nStates, symbols, randomConnect=True):
    self.minimized = False
    self.minimizedVersion = None
    self.symbols = symbols
    self.states = [State(i, random.choice(symbols)) for i in range(nStates)]
    self.initialState = random.choice(self.states)
    if randomConnect:
        for symbol in symbols:
            for state in self.states:
              connectedTo = random.choice(self.states)
              state.connect(symbol, connectedTo)

  def makeOffspring(self):
    res = self.copy()
    res.mutate()
    return res

  def copy(self):
    res = Automata(len(self.states), self.symbols, randomConnect=False)
    for i, state in enumerate(self.states):
      for (symbol, connectedTo) in state.connections.items():
        res.states[i].connect(symbol, res.states[connectedTo.index])
      res.states[i].outputSymbol = state.outputSymbol
    return res
  
  def mutate(self):
    choice = random.choice([1,2,3,4])
    if choice == 1: # add a new state and connect it randomly
      newState = State(len(self.states), random.choice(self.symbols))
      self.addState(newState)
      for symbol in self.symbols:
        newState.connect(symbol, random.choice(self.states))
    if choice == 2: # remove a state
      if len(self.states) > 1: # only remove if we have more than one node
        nodeRemoving = random.choice(self.states)
        # remmove it
        self.states.pop(nodeRemoving.index)
        # update indexes to new values
        for i in range(nodeRemoving.index, len(self.states)):
          self.states[i].index = i
        if nodeRemoving == self.initialState: # randomly reassign initial state if it is the one we are removing
          self.initialState = random.choice(self.states)
        # reassign stuff that were going into the removed node elsewhere
        connectionsInto = [x for x in nodeRemoving.connectionsInto.items()]
        for symbol, thingsGoingInto in connectionsInto:
          for thingGoingInto in list(thingsGoingInto): # list lets us modify it bu making a copy
            newConnection = random.choice(self.states)
            thingGoingInto.connect(symbol, newConnection)
    if choice == 3: # randomly redirect a transition link
      sym = random.choice(self.symbols)
      newToConnectTo = random.choice(self.states)
      newConnectingTo = random.choice(self.states)
      newConnectingTo.connect(sym, newToConnectTo)
    if choice == 4: # change the symbol produced by a state
      random.choice(self.states).outputSymbol = random.choice(self.symbols)
  
  
  def addState(self, state):
    self.states.append(state)
    
  def generate(self, maxTokens, inputGenerator=None):
    myOutputSymbol, myState, myProcess = self.process()
    inputSymbols = []
    outputSymbols = []
    for t in range(maxTokens):
        if inputGenerator is None:
            inputSymbol = random.choice(self.symbols)
        else:
            inputSymbol = inputGenerator()
        (myOutputSymbol, myState) = myProcess(inputSymbol, myState)
        inputSymbols.append(inputSymbol)
        outputSymbols.append(myOutputSymbol)
    return inputSymbols, outputSymbols
    
  def generateFromSelf(self, maxTokens, bailOnLoop=True):
    myOutputSymbol, myState, myProcess = self.process()
    res = []
    history = set()
    for t in range(maxTokens):
        history.add(myState)
        res.append(myOutputSymbol)
        (myOutputSymbol, myState) = myProcess(myOutputSymbol, myState)
        if bailOnLoop and myState in history: break # Don't do loops
    return res

  def process(self):
    def nextState(symbol, curState):
      outputSymbol = curState.outputSymbol
      curState = curState.connections[symbol]
      return outputSymbol, curState
    return self.initialState.outputSymbol, self.initialState, nextState
  
  def getLoop(self, other):
    myOutputSymbol, myState, myProcess = self.process()
    theirOutputSymbol, theirState, theirProcess = other.process()
    pairs = {}
    history = []
    while True:
      history.append((myState, theirState))
      # we are in a loop, we are good
      if (myState.index, theirState.index) in pairs:
        break
      pairs[(myState.index, theirState.index)] = len(history)-1
      (myOutputSymbolNext, myState), (theirOutputSymbolNext, theirState) = myProcess(theirOutputSymbol,myState), theirProcess(myOutputSymbol, theirState)
      myOutputSymbol, theirOutputSymbol = myOutputSymbolNext, theirOutputSymbolNext
    loopStart = pairs[(myState.index, theirState.index)]
    loopEnd = len(history)-1 # ignore the last node since it'll just be the same as the first node in the loop
    for i in range(loopStart, loopEnd):
      myState, theirState = history[i]
      yield (myState, theirState)

  def competeIMatch(self, other):
    #if not self.minimized:
    #    self.complexity()
    #    other.complexity()
    #    return self.minimizedVersion.competeIMatch(other.minimizedVersion)
    myScore = 0.0
    theirScore = 0.0
    loopSize = 0.0
    for myState, theirState in self.getLoop(other):
      if myState.outputSymbol == theirState.outputSymbol:
        myScore += 1.0
        theirScore += 0.0
      else:
        myScore += 0.0
        theirScore += 1.0
      loopSize += 1.0
    # Average over loop
    myScore /= loopSize
    theirScore /= loopSize
    return (myScore, theirScore, loopSize)

  def complexity(self):
    if self.minimizedVersion is None:
        self.minimizedVersion = self.copy()
        self.minimizedVersion.minimize()
    return len(self.minimizedVersion.states)
  
  
  def minimize(self):
    self.states, self.initialState = minimizeDFA(self.symbols, self.states, self.initialState)
    self.minimized = True
  

  def competeIDontMatch(self, other):
    otherScore, myScore, loopSize = other.competeIMatch(self)
    return (myScore, otherScore, loopSize)

  def cooperateIMatch(self, other):
      #if not self.minimized:
      #  return self.minimizedVersion.cooperateIMatch(other.minimizedVersion)
      myScore = 0.0
      theirScore = 0.0
      loopSize = 0.0
      for myState, theirState in self.getLoop(other):
        if myState.outputSymbol == theirState.outputSymbol:
          myScore += 1.0
          theirScore += 1.0
        else:
          myScore += 0.0
          theirScore += 0.0
        loopSize += 1.0
      # Average over loop
      myScore /= loopSize
      theirScore /= loopSize
      return (myScore, theirScore, loopSize)

    
  def toNx(self):
    graph = nx.MultiDiGraph()
    for state in self.states:
        graph.add_node(state.index, label=str(state.outputSymbol), isInitialState=(state == self.initialState))
    for state in self.states:
      for sym, stateConnectedTo in state.connections.items():
        graph.add_edge(state.index, stateConnectedTo.index, label=str(sym))
    return graph
    
  def toDot(self):
    dot = graphviz.Digraph()
    for state in self.states:
      if state == self.initialState:
        dot.node(str(state.index), str(state.outputSymbol), shape="box")
      else:
        dot.node(str(state.index), str(state.outputSymbol), shape="circle")
    for state in self.states:
      for sym, stateConnectedTo in state.connections.items():
        dot.edge(str(state.index), str(stateConnectedTo.index), label=str(sym))
    return dot

  def show(self, show=True):
    self.toDot().render("test-output/blah.gv", view=show)



class Species(object):
  def __init__(self, numInitialStates, alphabet, populationSize):
    self.pop = [Automata(numInitialStates, alphabet) for _ in range(populationSize)]

  def interactIMatch(self, otherSpecies, cooperate=True):
    myScores = [0.0 for _ in range(len(self.pop))]
    theirScores = [0.0 for _ in range(len(otherSpecies.pop))]
    myLoopSizes = [0.0 for _ in range(len(self.pop))]
    theirLoopSizes = [0.0 for _ in range(len(otherSpecies.pop))]
    for i, a in enumerate(self.pop):
      for j, b in enumerate(otherSpecies.pop):
        if cooperate:
            curAScore, curBScore, loopSize = a.cooperateIMatch(b)
        else:
            curAScore, curBScore, loopSize = a.competeIMatch(b)
        myScores[i] += curAScore / float(len(otherSpecies.pop))
        theirScores[j] += curBScore / float(len(self.pop))
        myLoopSizes[i] += loopSize / float(len(self.pop))
        theirLoopSizes[j] += loopSize / float(len(self.pop))
    return myScores, theirScores, myLoopSizes, theirLoopSizes

  def evolve(self, scores):
    scores = np.array(scores)
    totalScore = np.sum(scores)
    if totalScore == 0.0: # if all zero, do uniform pr
      scores = np.ones(scores.shape)/scores.shape[0]
    else:
      scores /= totalScore
    offspringFrom = np.random.choice(np.arange(len(self.pop)), size=len(self.pop), replace=True, p=scores)
    newPop = [self.pop[i].makeOffspring() for i in offspringFrom]
    self.pop = newPop


def getPops(numInitialStates,alphabet, populationSize):
  return Species(numInitialStates=numInitialStates, alphabet=alphabet, populationSize=populationSize), Species(numInitialStates=numInitialStates, alphabet=alphabet, populationSize=populationSize)

def step(A, B, cooperate=True):
  aScores, bScores, aLoopSizes, bLoopSizes = A.interactIMatch(B, cooperate=cooperate)
  A.prev = [(A.pop[i],aScores[i]) for i in  np.argsort(-np.array(aScores))]
  B.prev = [(B.pop[i],bScores[i]) for i in  np.argsort(-np.array(bScores))]
  A.evolve(aScores)
  B.evolve(bScores)
  return A, B, aScores, bScores, aLoopSizes, bLoopSizes



def doRun(steps, cooperate=True):
  A, B = getPops(numInitialStates=1, alphabet=[0,1], populationSize=25)
  
  means = []
  for i in range(steps):
    A, B, aScores, bScores, aLoopSizes, bLoopSizes = step(A, B, cooperate=cooperate)
    if i % 30 == 0:
      means.append((np.median([x.complexity() for x in A.pop]), np.median([x.complexity() for x in B.pop]), np.median(aLoopSizes), np.median(bLoopSizes)))
      print(means[-1], A.prev[0], B.prev[0], i)
  return means, A, B

