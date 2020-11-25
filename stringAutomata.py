import random
import graphviz
import numpy as np
import networkx as nx
import itertools

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



def minimizeDFA(symbols, states, initialState):
  '''
  DFA minimization algorithm, see https://www.youtube.com/watch?v=0XaGAkY09Wc
  '''
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
  
  # continue finding equivalence classes until they have converged
  madeChange = True
  while madeChange:
    madeChange = False
    newEquivalences = []
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

  def copy(self):
    res = Automata(len(self.states), self.symbols, randomConnect=False)
    for i, state in enumerate(self.states):
      if self.initialState == state: # make sure it has the same initial state
        res.initialState = res.states[i]
      for (symbol, connectedTo) in state.connections.items(): # copy over the conections
        res.states[i].connect(symbol, res.states[connectedTo.index])
      res.states[i].outputSymbol = state.outputSymbol # copy over the output symbol
    return res
  
  def addState(self, state):
    self.states.append(state)
    
  def generate(self, maxTokens, inputGenerator=None):
    '''
    generates maxTokens+1 tokens as output, since it produces one symbol in the initial state, and then one symbol per state entered after processing an input symbol.
    inputGenerator is called with the index of the current symbol
    (for example, inputGenerator=lambda i: arr[i] if you have some array you want to pass into the automata)
    '''
    myState, myProcess = self.process()
    inputSymbols = []
    outputSymbols = [myState.outputSymbol] # we first emit the symbol of our initial state
    for t in range(maxTokens):
        if inputGenerator is None:
            inputSymbol = random.choice(self.symbols)
        else:
            inputSymbol = inputGenerator(t)
        myState = myProcess(inputSymbol, myState)
        inputSymbols.append(inputSymbol)
        outputSymbols.append(myState.outputSymbol) # we emit a symbol from the position we are now in
    return inputSymbols, outputSymbols

  def process(self):
    '''
    Creates a function you can call that hops around the automata
    Returns
    initialState, processFunc
    Then you can call
    nextState = processFunc(symbol, currentState)
    '''
    def nextState(symbol, curState):
      return curState.connections[symbol]
    return self.initialState, nextState

  def complexity(self):
    '''
    Returns the size of the smallest possible dfa that does the same thing
    '''
    if self.minimizedVersion is None:
        self.minimizedVersion = self.copy()
        self.minimizedVersion.minimize()
    return len(self.minimizedVersion.states)
  
  def minimize(self):
    '''
    Mutates this DFA into it's minimized form
    '''
    self.states, self.initialState = minimizeDFA(self.symbols, self.states, self.initialState)
    self.minimized = True
    
  def toNx(self):
    '''
    Converts into a networkx MultiDiGraph (useful for analysis)
    '''
    graph = nx.MultiDiGraph()
    for state in self.states:
        graph.add_node(state.index, label=str(state.outputSymbol), isInitialState=(state == self.initialState))
    for state in self.states:
      for sym, stateConnectedTo in state.connections.items():
        graph.add_edge(state.index, stateConnectedTo.index, label=str(sym))
    return graph
    
  def toDot(self):
    '''
    Converts into a graphviz (multi) Digraph (useful for visualization)
    '''
    dot = graphviz.Digraph()
    for state in self.states:
      if state == self.initialState:
        dot.node(str(state.index), str(state.outputSymbol), shape="octagon")
      else:
        dot.node(str(state.index), str(state.outputSymbol), shape="circle")
    for state in self.states:
      for sym, stateConnectedTo in state.connections.items():
        dot.edge(str(state.index), str(stateConnectedTo.index), label=str(sym))
    return dot

  def show(self, show=True):
    self.toDot().render("test-output/blah.gv", view=show)


def nxToDot(graph):
    '''
    Utility function to convert automata networkx MultiDiGraphs into dot graphs
    '''
    dot = graphviz.Digraph()
    for node in graph:
        attrs = graph.nodes[node]
        dot.node(str(node), label=attrs['label'])
    for node in graph:
        for edge, data in graph[node].items():
            if type(node) is nx.DiGraph: # condensed digraph
                chars = data['chars']
                for c in chars:
                    dot.edge(str(node), str(edge), label=c)
            else: # multigraph
                for _, edgeData in data.items():
                    dot.edge(str(node), str(edge), label=edgeData['label'])
    return dot

def condenseMultigraphIntoDigraph(multigraph):
    '''
    Utility function to turn multi edges into single edges with all of the characters (sorted)
    This is useful because lots of the networkx algorithms only work on digraphs, not MultiDiGraphs
    '''
    digraph = nx.DiGraph()
    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        outputSymbol = attrs['label']
        digraph.add_node(i, label=outputSymbol, isInitialState=attrs['isInitialState'])
    for i in graph:
        for j, data in graph[i].items():
            edgeLabels = "".join(sorted([x['label'] for _, x in data.items()]))
            digraph.add_edge(i, j, chars=edgeLabels)
    return digraph
    

def getAllAutomataOfSize(size, symbols):
    '''
    Potentially very slow method to get all possible automata of a given size (before minimization)
    '''
    # generate all possible values for a single node
    valueOptions = []
    valueOptions.append(list(symbols)) # symbol outputting
    for symbol in symbols:
        valueOptions.append(list(itertools.product([symbol], range(size)))) # what node to point to for each symbol
    
    allNodeOptions = itertools.product(*valueOptions)
    
    graphs = []
    # any graph simply choose one of these for each node, so do a product over them
    for j, a in enumerate(itertools.product(*[list(allNodeOptions)]*size)):
        # also, loop over which node is the initial state
        for initialState in range(size):
            curGraph = nx.MultiDiGraph()
            for i, node in enumerate(a):
                outputSymbol = node[0]
                curGraph.add_node(i, label=outputSymbol, isInitialState=(i==initialState))
            for i, node in enumerate(a):
                connections = list(node)[1:]
                for symb, connectToNode in connections:
                    curGraph.add_edge(i, connectToNode, label=symb)
            graphs.append(curGraph)
    return graphs


   
def automataFromNetworkx(graph, symbols):
    '''
    Convert a networkx graph into an automata
    '''
    states = []
    numNodes = len(graph.nodes)
    initialState = None
    for i in range(len(graph.nodes)):
        attrs = graph.nodes[i]
        outputSymbol = attrs['label']
        state = State(i, outputSymbol)
        states.append(state)
        if attrs['isInitialState']: initialState = state
    for i in graph:
        for j, data in graph[i].items():
            # different processing for condensed digraphs
            if type(graph) is nx.DiGraph:
                chars = data['chars']
                for c in chars:
                    states[i].connect(c, states[j])
            # we are a multigraph, just attach each edge
            else:
                for _, edgeData in data.items():
                    states[i].connect(edgeData['label'], states[j])
    resAutomata = Automata(numNodes, symbols, randomConnect=False)
    resAutomata.states = states
    resAutomata.initialState = initialState
    return resAutomata


def isomorphic(graphA, graphB):
    '''
    Fairly inefficient test if two graphs are isomorphic
    Works for the MultiDiGraphs, needs to be tweaked for DiGraphs (condensed graphs)
    '''
    def nodeMatch(aAttrs, bAttrs):
        return aAttrs['label'] == bAttrs['label'] and aAttrs['isInitialState'] == bAttrs['isInitialState']
    def edgeMatch(aAttrs, bAttrs):
        labelsA = sorted(list([x['label'] for _, x in aAttrs.items()]))
        labelsB = sorted(list([x['label'] for _, x in bAttrs.items()]))
        return labelsA == labelsB
    return nx.is_isomorphic(graphA, graphB, node_match=nodeMatch, edge_match=edgeMatch)


def generateRandomDataset(datasetSize, maxTrials, nStates, symbols):
    curData = []
    
    # keep making a random automata until we get one that is the right size when minimized
    def getCorrectlySizedAutomata(nStates, symbols):
        a = Automata(nStates, symbols)
        while a.complexity() != nStates:
            a = Automata(nStates, symbols)
        a.minimize()
        return a
    
    # get an automata, add it to our list as long as it's not isomorphic to anything that already exists
    for t in range(maxTrials):
        if t % 100 == 0: print(t, "/", maxTrials, " datasetSize:", len(curData))
        if len(curData) >= datasetSize: break
        curAutomata = getCorrectlySizedAutomata(nStates, symbols)
        nxAutomata = curAutomata.toNx()
        isIsomorphic = False
        for a, nxA in curData:
            if isomorphic(nxAutomata, nxA):
                isIsomorphic = True
                break
        if not isIsomorphic:
            curData.append((curAutomata, nxAutomata))
    # just return the automatas, not the networkx graphs
    return [x[0] for x in curData] 
        


def removeIsomorphic(graphs):
    '''
    Returns a list of graphs with any two graphs that are isomorphic removed
    '''
    processed = []
    notProcessed = list(graphs)
    # go through each one and only keep graphs that aren't isomorphic to it
    while len(notProcessed) > 0:
        curGraph = notProcessed.pop(0)
        print(len(notProcessed))
        notIsomorphic = []
        for graph in notProcessed:
            if not isomorphic(curGraph, graph):
                notIsomorphic.append(graph)
        notProcessed = notIsomorphic
        processed.append(curGraph)
    return processed


