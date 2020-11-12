from importlib import reload
import automataBattle
reload(automataBattle)
import itertools
import networkx as nx
import graphviz
from automataBattle import State, Automata

def nxToDot(graph):
    dot = graphviz.Digraph()
    for node in graph:
        attrs = graph.nodes[node]
        dot.node(str(node), label=str(node) + attrs['label'])
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
# Turns multi edges into single edges with all of the characters
def condenseMultigraphIntoDigraph(multigraph):
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
    def nodeMatch(aAttrs, bAttrs):
        return aAttrs['label'] == bAttrs['label'] and aAttrs['isInitialState'] == bAttrs['isInitialState']
    def edgeMatch(aAttrs, bAttrs):
        labelsA = sorted(list([x['label'] for _, x in aAttrs.items()]))
        labelsB = sorted(list([x['label'] for _, x in aAttrs.items()]))
        return labelsA == labelsB
    return nx.is_isomorphic(graphA, graphB, node_match=nodeMatch, edge_match=edgeMatch)


def generateRandomDataset(datasetSize, maxTrials, nStates, symbols):
    curData = []
    
    # keep making a random automata until we get one that is the right size when minimized
    def getCorrectlySizedAutomata(nStates, symbols):
        a = automataBattle.Automata(nStates, symbols)
        while a.complexity() != nStates:
            a = automataBattle.Automata(nStates, symbols)
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
    # just return the graphs, not the automatas
    return [x[1] for x in curData] 
        


def removeIsomorphic(graphs):
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

def minimizeGraphs(graphs, symbols):
    res = []
    for i, g in enumerate(graphs):
        if i % 100000 == 0: print(i+1, "/", len(graphs))
        a = automataBattle.automataFromNetworkx(g, symbols)
        a.minimize()
        res.append(a.toNx())
    return res
        

