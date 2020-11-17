import inspect
from collections import defaultdict

from torch.autograd.profiler import format_memory, format_time_share, format_time
from torch.autograd.profiler import FunctionEventAvg, EventList

def key_averages(eventArr, group_by_input_shapes=False, stackKey=None, stackParse=None):
    """Averages all function events over their keys.
    Modified from https://github.com/pytorch/pytorch/blob/49f0e5dfeb64a928c6a2368dd5f86573b07d20fb/torch/autograd/profiler.py#L250
    Arguments:
        eventArr: the events we are averaging over
        group_by_input_shapes: group entries by
        (event name, input shapes) rather than just event name.
        This is useful to see which input shapes contribute to the runtime
        the most and may help with size-specific optimizations or
        choosing the best candidates for quantization (aka fitting a roof line)
        stackKey: a function that takes a stack line and returns a string representing
        a key, or None if that line should not be considered
        stackParse: a function that parses the stack line into a new more helpful value,
        which will be returned as the stack line in the result event list. This function
        can return None and it'll ignore that stack line
    Returns:
        An EventList containing FunctionEventAvg objects.
    """
    eventArr.populate_cpu_children()
    stats: Dict[Tuple[int, Tuple[int, int]], FunctionEventAvg] = defaultdict(FunctionEventAvg)

    def get_key(event, group_by_input_shapes, stackKey):
        key = [str(event.key), str(event.node_id)]
        if group_by_input_shapes:
            key.append(str(event.input_shapes))
        if stackKey is not None:
            key += [stackKey(x) for x in event.stack if stackKey(x) is not None]
        return tuple(key)
    for evt in eventArr:
        stats[get_key(evt, group_by_input_shapes, stackKey)].add(evt)
    
    avg_list = EventList(stats.values(), use_cuda=eventArr._use_cuda, profile_memory=eventArr._profile_memory)
    
    for evt in avg_list:
        if not group_by_input_shapes:
            evt.input_shapes = ""
        if stackParse is not None:
            evt.model_stack = [stackParse(x) for x in evt.stack if stackParse(x) is not None]
    return avg_list


# For every line in the forward function of the given module, this returns something that looks like
def getFunctionLines(func):
    """Extracts the lines of the given function
    Arguments:
        func: The function we are extracting from
    Returns:
        This is an enumerator, it yields tuples of things that look like
        ('pathToForwardFile.py(206): forward', 206, '        embeddings = torch.cat([embs, posEmbs], axis=3)')
        Which is
        (Path(lineNum): func.__name__, lineNum, codeOnThatLine)
    """
    lines, lineNum = inspect.getsourcelines(func)
    filePath = inspect.getsourcefile(func)
    for i, line in enumerate(lines):
        yield f"{filePath}({lineNum+i}): {func.__name__}", (lineNum+i), line

def makePathMapping(model):
    """Returns a dict that can take a path string that looks like
    /home/azureuser/openai_learning/customTransformer.py(206): forward
    And that dict will return a tuple:
    (module name using .named_modles(), lineNum, codeOnLine)
    "model." is appended to the front of all module names entries so there is no empty string
    """
    mapping = {}
    for mn, m in model.named_modules():
        # add model. to front so we don't have empty string for model
        if mn == "": mn = "model"
        else: mn = "model." + mn
        for forwardPath, lineNum, line in getFunctionLines(m.forward):
            mapping[forwardPath] = (mn, lineNum, line)
    return mapping

def makePathModuleMapping(model):
    """Returns a dict that can take a path string that looks like
    /home/azureuser/openai_learning/customTransformer.py(206): forward
    And that dict will return the module name, using .named_modles()
    "model." is appended to the front of all entries so there is no empty string
    """
    mapping = {}
    for mn, m in model.named_modules():
        # add model. to front so we don't have empty string for model
        if mn == "": mn = "model"
        else: mn = "model." + mn
        for forwardPath, lineNum, line in getFunctionLines(m.forward):
            mapping[forwardPath] = mn
    return mapping




def inspectModel(model, prof):
    mappingToCodeLine = makePathMapping(model)
    mappingToModule = makePathModuleMapping(model)
    # simple callbacks for the key_averages function
    def stackToModule(stackStr):
        if stackStr in mappingToModule:
            return mappingToModule[stackStr]
        else:
            return None
    
    def stackToLine(stackStr):
        if stackStr in mappingToCodeLine:
            return mappingToCodeLine[stackStr]
        else:
            return None

    averages = key_averages(prof.function_events, stackKey=stackToModule, stackParse=stackToLine)
    return averages
    
def displayInspect(inspect, sort_key, row_limit=10):
    events = list(inspect)
    events.sort(key=sort_key)
    def stackItemToStr(stackItem):
        moduleName, lineNum, code = stackItem
        return f"{moduleName}:({lineNum})"

    output = []
    largestModuleNameSize = max([max([len(stackItemToStr(s)) for s in event.model_stack]) for event in events if len(event.model_stack) > 0])
    for event in events[::-1][:row_limit]:
        output.append(f"{event.key}: Self CPU time: {event.self_cpu_time_total} CPU Time: {event.cpu_time_total} CPU Memory Usage: {format_memory(event.cpu_memory_usage)}")
        for moduleName, lineNum, code in event.model_stack:
            moduleStr = "  " + stackItemToStr((moduleName, lineNum, code))
            padding = " "*(largestModuleNameSize-len(moduleName)) # ensures code lines all line up
            output.append(moduleStr + padding + code)
    return "\n".join(output)