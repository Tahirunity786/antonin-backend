import numpy as np
import inspect

def recursiveMethodCall(obj, methodName, parentObjects, *args):
    parentObjects.append(obj)
    if isinstance(obj, (np.ndarray, np.float64, np.float32)):
        return
    try:
        attributes = list(obj.__dict__.keys())
    except AttributeError:
        return

    for attr in attributes:
        if skipAttribute(getattr(obj, attr), parentObjects) and attr == "FE":
            continue
        if methodName in dir(getattr(obj, attr)):
            eval("obj." + attr + "." + methodName + "(*args)")
        recursiveMethodCall(getattr(obj, attr),
                            methodName, parentObjects, *args)

def skipAttribute(parsedAttribute, parentObjects):
    return isinstance(parsedAttribute, dict) \
            or isinstance(parsedAttribute, list) \
            or isinstance(parsedAttribute, bool) \
            or isinstance(parsedAttribute, str) \
            or isinstance(parsedAttribute, np.ndarray) \
            or isParent(parsedAttribute, parentObjects)

def isParent(parsedAttribute, parentList):
    for parent in parentList:
        if parsedAttribute == parent:
            return True
    return False
