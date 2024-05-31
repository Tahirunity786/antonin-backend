import numpy as np
import pandas as pd
import re

def getTabulatedProperties(obj, parentObjs=[], recursive=False, prefix="", suffix="",
                           arrays1x3AreCoordinatesXYZ=False, excludedProperties=[]):
    output = {}
    parameters = locals()
    parentObjs.append(obj)

    if isinstance(obj, (dict, np.ndarray, np.float64, np.float32, str)):
        return {}
    properties = list(obj.__dict__.keys())

    notExportableProps = ["_xAxis", "_yAxis", "_zAxis", "id4c", "dataCTPath"]
    for property_ in properties:
        propertyToTabulate = {}
        if property_ in excludedProperties or property_ in notExportableProps:
            continue

        inputProperty = getattr(obj, property_)

        # Avoid infinite loop recursion
        if isParent(inputProperty, parentObjs):
            continue
        # Recursion
        if not isinstance(inputProperty, (str, list, dict, int, float, bool, np.ndarray, np.float64, np.float32)) \
                and not inputProperty is None:
        #if True:
            if recursive:
                output.update(getTabulatedProperties(inputProperty,
                    recursive=True,
                    parentObjs=parentObjs,
                    prefix="_".join([prefix, property_]),
                    suffix=suffix,
                    arrays1x3AreCoordinatesXYZ=arrays1x3AreCoordinatesXYZ,
                    excludedProperties=excludedProperties))
            continue
        # Remove coma from strings that would corrupt the csv file
        if isinstance(inputProperty, str):
            inputProperty = inputProperty.replace(",", "")
        
        # Scalar values are straightforward
        #if isinstance(inputProperty, (int, float)) and not isinstance(inputProperty, bool):
        if np.isscalar(inputProperty) or (isinstance(inputProperty, np.ndarray)\
                                          and inputProperty.shape == (1,) \
                                          and inputProperty.shape == (1,1) \
                                          and np.isnan(inputProperty)):
            propertyToTabulate[fieldnameToColumnname(property_, prefix, suffix)] = nanIfEmpty(inputProperty)
        
        # Value is a single point coordinates 
        if isinstance(inputProperty, np.ndarray) and\
                (inputProperty.shape == (1, 3) or inputProperty.shape == (3,)) and\
                arrays1x3AreCoordinatesXYZ:
            propertyToTabulate[fieldnameToColumnname(property_, prefix, suffix)] = True
            point = inputProperty.reshape(1, 3)
            propertyToTabulate[fieldnameToColumnname(property_, prefix, "x_" + suffix)] = nanIfEmpty(point[0, 0])
            propertyToTabulate[fieldnameToColumnname(property_, prefix, "y_" + suffix)] = nanIfEmpty(point[0, 1])
            propertyToTabulate[fieldnameToColumnname(property_, prefix, "z_" + suffix)] = nanIfEmpty(point[0, 2])

        if isinstance(inputProperty, dict):
            if len(inputProperty.values()) == 3:
                point = np.array(list(inputProperty.values())).reshape(1, 3)
                if np.all(point.dtype == float):
                    propertyToTabulate[fieldnameToColumnname(property_, prefix, "x_" + suffix)] = nanIfEmpty(point[0, 0])
                    propertyToTabulate[fieldnameToColumnname(property_, prefix, "y_" + suffix)] = nanIfEmpty(point[0, 1])
                    propertyToTabulate[fieldnameToColumnname(property_, prefix, "z_" + suffix)] = nanIfEmpty(point[0, 2])

        if isinstance(inputProperty, np.ndarray) and inputProperty.shape[0] == 1 and len(inputProperty.shape) > 1 and \
                inputProperty.shape[1] > 1 and inputProperty.shape[1] != 3:
                propertyToTabulate[fieldnameToColumnname(property_,
                            prefix, suffix)] = True
                array = inputProperty
                for i in range(len(array)):
                    propertyToTabulate[fieldnameToColumnname(property_,
                            prefix, "%d_"%i+suffix)] = nanIfEmpty(array[i])
        if propertyToTabulate:
            output.update(propertyToTabulate)
    return output
            
def fieldnameToColumnname(fieldname, prefix, suffix):
    """
    Tries to extract words from a string based on the capitalized letters in the 
    string. Capitalized concatenated words is a standard way of naming variables,
    we want to extract a string containing these words modified to lower case
    and separated by the "_" character. For example:
        FooBARFooBAR -> foo_bar_foo_bar
        where the words are "foo" and "bar".
    """
    columnname = fieldname
    # FooBARFooBAR -> FooBARFoo_bar
    match = re.search("([A-Z]{2,})$", columnname)
    if not match is None:
        match = match.group().lower()
        columnname = re.sub("([A-Z]{2,})$", "_" + match, columnname)
    # FooBARFoo_bar -> FooBARF#oo_bar
    match = re.search("([A-Z]{2,})", columnname)
    if not match is None:
        columnname = re.sub("([A-Z]{2,})", match.group() + "#", columnname)
    # FooBARF#oo_bar -> FooBAR_f#oo_bar
    match = re.search("([A-Z]#)", columnname)
    if not match is None:
        match = match.group().lower()
        columnname = re.sub("([A-Z]#)", "_" + match, columnname)
    # FooBAR_f#oo_bar -> FooBAR_foo_bar
    columnname = re.sub("#", "", columnname)
    # FooBAR_foo_bar -> _foo_bar_foo_bar
    match = re.findall("([A-Z]{1,})", columnname)
    for m in match:
        columnname = re.sub(m, "_" + m.lower(), columnname)
    # _foo_bar_foo_bar -> foo_bar_foo_bar
    columnname = re.sub("^_", "", columnname)
    
    columnname = "_".join([prefix, columnname, suffix])
    columnname = columnname.strip("_")
    return columnname

def nanIfEmpty(value):
    if not value:
        return np.nan
    return value

def isParent(parsedAttribute, parentList):
    for parent in parentList:
        if isinstance(parsedAttribute, (np.ndarray, pd.Series)) or isinstance(parent, np.ndarray):
            continue
        if parsedAttribute == parent:
            return True
    return False


    
    

    
                
                    
                
                            
                            
            
            
            
        
        
            
        
        
    

