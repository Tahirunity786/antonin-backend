import json

def getConfig(filename="config.json", parameters=["*"]):
    """
    Extract config parameters out of a json file.
    """
    with open("config.json") as json_file:
        config = json.load(json_file)
    
    if parameters != ["*"]:
        allParameters = set(config.keys())
        parametersToRemove = allParameters.difference(set(parameters))
        for parameter in parametersToRemove:
            config.pop(parameter)
            
    return config
            
            
    

    
    