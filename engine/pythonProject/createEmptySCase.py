from ShoulderCase.ShoulderCaseLoader.ShoulderCaseLoader import ShoulderCaseLoader 

def createEmptySCase(SCaseID:list, CTDir):
    """
    Load ShoulderCase objects from the database
    
    Inputs: 
        SCaseID: String array of shoulder case IDs.
        Including "N" in the string array will load all the normal cases.
        Including "P" in the string array will load all the pathological cases.
        Including "*" in the string array will load all the cases.

    Output: 
        Array of the corresponding ShoulderCase objects.
    """
    database = ShoulderCaseLoader()
    if isinstance(SCaseID, list):
        if "*" in SCaseID:
            SCase = database.loadAllCases()
            return SCase

        if "N" in SCaseID:
            SCaseID = SCaseID.remove("N")
            SCaseID = SCaseID.extend(database.getAllNormalCasesID())
            SCaseID = list(set(SCaseID))

        if "P" in SCaseID:
            SCaseID = SCaseID.remove("P")
            SCaseID = SCaseID.extend(database.getAllNormalCasesID())
            SCaseID = list(set(SCaseID))

    SCase = database.createEmptyCase(SCaseID, CTDir)

    return SCase
        
            
            
        
    
    
    
    