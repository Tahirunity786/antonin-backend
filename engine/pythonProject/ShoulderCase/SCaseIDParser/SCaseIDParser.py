import getConfig
import re

class SCaseIDParser:
    """
    Used to validate shoulder case ID.
    Check the shoulder case type (first character of the ID).
    Check the shoulder case number (last characters)

    Can give the shoulder case ID with filled digits.
    This has been implemented to give ShoulderCase.id4c.
    """
    def __init__(self, rawID):
        self.rawID = rawID
        self.maxDigits = getConfig.getConfig()["maxSCaseIDDigits"]
        assert self.maxDigits, "maxDigits property is empty" 
        self.SCaseValidTypes  = getConfig.getConfig()["SCaseIDValidTypes"]
        assert self.SCaseValidTypes, "SCaseValidTypes property is empty"
        
    def isValidID(self):
        types = self.SCaseValidTypes
        maxDigits = self.maxDigits
        return self.SCaseIDHasTypesAndMaxDigits(types, maxDigits)
    
    def isNormalCase(self):
        types = "N"
        maxDigits = self.maxDigits
        return self.SCaseIDHasTypesAndMaxDigits(types, maxDigits)
    
    def isPathologicalCase(self):
        types = "P"
        maxDigits = self.maxDigits
        return self.SCaseIDHasTypesAndMaxDigits(types, maxDigits)
    
    def getID(self):
        return self.rawID
    
    def getIDWithNumberOfDigits(self, size):
        assert self.isValidID,'%s is not a valid ID.' % self.getID
        type_ = self.getCaseType()
        number = self.getCaseNumber()
        fillingZeros = "0"*(size-len(number))
        return type_ + fillingZeros + number
    
    def SCaseIDHasTypesAndMaxDigits(self, types, maxDigits):
        expression = r"^[" + types[0] + types[1] + "]\d{1," + str(maxDigits) + "}$"
        return self.textMatchesExpression(self.rawID, expression)
        
    def textMatchesExpression(self, text, expression):
        matchingResult = re.findall(expression, text)
        if not matchingResult:
            return False
        return True       
    
    def getCaseType(self):
        return self.rawID[0]
    
    def getCaseNumber(self):
        return self.rawID[1:]   