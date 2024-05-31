class UpdatableText:
    """
    Beware not to print anything else in the command window while
    an UpdatabaleText is being used since updating an UpdatableText
    deletes characters among the last printed characters.
    """
    def __init__(self, *args):
        self.lastPrintLength = 0
        self.prefix = ""
        self.text = ""
        self.percent = ""
        self.progressBar = ""
        self.suffix = "\n"
        if len(args) >= 1:
            assert isinstance(args[0], str), "The first argument must be a char array to set the prefix."
            self.prefix = args[0]
        if len(args) >= 2:
            assert isinstance(args[1], str), "The second argument must be a char array to set the suffix."
            self.suffix = args[1] + "\n"
        
    def erase(self):
        print(self.getFormerTextPrintableDeleter())
        self.lastPrintLength = 0
    
    def getFormerTextPrintableDeleter(self):
        return "\b"*self.lastPrintLength
    
    def print_(self, *args):
        """
        The text to print after the prefix is the
        only optional argument.
        """
        if len(args) == 2:
            self.text = args[0]
        
        newPrintText = self.getFormerTextPrintableDeleter() + self.prefix + \
            self.text + self.percent + self.progressBar + self.suffix
        
        print(newPrintText)
        newPrintLength = len(newPrintText) 
        
        self.lastPrintLength = newPrintLength - self.lastPrintLength
        
    def printAbove(self, text):
        self.erase()
        print(text)
        self.print_()
    
    def printPercent(self, fraction):
        self.percent = "(" + "%.2f" % (100*fraction) + "%)" 
        self.print_()
        
    def printProgressBar(self, fraction, *args):
        """
        The length of the progress bar is the only optional argument.
        The progress bar default length is 20 characters.
        """
        if len(args) == 3:
            progressBarLength = args[0]
        else:
            progressBarLength = 20
            
        numberOfProgressCharacters = int(progressBarLength*fraction)
        self.progressBar = "[" + "="*numberOfProgressCharacters + \
            " "*(progressBarLength-numberOfProgressCharacters) + "]"
        
        self.print_()
            
            
        
        
        
        
        
        
        
        
        
        
        