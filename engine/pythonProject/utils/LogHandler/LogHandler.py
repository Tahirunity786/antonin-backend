import os
import warnings
from utils.LogWriter import LogWriter

class LogHandler:
    # Use to start an stop using a LogWriter.
    # Useful complement to static Logger. 
    def __init__(self):
        self.active = False
        self.handle = []
    
    def start(self, logFileCategory):
        if not os.path.isdir(logFileCategory):
            warnings.warn("The %s folder does not exist. Log file not created."\
                          %os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)),
                                        "log"))
            return
        if self.active:
            self.handle.closeBlock()
            self.handle.logn("LOG FILE WRITING INTERRUPTED BY STARTING A NEW LOG FILE.")
            self.stop()
            
        self.handle = LogWriter.LogWriter(logFileCategory)
        self.active = True
    
    def stop(self):
        if self.active:
            self.handle.closeLog()
            self.handle = []
            self.active = False
    
                
            

    