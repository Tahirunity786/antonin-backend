from datetime import datetime
#import logging
import warnings
import glob
import os

class LogWriter:
    """
    Class to create and manage log files.
    
     Log files are created automatically and include starting and ending timstamps.
     Methods are available to create section and delimitations in the 
     handled log file.
     
     The log file name is chosen based on the logFileCategory argument
     given to the contructor. This argument is also used to delete
     the oldest log files according to the maxNumberOfLogFiles (default = 5)
     used with a LogWriter(logFileCategory,maxNumberOfLogFiles).
    """
    def __init__(self, logFileCategory, *args):
        self.starttime = datetime.datetime.today()
        self.logFileCategory = logFileCategory
        if len(args) == 2:
            self.maxNumberOfLogFiles = args[0]
        else:
            self.maxNumberOfLogFiles = 5
        
        self.logFid = ""
        self.prefix = ""
        self.suffix = ""
        self.horizontalDelimiter = "_______________________________________"
        self.createNewLog()
    
    def log(self, textToLog, *args):
        textToLog  = textToLog.replace("\\", "/")
        textToLog = textToLog % args
        with open(self.logFid, 'a') as f:
            f.write(textToLog+self.suffix)
            
    def logn(self, textToLog, *args):
        self.suffix = self.suffix + "\n"
        try:
            self.log(textToLog, *args)
        except Exception as e:
            warnings.warn(e)
        self.suffix = self.suffix.replace("\n", "")
        with open(self.logFid, 'a') as f:
            f.write(self.suffix)
            
    def newSection(self, title):
        if self.prefix == "":
            self.newBlock()
        self.logn("")
        self.prefix = self.prefix + "  "
        self.logn(title)
    
    def newDelimitedSection(self, title):
        if self.prefix != "":
            prefixSave = self.prefix
            self.prefix = self.prefix[0] #0?
            self.logn("")
            self.logn(self.horizontalDelimiter)
            self.prefix = prefixSave
            self.logn("")
            self.prefix = self.prefix+"  "
            self.log(title)
        else:
            self.newSection(title)
        
    def closeSection(self):
        if len(self.prefix > 3):
            self.prefix = self.prefix[:-2]
            self.logn("")
        elif len(self.prefix > 0):
            self.closeBlock
    
    def closeBlock(self):
        if len(self.prefix > 0):
            self.prefix = self.prefix.replace(" ", "")
            self.logn("")
            self.log(self.horizontalDelimiter)
            self.prefix = ""
            self.logn(self.horizontalDelimiter)
            self.logn("")
            self.logn("")
    
    def closeLog(self):
        self.delete()
    
    def deleteExtraLogFiles(self):
        existingLogFiles = glob.glob(os.path.join("log",
                                                  self.logFileCategory),
                                                 "*.log")
        numberOfExtraFiles = len(existingLogFiles) - self.maxNumberOfLogFiles
        if numberOfExtraFiles > 0:
            for i in range(numberOfExtraFiles):
                os.remove(existingLogFiles[i])
                
    def _createNewLog(self):
        if not len(os.listdir("log")):
            os.mkdir("log")
        self._createLogFile()
        self.deleteExtraLogFiles()
        self._logHeader()
    
    def _createLogFile(self):
        self.logFid = open(os.path.join("log", self.logFileCategory + \
                      datetime.datetime.today().strftime("%Y_%m_%d_%H%M%S") + \
                      ".log"), "w")
            
    def _logHeader(self):
        self.logn(self.logFileCategory+".log")
        self.logn("Date: %s", str(self.startTime))
        self.logn("")
        
    def _newBlock(self):
        self.logn("")
        self.logn("")
        self.log(self.horizontalDelimiter)
        self.prefix = "|  "
        self.logn(self.horizontalDelimiter)
        self.logn("")
        
    def _delete(self):
        self.closeBlock()
        self.logn("")
        self.logn("")
        self.logn("Stop time: %s", str(datetime.datetime.today()))
        self.log("End of log file.")
        self.logFid.close()
        self.logFid = -1
        
        
    

            
                
        
        
        
            
            
        
        
        
        
    
    
            
        
        