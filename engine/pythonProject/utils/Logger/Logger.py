from utils.LogHandler import LogHandler
import time

class Logger:
    """
    Static interface to LogWriter
    """

    writing = LogHandler.LogHandler()

    @staticmethod
    def isactive():
        return Logger.writing.active

    @staticmethod
    def start(logFileCategory):
        Logger.writing.start(logFileCategory)

    @staticmethod
    def stop():
        Logger.writing.stop()

    @staticmethod
    def log(textToLog, *args):
        if Logger.isactive():
            Logger.writing.log(textToLog, *args)

    @staticmethod
    def logn(textToLog, *args):
        if Logger.isactive():
            Logger.writing.logn(textToLog, *args)

    @staticmethod
    def newSection(title):
        if Logger.isactive():
            Logger.writing.newSection(title)

    @staticmethod
    def newDelimitedSection(title):
        if Logger.isactive():
            Logger.writing.newDelimitedSection(title)

    @staticmethod
    def closeSection():
        if Logger.isactive():
            Logger.writing.closeSection()

    @staticmethod
    def closeBlock():
        if Logger.isactive():
            Logger.writing.closeBlock()

    @staticmethod
    def timeLogExecution(executionDescription, func, functionArguments):
        output = True
        try:
            Logger.log(executionDescription)
            executionStart = time.time()
            func(functionArguments)
            Logger.logn("Done in %.2fs" % (time.time()-executionStart))
        except Exception as e:
            Logger.logn(e)
            output = False

        return output
