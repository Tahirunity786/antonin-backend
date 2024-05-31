from utils.LinkedList.LinkedList import LinkedList
import datetime

class Timer:
    """
     Handy timer class to start and stop mutliple timer in parallel.

     It's basically a linked list with the timers' starting timestamp
     stored in the data.

     Timer does not need to be instanciated. Simply use its start() and
     stop() static methods.

     example:

       Timer.start();
       # run some code here
       Timer.stop()

     will return the running time between the two Timer's methods call.

     example:

       timer1 = Timer.start();
       # run some code here
         timer2 = Timer.start();
         3 run some other code here
         Timer.stop(timer2);
       # run some other code here
       Timer.stop(timer1)

     Stop the timers when wanted. If no argument is given to Timer.stop()
     it will return the running time between Timer.stop() and the
     last time Timer.start() has been called.

     The Timer.format property is not used yet but it would be a nice
     feature.
    """


    data = LinkedList()
    format = {"string":"%sh %sm %ss %sms"}

    @staticmethod
    def start():
        return Timer.data.append(datetime.datetime.today())

    @staticmethod
    def stop(*args):
        if Timer.data.length == 0:
            raise "There is no timer currently running."
        if len(args) == 1:
            timerStartTime = Timer.data.pop(args[0])
        else:
            timerStartTime = Timer.data.pop
        return Timer.getElapsedTime(datetime.datetime.today()-timerStartTime)

    @staticmethod
    def clearAll(self):
        Timer.data.clear

    @staticmethod
    def getElapsedTime(duration):
        elapsedTime = duration.split()

    @staticmethod
    def getElapsedMilliseconds(duration):
        durationInMicrosecond = str(duration.microsecond)
        if not durationInMicrosecond:
            return "000000"
        return durationInMicrosecond
