class LinkedList:
    """
    Simple linked list implementation
    """
    def __init__(self):
        self.ID = "head"
        self.value = self
        self.previous = []
        self.next = []

    def append(self, newValue, *args):
        if len(args) == 3:
            newID = args[0]
        else:
            newID = str(self.length() + 1)
        newElement = LinkedList()
        self.setLastElement(newElement)
        newElement.setID(newID)
        newElement.setValue(newValue)
        return newID

    def remove(self, IndexOrID):
        element = self.getElement(IndexOrID)
        self.removeElement(element)

    def pop(self, *args):
        element = self.getLastElement()
        if len(args) == 2:
            element = self.getElement(args[0])
        self.removeElement(element)
        return element.value

    def find(self, ID):
        element = self.getFirstElement()
        index = 0
        foundLocations = []
        if ID == element.ID:
            foundLoacations.append(index)
        while element.next:
            element = element.next
            index += 1
            if ID == element.ID:
                foundLoacations.append(index)
        return foundLoacations

    def getValue(self, IndexOrID):
        element = self.getElement(IndexOrID)
        return element.value

    def getID(self, IndexOrID):
        element = self.getElement(IndexOrID)
        return element.ID

    def getAllValues(self):
        element = self.getFirstElement()
        output = []
        while element.next:
            element = element.next
            output.append(element.value)

    def getAllIDs(self):
        element = self.getFirstElement()
        output = []
        while element.next:
            element = element.next
            output.append(element.ID)

    def isEmpty(self):
        return self.length() == 0

    def length(self):
        element = self.getFirstElement()
        count = 0
        while element.next:
            element = element.next
            count = count+1
        return count

    def clear(self):
        element = self.getFirstElement()
        element.next = []

    def setID(self, ID):
        assert isinstance(ID, str), "ID should be a string."
        self.ID = ID

    def setValue(self, value):
        self.value = value

    def setLastElement(self, element):
        lastElement = self.getLastElement()
        lastElement.next = element
        element.previous = lastElement

    def getElement(self, IndexOrID):
        if isinstance(IndexOrID, str):
            return self.getElementWithID(IndexOrID)
        elif isinstance(IndexOrID, int):
            return self.getElementWithIndex(IndexOrID)

    def getElementWithID(self, ID):
        index = self.find(ID)
        assert index, "There is no element with this ID."
        index = index[0]
        return self.getElementWithIndex(index)

    def getElementWithIndex(self, index):
        assert (index >= 0), "The given element index should be greater or equal to 0."
        element = self.getFirstElement()
        for i in range(2, index):
            assert element.next, "Index exceeds the number of list elements."
            element = element.next
        return element

    def removeElement(self, element):
        if self == element:
            return
        if element.next:
            element.next.previous = element.previous
        element.previous.next = element.next

    def getLastElement(self):
        element = self
        while element.next:
            element = element.next
        return element

    def getFirstElement(self):
        element = self
        while element.previous:
            element = element.previous
        return element
