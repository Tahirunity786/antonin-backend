import numpy as np

class AbaqusInpFile:
  """
    Class used to extract data from .inp Abaqus files
  """
  lines = []
  linesPosition = {}


  def __init__(self, filepath):
    with open(filepath, "r") as fid:
      self.lines = [line.replace("\n", "") for line in fid.readlines()]
    self.parse()


  def parse(self):
    self.linesPosition["starred"] = self.linesStartingWith("*")
    self.linesPosition["Part"] = self.linesStartingWith("*Part")
    self.linesPosition["Node"] = self.linesStartingWith("*NODE")


  def linesStartingWith(self, lineBeginning):
    """
    Return an array with the indices of the lines of the file that starts with
    the string given in argument.
    """
    return np.array([index for index, line in enumerate(self.lines)
      if line.startswith(lineBeginning)])


  def linesEndingWith(self, lineEnding):
    """
    Return an array with the indices of the lines of the file that ends with
    the string given in argument.
    """
    return np.array([index for index, line in enumerate(self.lines)
      if line.endswith(lineEnding)])


  def linesContaining(self, lineContent):
    """
    Return an array with the indices of the lines of the file that contains
    the string given in argument.
    """
    return np.array([index for index, line in enumerate(self.lines)
      if lineContent in line])


  def getNodesCoordinatesOfPart(self, partName):
    """
    Return a dictionary where the keys are the part's nodes labels and the
    values are the part's nodes coordinates.
    """
    nodeLinesPosition = self.linesPosition["Node"]
    firstNodePosition = nodeLinesPosition[0] + 1


    try:
      lastNodePosition = self.linesStartingWith("******* E L E M E N T S")[0] - 1
    except:
      lastNodePosition = self.linesStartingWith("*ELEMENT")[0] - 1


    return self.getLabeledDataBetweenLines(firstNodePosition, lastNodePosition)


  def getPositionOfPartBeginning(self, partName):
    """
    Return the index of the line where the definition of the part begins.
    """
    return np.array(list(set(self.linesPosition["Part"]) & set(self.linesContaining(partName))))


  def getLabeledDataBetweenLines(self, firstLine, lastLine):
    """
    Return a dictionary where the keys are the first element of the considered
    lines and the values are the following elements.
    """
    linesElements = [line.replace(" ", "").split(",") for line in self.lines[firstLine:lastLine+1]]
    return {lineElements[0]: [float(element) for element in lineElements[1:]] for lineElements in linesElements}