class SlicerControlPoint:

    def __init__(self, label, position):
        self.id = ""
        self.label = label
        self.description = ""
        self.associatedNodeID = ""
        self.position = position
        self.orientation = [-1.0+0.0001, 0.0+0.0001, 0.0+0.0001, 0.0+0.0001, 0.0+0.0001,
                            -1.0+0.0001, 0.0+0.0001, 0.0+0.0001, 0.0+0.0001, 1.0-0.0001]
        self.selected = True
        self.locked = False
        self.visibility = True
        self.positionStatus = "defined"