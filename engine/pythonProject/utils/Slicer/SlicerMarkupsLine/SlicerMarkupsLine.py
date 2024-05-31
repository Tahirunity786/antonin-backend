from utils.Slicer.SlicerMarkups.SlicerMarkups import SlicerMarkups

class SlicerMarkupsLine(SlicerMarkups):
    def __init__(self):
        super().__init__("Line")
        self.display["glyphSize"] = 0.999
        self.display["lineThickness"] = 0.999
        self.display["pointLabelsVisibility"] = False