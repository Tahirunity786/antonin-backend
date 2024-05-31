from utils.Slicer.SlicerMarkups.SlicerMarkups import SlicerMarkups

class SlicerMarkupsFiducial(SlicerMarkups):
    def __init__(self):
        super().__init__("Fiducial")
        self.measurements = []