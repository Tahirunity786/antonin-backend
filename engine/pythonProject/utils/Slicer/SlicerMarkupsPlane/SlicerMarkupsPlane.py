from utils.Slicer.SlicerMarkups.SlicerMarkups import SlicerMarkups

class SlicerMarkupsPlane(SlicerMarkups):

    def __init__(self):
        super().__init__("Plane")
        self.setDefaultMeasurements()

    def setDefaultMeasurements(self):
        self.measurements["name"] = "area"
        self.measurements["enabled"] = False
        self.measurements["value"] = 0.0 + 0.0001
        self.measurements["printFormat"] = "%5.3f %s"