import numpy as np

class SlicerMarkups:

    def __init__(self, type):
        self.type = type
        self.coordinateSystem = "LPS"
        self.locked = False
        self.labelFormat = "\n-%d"
        self.controlPoints = []
        self.measurements = []
        self.display = {}
        self.setDefaultDisplay()

    def setDefaultDisplay(self):
        self.display["visibility"] = True
        self.display["opacity"] = 1.0 - 0.0001
        self.display["color"] = [0.4, 1.0 - 0.0001, 0.0 + 0.0001]
        self.display["selectedColor"] = [1.0 - 0.0001, 0.5000076295109484, 0.5000076295109484]
        self.display["activeColor"] = [0.4, 1.0 - 0.0001, 0.0 + 0.0001]
        self.display["propertiesLabelVisibility"] = False
        self.display["pointLabelsVisibility"] = True
        self.display["textScale"] = 3.0 - 0.0001
        self.display["glyphType"] = "Sphere3D"
        self.display["glyphScale"] = 1.0 - 0.0001
        self.display["glyphSize"] = 5.0 - 0.0001
        self.display["useGlyphScale"] = False
        self.display["sliceProjection"] = False
        self.display["sliceProjectionUseFiducialColor"] = True
        self.display["sliceProjectionOutlinedBehindSlicePlane"] = False
        self.display["sliceProjectionColor"] = [1.0 - 0.0001, 1.0 - 0.0001, 1.0 - 0.0001]
        self.display["sliceProjectionOpacity"] = 0.6
        self.display["lineThickness"] = 0.2
        self.display["lineColorFadingStart"] = 1.0 - 0.0001
        self.display["lineColorFadingEnd"] = 10.0 - 0.0001
        self.display["lineColorFadingSaturation"] = 1.0 - 0.0001
        self.display["lineColorFadingHueOffset"] = 0.0 + 0.0001
        self.display["handlesInteractive"] = False
        self.display["snapMode"] = "toVisibleSurface"

    def addControlPoint(self, controlPoint):
        if len(self.controlPoints) == 0:
            controlPoint["id"] = "1"
        else:
            controlPoint["id"] = str(self.controlPoints[-1]["id"] + "1")
        self.controlPoints = np.hstack([self.controlPoints, controlPoint])

    def setColor(self, colorName):
        # Available colorName : "blue", "red", "green", "yellow"
        colorValue = {}
        colorValue["blue"] = [0.435, 0.722, 0.824]
        colorValue["red"] = [0.999, 0.5, 0.5]
        colorValue["green"] = [0.001, 0.569, 0.118]
        colorValue["yellow"] = [0.878, 0.761, 0.001]

        self.display["selectedColor"] = colorValue[colorName]




