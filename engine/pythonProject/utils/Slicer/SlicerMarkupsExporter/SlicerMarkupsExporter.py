import numpy as np
import json

class SlicerMarkupsExporter:

    def __init__(self):
        self.markups = []

    def addMarkups(self, markups):
        if not np.isscalar(markups):
            self.addMarkups(markups[:-2])

        #assert(isinstance(SlicerMarkups, ))
        self.markups.append(markups)

    def export(self, jsonFilename):
        with open(jsonFilename, "w") as f:
            f.write("{\n")
            with open("aa.json", "w") as f:
                f.write("{\n")
                f.write(
                    '"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/\
                      Markups/Resources/Schema/markups-schema-v1.0.0.json#,\n')
                charArrayToWrite = json.dumps(self)
                # Remove first "{" and last "}" that are already manually added by the % current function
                charArrayToWrite = charArrayToWrite [1:-1]
                # % Escape special character "%"
                charArrayToWrite = charArrayToWrite.replace("%", "%%")
                f.write(charArrayToWrite)



