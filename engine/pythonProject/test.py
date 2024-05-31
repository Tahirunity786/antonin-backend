from setup import setup
from measureSCase import measureSCase
from plotSCases import plotSCases
from exportSCaseData import exportSCaseData
import pickle
import numpy as np
import pandas as pd
from loadSCase import loadSCase

# run setup
setup()

#measureSCase(["P529"])

n128 = loadSCase("P281")
print(n128.shoulders['right']['auto'].scapula.glenoid.version)


#load scase
#n29 = loadSCase("N29")
#print(f"humerus radius, auto:{n29.shoulders['right']['auto'].humerus.radius}, manual:{n29.shoulders['right']['manual'].humerus.radius}")
#print(f"humerus center, auto:{n29.shoulders['right']['auto'].humerus.center}, manual:{n29.shoulders['right']['manual'].humerus.center}")
#print(f"humerus SHSIS, auto:{n29.shoulders['right']['auto'].humerus.SHSIS}, manual:{n29.shoulders['right']['manual'].humerus.SHSIS}")
#print(f"humerus GHSAmpl, auto:{n29.shoulders['right']['auto'].humerus.GHSAmpl}, manual:{n29.shoulders['right']['manual'].humerus.GHSAmpl}")
#print(f"humerus subluxationIndex3D, auto:{n29.shoulders['right']['auto'].humerus.subluxationIndex3D}, manual:{n29.shoulders['right']['manual'].humerus.subluxationIndex3D}")

#plot scase
#n458.plot("right",
#          "auto",
#          lightingFeatures=dict(ambient=0.5, diffuse=0.5, roughness = 0.9, fresnel=0.2, specular=0.6),
#          lightPosition=dict(x=0, y = 0, z=0),
#          plot_humeral_head=False,
#          plot_glenoid_surface=False)

#export csv
#exportSCaseData(["*"], "result.csv")

""""
#Examples of measures
print("radius", scc.shoulders["right"]["manual"].scapula.glenoid.radius)
print("width", scc.shoulders["right"]["manual"].scapula.glenoid.width)
print("height", scc.shoulders["right"]["manual"].scapula.glenoid.height)
print("anteroSuperiorAngle", scc.shoulders["right"]["manual"].scapula.glenoid.anteroSuperiorAngle)
print("versionAmplitude", scc.shoulders["right"]["manual"].scapula.glenoid.versionAmplitude)
print("versionOrientation", scc.shoulders["right"]["manual"].scapula.glenoid.versionOrientation)
print("version", scc.shoulders["right"]["manual"].scapula.glenoid.version)
print("inclination", scc.shoulders["right"]["manual"].scapula.glenoid.inclination)
print("retroversion", scc.shoulders["right"]["manual"].scapula.glenoid.retroversion)
print("rimInclination", scc.shoulders["right"]["manual"].scapula.glenoid.rimInclination)
print("beta", scc.shoulders["right"]["manual"].scapula.glenoid.beta)
print("density", scc.shoulders["right"]["manual"].scapula.glenoid.density)
"""

