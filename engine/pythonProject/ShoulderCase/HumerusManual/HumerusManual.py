import os
import numpy as np
import pandas as pd
from utils.Logger.Logger import Logger
from utils.Sphere.Sphere import Sphere
from utils.Vector.Vector import Vector
from utils.Plane.Plane import Plane
from ShoulderCase.fitSphere import fitSphere
# from utils.Slicer.SlicerControlPoint.SlicerControlPoint import SlicerControlPoint
# from utils.Slicer.SlicerMarkupsPlane.SlicerMarkupsPlane import SlicerMarkupsPlane
# from utils.Slicer.SlicerMarkupsExporter.SlicerMarkupsExporter import SlicerMarkupsExporter
import json
from gensim.utils import tokenize
from spellchecker import SpellChecker
from collections import OrderedDict
from ShoulderCase.Humerus.Humerus import Humerus
from ShoulderCase.Humerus.Humerus import landmarksBelongToCorrectShoulder


class HumerusManual(Humerus):
    """
    HUMERUS Properties and methods associated to the humerus
    Detailed explanation goes here

    Author: Alexandre Terrier, EPFL-LBO
    Creation date: 2018-07-01
    Revision date: 2019-06-29

    TO DO:
    Local coordinate system
    """

    def __init__(self, shoulder):
        super().__init__(shoulder)

    def loadData(self):
        """
        Call methods that can be run after the ShoulderCase object has
        been instantiated.
        """
        if self.hasSlicerLandmarks():
            success = Logger.timeLogExecution("Humerus load landmarks (slicer): ",
                                              lambda self: self.loadSlicerLandmarks(), self)
        elif self.hasAmiraLandmarks():
            success = Logger.timeLogExecution("Humerus load landmarks (amira): ",
                                              lambda self: self.loadAmiraLandmarks(), self)
        else:
            success = Logger.timeLogExecution("Humerus load landmarks: ",
                                              lambda message: raise_(Exception(message)), "No landmarks file found")
        return success

    def hasSlicerLandmarks(self):
        return os.path.isfile(os.path.join(self.shoulder.SCase.dataSlicerPath(),
                                           "HH_landmarks_fitting_sphere_" + self.shoulder.side + ".mrk.json"))

    def hasAmiraLandmarks(self):
        return os.path.isfile(os.path.join(self.shoulder.SCase.dataAmiraPath(),
                                           "newHHLandmarks" + self.shoulder.SCase.id4c + ".landmarkAscii")) \
               or os.path.isfile(os.path.join(self.shoulder.SCase.dataAmiraPath(),
                                              "HHLandmarks" + self.shoulder.SCase.id4c + ".landmarkAscii"))

    def loadSlicerLandmarks(self):
        self.landmarks = {}
        self.loadFittingSphereLandmarks()
        self.loadInsertionsLandmarks()

    def loadFittingSphereLandmarks(self):
        landmarksFilename = os.path.join(self.shoulder.SCase.dataSlicerPath(),
                                         "HH_landmarks_fitting_sphere_" + self.shoulder.side + ".mrk.json")
        self.landmarks["sphere"] = self.readSlicerLandmarks(landmarksFilename)

    def readSlicerLandmarks(self, filename):
        with open(filename) as f:
            loadedLandmarks = json.load(f)["markups"][0]["controlPoints"]
        validWordsInLabels = ["subscapularis", "supraspinatus", "infraspinatus",
                              "teres", "minor", "intratubercular"]
        landmarks = OrderedDict()
        for i in range(len(loadedLandmarks)):
            spell = SpellChecker()
            spell.unknown(validWordsInLabels)
            label = []
            for word in list(tokenize(loadedLandmarks[i]["label"].replace("_", " "))):
                label.append(spell.correction(word))
            landmarks["_".join(label)] = loadedLandmarks[i]["position"]
        return landmarks

    def loadInsertionsLandmarks(self):
        landmarksFilename = os.path.join(self.shoulder.SCase.dataSlicerPath(),
                                         "HH_landmarks_insertions_" + self.shoulder.side + ".mrk.json")
        self.landmarks["insertions"] = self.readSlicerLandmarks(landmarksFilename)

    def loadAmiraLandmarks(self):
        """
        LOAD Load 5 humeral head landmarks
        """
        assert self.hasAmiraLandmarks(), "No Amira landmarks found."

        landmarks = self.getAmiraLandmarks()

        assert landmarksBelongToCorrectShoulder(landmarks,
                                                self.shoulder), "Loaded Amira landmarks belong to the other shoulder."

        self.landmarks["landmarks"] = landmarks

        return 1

    def getAmiraLandmarks(self):
        SCase = self.shoulder.SCase

        filename = "newHHLandmarks" + SCase.id4c + ".landmarkAscii"
        filepath = os.path.join(SCase.dataAmiraPath(), filename)

        if not os.path.isfile(filepath):
            # Check for old version of humeral head landmarks
            filename = "HHLandmarks" + SCase.id4c + ".landmarkAscii"
            filepath = os.path.join(SCase.dataAmiraPath(), filename)

        assert os.path.isfile(filepath), "No Amira landmarks found."

        importedLandmarks = np.array(pd.read_table(filepath,
                                                   skiprows=[i for i in range(14)],
                                                   sep=" ",
                                                   header=None).iloc[:, :-1])
        return importedLandmarks

    def measureCenterAndRadius(self):
        """
        By fitting a sphere on humeral head landmarks
        """
        if self.hasSlicerLandmarks():
            landmarksToFit = np.array(list(self.landmarks["sphere"].values()))
            center, radius, _, _ = fitSphere(landmarksToFit)
            self.center = center.T
            self.radius = radius
        elif self.hasAmiraLandmarks():
            landmarksToFit = self.getAmiraLandmarks()
            center, radius, _, _ = fitSphere(landmarksToFit)
            self.center = center.T
            self.radius = radius

    def measureInsertionsRing(self):
        """
        The insertions ring is the intersection of the plane fitted to the
        insertions' landmarks with the humeral head fitted sphere.
        """
        insertionsArray = np.concatenate(
            [np.array(self.landmarks["insertions"]["subscapularis_inferior"]).reshape(1, -1),
             np.array(self.landmarks["insertions"]["supraspinatus_anterior"]).reshape(1, -1),
             np.array(self.landmarks["insertions"]["teres_minor_inferior"]).reshape(1, -1)],
            axis=0)
        insertionsPlane = Plane()
        insertionsPlane.fit(insertionsArray)

        ringNormal = Vector(-insertionsPlane.normal)
        humerusCenterToPlanePoint = Vector(self.center, insertionsPlane.point)

        # The ring's center is the projection of any point of the insertions' plane
        # on the vector starting at the humeral head center and with the same
        # direction as the insertion's plane normal direction.
        ringCenter = self.center + (ringNormal.dot(humerusCenterToPlanePoint) * ringNormal.vector())
        humerusCenterToRingCenter = Vector(self.center, ringCenter)
        ringRadius = np.sqrt(self.radius ** 2 - humerusCenterToRingCenter.norm() ** 2)

        self.insertionsRing["center"] = ringCenter
        self.insertionsRing["radius"] = ringRadius
        self.insertionsRing["normal"] = ringNormal

    def exportHumeralHeadSphere(self):
        exportFileName = os.path.join(self.shoulder.SCase.dataSlicerPath,
                                      "HH_fitted_sphere_" + self.shoulder.side + ".ply")

        humeralHead = Sphere()
        humeralHead.fitTo(self.landmarks.sphere.T)
        humeralHead.exportPly(exportFileName)

    """"  
    def exportInsertionsPlane(self):
        towardSubscapularis = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["subscapularis_inferior"].T)
        towardSupraspinatus = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["supraspinatus_anterior"].T)
        towardTeresMinor = 2 * Vector.Vector(self.insertionsRing.center,
            self.landmarks["insertions"]["teres_minor_inferior"].T)

        insertionsPlane = SlicerMarkupsPlane();
        insertionsPlane.addControlPoint(SlicerControlPoint("near subscapularis",
                                                           towardSubscapularis.target))
        insertionsPlane.addControlPoint(SlicerControlPoint("near supraspinatus",
                                                           towardSupraspinatus.target))
        insertionsPlane.addControlPoint(SlicerControlPoint("near teresMinor",
                                                           towardTeresMinor.target))

        exporter = SlicerMarkupsExporter()
        exporter.addMarkups(insertionsPlane)
        exportFilename = os.path.join(self.shoulder.SCase.dataSlicerPath,
            "rotator_cuff_insertions_plane_" + self.shoulder.side + ".mrk.json")
        exporter.export(exportFilename)
    """

    def getInsertionsExtremitiesByMuscle(self):
        subscapularisExtremities = np.concatenate([self.landmarks["insertions"]["subscapularis_inferior"],
                                                   self.landmarks["insertions"]["subscapularis_superior"]],
                                                  axis=1)
        supraspinatusExtremities = np.concatenate([self.landmarks["insertions"]["supraspinatus_anterior"],
                                                   self.landmarks["insertions"]["supraspinatus_infraspinatus"]],
                                                  axis=1)
        infraspinatusExtremities = np.concatenate([self.landmarks["insertions"]["supraspinatus_infraspinatus"],
                                                   self.landmarks["insertions"]["infraspinatus_teres_minor"]],
                                                  axis=1)
        teresMinorExtremities = np.concatenate([self.landmarks["insertions"]["infraspinatus_teres_minor"],
                                                self.landmarks["insertions"]["teres_minor_inferior"]],
                                               axis=1)
        insertionsExtremities = {}
        insertionsExtremities["subscapularis"] = subscapularisExtremities
        insertionsExtremities["supraspinatus"] = supraspinatusExtremities
        insertionsExtremities["infraspinatus"] = infraspinatusExtremities
        insertionsExtremities["teres_minor"] = teresMinorExtremities
        return insertionsExtremities

def raise_(ex):
    raise ex