from utils.Logger.Logger import Logger
import os
import getConfig
import pandas as pd
import glob
import pydicom

class Patient:
    """
    """
    def __init__(self, SCase):
        self.SCase = SCase
        self.id = SCase.id # id in the anonymized database
        self.idMed = '' # id of the medical center (anonymous key, but IPP while SecTrial not active)
        self.gender = '' # M or F
        self.age = [] # year at toime of preoperative CTscan
        self.ethnicity = ''
        self.weight = []
        self.height = []
        self.BMI = []
        self.comment = ''

    def loadData(self):
        """
        Call methods that can be run after the ShoulderCase object has
        been instanciated.
        """
        return Logger().timeLogExecution("Patient data: ",
                        lambda self: self.readPatientData(), self)

    def readPatientData(self):
        self.readExcelData()
        self.readDicomData()

    def readExcelData(self):
        filename = os.path.join(getConfig.getConfig()["dataDir"],
                                "Excel", "ShoulderDataBase.xlsx")
        patientData = pd.read_excel(filename, sheet_name="SCase")
        patientData = patientData[patientData.SCase_ID == self.SCase.id]

        self.SCase.diagnosis = patientData.diagnosis_name
        self.SCase.treatment = patientData.treatment_name


    def readDicomData(self):
        dicomFiles = glob.glob(os.path.join(self.SCase.dataDicomPath(),"*.dcm"))
        filename = dicomFiles[0]
        dicomInfos = pydicom.dcmread(filename)
        self.SCase.patient.gender = dicomInfos.PatientSex

        self.age = float(dicomInfos.PatientAge[:-1])

        self.height = 100 * float(dicomInfos.PatientSize)
        self.weight = dicomInfos.PatientWeight
        self.BMI = dicomInfos.PatientWeight / dicomInfos.PatientSize**2
