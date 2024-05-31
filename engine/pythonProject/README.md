# Shoulder Database handling and measurements

## Introduction
This repository contains Python codes to import anonymised clinical cases in a shoulder database (shoulderDB), and update this database with complementary clinical, radiological, morphological, or biomechanical data. This project is a collaboration between Laboratory of Biomechanical Orthopedics of the Ecole Polytechnique Federal de Lausanne (EPFL-LBO), the ARTORG center of the University of Bern, the Orthopedic Service of the University Hospital of Lausanne (CHUV-OTR), and the Radiological Department of the University Hospital of Lausanne (CHUV-RAD).

## Setup

### Download

Clone or download this repository.
Open Python, set the working directory to be the downloaded repository and run setup.py.
This setup() function will create a config.json file that is detailed below.

### Write the config file

The base folder must contain a "config.json" with the following keys:

* maxSCaseIDDigits: This is the maximum allowed number of digits in the SCase IDs.

        "maxSCaseIDDigits": 3 -> 'P123' is a valid ID, 'P1234' is not valid.

* SCaseIDValidTypes: These are the allowed letters used as a prefix in the SCase IDs.                   

        "SCaseIDValidTypes": ["N", "P"] -> 'P123' is a valid ID, 'Z123' is not valid.

* pythonDir: This is the path to the rotator cuff segmentation system.              

* dataDir: This is the root path to the data used i.e. "/shoulderDev/dataDev" or "/shoulder/data" on lbovenus.epfl.ch

* casesToMeasure: Array of cases ID to be measured when executing measureSCase() (with no argument).

        "casesToMeasure": "*" -> measureSCase() will measure all the cases found in the database.
        "casesToMeasure": "P1" -> measureSCase() will measure P1 if it's in the database.
        "casesToMeasure": ["P1", "P2"] -> measureSCase() will measure P1 and P" if they are found in the database.

* shouldersToMeasure: Hash table used by measureSCase() where keys correspond to the four available shoulders of a case and values are booleans allowing the shoulders to be measured. For example:

        "shouldersToMeasure" {
        "rightAuto": true,
        "rightManual": false,
        "leftAuto": true,
        "leftManual": false
        }

* standardMeasurementsToRun: Hash table used by measureSCase() where keys correspond to the standard measurement steps of measureSCase() and values are booleans allowing the steps to be run. For example:

        "standardMeasurementsToRun":{
        "loadData":true,
        "morphology":true,
        "measureFirst":true,
        "measureSecond":false
        }

* sliceRotatorCuffMuscles: For slicing rotator cuffs, this should be True. The numberOfObliqueSlices key can be 1, 3 or 10. The oblique slice(s) can be cropped or not which is determined by croppedRotatorCuff key.

These standard measurements are built in a way that the next measurement needs the results of the former measurement, e.g  "morphology" needs to be run after "loadData". The successive standard measurements are "loadData", "morphology", "measureFirst", "measureSecond".

* specialMeasurementsToRun: Hash table used by measureSCase() where keys correspond to specific SCase functions and values are booleans allowing these functions to be called. For example:

        "specialMeasurementsToRun":{
        "sliceAndSegment":true,
        "calcDensity":false
        }

All the special measurements are called after the execution of the standard measurements. The special measurements SCase functions are recursively called in a Shoulder object and all the objects within.

* specialMeasurementsArguments: Hash table used by measureSCase() where keys correspond to the specific SCase functions listed in "specialMeasurementsToRun" and values are arrays of arguments to give to the functions for their execution. For example:

        "specialMeasurementsArguments":{
        "sliceAndSegment":[],
        "calcDensity":[]
        }

* numberOfMuscleSubdivisions: Hash table used by measureSCase() where keys correspond to the muscle divisions dimensions and values are the number of these divisions. For example:

        "numberOfMuscleSubdivisions":{
        "radial": 10,
        "angular": 10
        }

* overwriteMeasurements: Boolean that will tell measureSCase() to create an empty SCase (true) or to load an existing SCase (false) before measuring it.

* saveMeasurements: Boolean used by measureSCase() to save or not each individual SCase.mat after measurements.

* saveAllMeasurementsInOneFile: Boolean used by measureSCase() to save all the SCase.mat loaded from the whole database into one file in dataDir/matlab/SCaseDB.mat. This won't take into account the last measurements if "saveMeasurements" is false.

## How to use

There are five main features within this system.
Be sure to add this system's folders and subfolders to your Python path.

You need some python libraries for this code.
Please install them with:

    python -m pip install -r requirements.txt

Which installs the libraries listed in the requirements.txt file.


First you need to run the setup.py file.
Run this file in the python console or in a seperated .py file.

For measurements, you need to run the measureScase function which is in measureSCase file.
You can import it with:

    from measureSCase import measureScase

Example: 

    measureScase(["N161"])

All of the measures will be saved in a SCase.pkl file in the python subdirectory of every case.

After measurement for one case, you can have a plot of it which includes scapula landmarks, coordinate system, glenoid surface and its center line.
This can be done as follows

    n213 = loadSCase("N213")

    n213.plot("right", "auto")

By default, it will save the html file of the plot in your working directory.
You can change the address for saving the plot as follows:

    n213.plot("right", "auto", pathToSavePlot="<the direction you want to save the plot>")

You may also want to change the colors as follows:
    
    n213.plot("right",
              "auto",
              landmarksColor="green",
              scapulaSurfaceColor="rgb(20, 20, 20)",
              glenoidSurfaceColor="rgb(250, 0, 250)",
              humeralHeadColor="fall")

You have control on the lighting features and the position of the light source for plotting scapula surface by setting the lightingFeatures and the lightPosition arguments as a dict as follows:

    n231.plot("right",
              "auto",
              lightingFeatures=dict(ambient=0.5, diffuse=0.5, roughness = 0.9, fresnel=0.2, specular=0.6),
              lightPosition=dict(x=0, y = 0, z=0))

For more information regarding these variables please check the following link:
https://plotly.com/python/v3/3d-surface-lighting/

If you don't want the humeral head and glenoid surface in your plot, you can specify it in the arguments as follows:
    
    n213.plot("right",
              "auto",
              landmarksColor="green",
              scapulaSurfaceColor="rgb(20, 20, 20)",
              glenoidSurfaceColor="rgb(250, 0, 250)",
              humeralHeadColor="fall",
              plot_humeral_head=False,
              plot_glenoid_surface=False)

For colors you may check plotly:

https://plotly.com/python/builtin-colorscales/

https://plotly.com/python/discrete-color/

Or you can plot a list of cases with plotSCases function as:

    plotSCases(["N419", "N418"])

To save the measurements in a csv file you can run the exportSCaseData.py file.

Example: 
    
    exportSCaseData(["N161"], "n161_results.csv")

The first argument is the list of cases for which you want to save csv file ("*" for all of the cases).
The second argument is the csv file in which the results will be saved.



#### Access data

Data are restricted to authorised users, in a dedicated EPFL server.
The data related to a specific case can be accessed thanks to the ShoulderCaseLoader class. For example:



#### Visualize data

All the measured data are directly available in a loaded ShoulderCase's properties.


#### Update data

To update the measured values of the shoulder cases use the function measureSCase(). Refer to the function's documentation for further informations.

#### Import new data

To import new cases (basically add a CT-files folder at the right place) use the importSCase GUI. The importSCase code is located in the sub-directory importSCase. Its description is detailed there. It currently only work for Windows.

#### Export data



The summary of every exported variable definition is given below.
WARNING: The following variables descriptions are defined for v2.1.0 of present code.
DEADVAR indicates a variable to which no value is assigned by the code
DUPLICATE indicates a variable which duplicates the data contained somewhere else


* id: ShoulderCase.id is the id of the shoulder case as it appears in the database [NP][0-9]{1,3}

* diagnosis: ShoulderCase.diagnosis is the given diagnosis (found in REDCap export)

* treatment: ShoulderCase.treatment is the given treatment

* data_ct_path: ShoulderCase.dataCTPath is the path to the folder containing the dicom folder and other data related to the case (Amira/3DSlicer segmentations, rotator cuff segmentations, matlab measurements,...)

* patient_id: Patient.id is the id in the anonymised database (ShoulderCase.id DUPLICATE)

* patient_id_med: Patient.idMed is the id of the medical center DEADVAR

* patient_gender: Patient.gender is the patient's gender found in dicom metadata

* patient_age: Patient.age is the patient's age at time of preoperative CT scan found in dicom metadata

* patient_ethnicity: Patient.ethnicity is a medical metadata DEADVAR

* patient_weight: Patient.weight is the patient's height found in dicom metadata

* patient_height: Patient.height is the patient's weight found in dicom metadata

* patient_BMI: Patient.BMI is the patient's Body Mass Index computed with dicom metadata (BMI = weight / height^2)

* patient_comment: Patient.comment DEADVAR

* shoulder_side: Shoulder.side is the considered shoulder side ("R"/"L" tells Right/Left)

* shoulder_landmarks_acquisition: Shoulder.landmarksAcquisition is the method used to obtain scapula landmarks ("auto" or "manual")

* shoulder_has_measurements: Shoulder.hasMeasurements indicates the presence of scapula landmarks (1/0 tells presence/absence)

* shoulder_ct_scan: Shoulder.CTScan DEADVAR

* shoulder_comment: Shoulder.comment DEADVAR

* shoulder_scapula_angulus_inferior: Scapula.angulusInferior indicates the presence of angulus inferior landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_angulus_inferior_x: Scapula.angulusInferior gives the x coordinates of angulus inferior landmark

* shoulder_scapula_angulus_inferior_y: Scapula.angulusInferior gives the y coordinates of angulus inferior landmark

* shoulder_scapula_angulus_inferior_z: Scapula.angulusInferior gives the z coordinates of angulus inferior landmark

* shoulder_scapula_trigonum_spinae: Scapula.trigonumSpinae indicates the presence of trigonum spinae landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_trigonum_spinae_x: Scapula.trigonumSpinae gives the x coordinates of trigonum spinae landmark

* shoulder_scapula_trigonum_spinae_y: Scapula.trigonumSpinae gives the y coordinates of trigonum spinae landmark

* shoulder_scapula_trigonum_spinae_z: Scapula.trigonumSpinae gives the z coordinates of trigonum spinae landmark

* shoulder_scapula_processus_coracoideus: Scapula.processusCoracoideus indicates the presence of processus coracoideus landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_processus_coracoideus_x: Scapula.processusCoracoideus gives the x coordinates of processus coracoideus landmark

* shoulder_scapula_processus_coracoideus_y: Scapula.processusCoracoideus gives the y coordinates of processus coracoideus landmark

* shoulder_scapula_processus_coracoideus_z: Scapula.processusCoracoideus gives the z coordinates of processus coracoideus landmark

* shoulder_scapula_acromio_clavicular: Scapula.acromioClavicular indicates the presence of acromio clavicular landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_acromio_clavicular_x: Scapula.acromioClavicular gives the x coordinates of acromio clavicular landmark

* shoulder_scapula_acromio_clavicular_y: Scapula.acromioClavicular gives the y coordinates of acromio clavicular landmark

* shoulder_scapula_acromio_clavicular_z: Scapula.acromioClavicular gives the z coordinates of acromio clavicular landmark

* shoulder_scapula_angulus_acromialis: Scapula.angulusacromialis indicates the presence of angulus acromialis landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_angulus_acromialis_x: Scapula.angulusacromialis gives the x coordinates of angulus acromialis landmark

* shoulder_scapula_angulus_acromialis_y: Scapula.angulusacromialis gives the y coordinates of angulus acromialis landmark

* shoulder_scapula_angulus_acromialis_z: Scapula.angulusacromialis gives the z coordinates of angulus acromialis landmark

* shoulder_scapula_spino_glenoid_notch: Scapula.spinoGlenoidNotch indicates the presence of spino glenoid notch landmark coordinates (1/0 tells presence/absence)

* shoulder_scapula_spino_glenoid_notch_x: Scapula.spinoGlenoidNotch gives the x coordinates of spino glenoid notch landmark

* shoulder_scapula_spino_glenoid_notch_y: Scapula.spinoGlenoidNotch gives the y coordinates of spino glenoid notch landmark

* shoulder_scapula_spino_glenoid_notch_z: Scapula.spinoGlenoidNotch gives the z coordinates of spino glenoid notch landmark

* shoulder_scapula_friedmansLine_origin: Scapula.friedmansLine indicates the presence of friedmansLine vector origin coordinates (1/0 tells presence/absence)

* shoulder_scapula_friedmansLine_origin_x: Scapula.friedmansLine gives the x coordinates of friedmansLine vector origin

* shoulder_scapula_friedmansLine_origin_y: Scapula.friedmansLine gives the y coordinates of friedmansLine vector origin

* shoulder_scapula_friedmansLine_origin_z: Scapula.friedmansLine gives the z coordinates of friedmansLine vector origin

* shoulder_scapula_friedmansLine_target: Scapula.friedmansLine indicates the presence of friedmansLine vector target coordinates (1/0 tells presence/absence)

* shoulder_scapula_friedmansLine_target_x: Scapula.friedmansLine gives the x coordinates of friedmansLine vector target

* shoulder_scapula_friedmansLine_target_y: Scapula.friedmansLine gives the y coordinates of friedmansLine vector target

* shoulder_scapula_friedmansLine_target_z: Scapula.friedmansLine gives the z coordinates of friedmansLine vector target

* shoulder_scapula_coordSys_ml: CoordinateSystemAnatomical.ML indicates the presence of medio-lateral axis coordinates (1/0 tells presence/absence)

* shoulder_scapula_coordSys_ml_x: CoordinateSystemAnatomical.ML gives the x coordinates of medio-lateral axis

* shoulder_scapula_coordSys_ml_y: CoordinateSystemAnatomical.ML gives the y coordinates of medio-lateral axis

* shoulder_scapula_coordSys_ml_z: CoordinateSystemAnatomical.ML gives the z coordinates of medio-lateral axis

* shoulder_scapula_coordSys_pa: CoordinateSystemAnatomical.PA indicates the presence of postero_anterior axis coordinates (1/0 tells presence/absence)

* shoulder_scapula_coordSys_pa_x: CoordinateSystemAnatomical.PA gives the x coordinates of postero_anterior axis

* shoulder_scapula_coordSys_pa_y: CoordinateSystemAnatomical.PA gives the y coordinates of postero_anterior axis

* shoulder_scapula_coordSys_pa_z: CoordinateSystemAnatomical.PA gives the z coordinates of postero_anterior axis

* shoulder_scapula_coordSys_is: CoordinateSystemAnatomical.IS indicates the presence of infero-superior axis coordinates (1/0 tells presence/absence)

* shoulder_scapula_coordSys_is_x: CoordinateSystemAnatomical.IS gives the x coordinates of infero-superior axis

* shoulder_scapula_coordSys_is_y: CoordinateSystemAnatomical.IS gives the y coordinates of infero-superior axis

* shoulder_scapula_coordSys_is_z: CoordinateSystemAnatomical.IS gives the z coordinates of infero-superior axis

* shoulder_scapula_coordSys_origin: CoordinateSystemAnatomical.origin indicates the presence of the coordinate system's origin coordinates (1/0 tells presence/absence)

* shoulder_scapula_coordSys_origin_x: CoordinateSystemAnatomical.origin gives the x coordinates of the coordinate system's origin

* shoulder_scapula_coordSys_origin_y: CoordinateSystemAnatomical.origin gives the y coordinates of the coordinate system's origin

* shoulder_scapula_coordSys_origin_z: CoordinateSystemAnatomical.origin gives the z coordinates of the coordinate system's origin

* shoulder_scapula_plane_normal: Plane.normal indicates the presence of plane's normal vector coordinates (1/0 tells presence/absence)

* shoulder_scapula_plane_normal_x: Plane.normal gives the x coordinates of plane's normal vector

* shoulder_scapula_plane_normal_y: Plane.normal gives the y coordinates of plane's normal vector

* shoulder_scapula_plane_normal_z: Plane.normal gives the z coordinates of plane's normal vector

* shoulder_scapula_plane_point: Plane.point indicates the presence of plane's point coordinates (1/0 tells presence/absence)

* shoulder_scapula_plane_point_x: Plane.point gives the x coordinates of plane's point

* shoulder_scapula_plane_point_y: Plane.point gives the y coordinates of plane's point

* shoulder_scapula_plane_point_z: Plane.point gives the z coordinates of plane's point

* shoulder_scapula_plane_fitPerformance_rmse: Plane.fitPerformance.RMSE is the root mean square error of the points fitted by the plane and the plane itself

* shoulder_scapula_plane_fitPerformance_r2: Plane.fitPerformance.R2 is the coefficient of determination of the points to plane distances

* shoulder_scapula_segmentation: Scapula.segmentation indicates the type of scapula surface segmentation ("N"/"M"/"A" tells None/Manual/Automatic)

* shoulder_scapula_glenoid_center: Glenoid.center indicates the presence of glenoid's center coordinates (1/0 tells presence/absence). Glenoid's center is the point of the segmented glenoid's surface which is the closest to the mean of all the points of the segmeted glenoid's surface 

* shoulder_scapula_glenoid_center_x: Glenoid.center gives the x coordinates of glenoid's center

* shoulder_scapula_glenoid_center_y: Glenoid.center gives the y coordinates of glenoid's center

* shoulder_scapula_glenoid_center_z: Glenoid.center gives the z coordinates of glenoid's center

* shoulder_scapula_glenoid_centerLocal: Glenoid.centerLocal indicates the presence of glenoid's center coordinates expressed in scapula's coordinate system (1/0 tells presence/absence)

* shoulder_scapula_glenoid_centerLocal_x: Glenoid.centerLocal gives the x coordinates of glenoid's center expressed in scapula's coordinate system

* shoulder_scapula_glenoid_centerLocal_y: Glenoid.centerLocal gives the y coordinates of glenoid's center expressed in scapula's coordinate system

* shoulder_scapula_glenoid_centerLocal_z: Glenoid.centerLocal gives the z coordinates of glenoid's center expressed in scapula's coordinate system

* shoulder_scapula_glenoid_radius: Glenoid.radius is the radius is the radius of the sphere fitted to the glenoid's surface points

* shoulder_scapula_glenoid_center_line: Glenoid.centerLine indicates the presence of glenoid's center line coordinates (1/0 tells presence/absence). Glenoid's center line is the vector from glenoid's center to glenoid's fitted sphere center

* shoulder_scapula_glenoid_center_line_x: Glenoid.centerLine gives the x coordinates of glenoid's center line

* shoulder_scapula_glenoid_center_line_y: Glenoid.centerLine gives the y coordinates of glenoid's center line

* shoulder_scapula_glenoid_center_line_z: Glenoid.centerLine gives the z coordinates of glenoid's center line

* shoulder_scapula_glenoid_posteroAnteriorLine_origin: Glenoid.posteroAnteriorLine.origin indicates the presence of glenoid's postero-anterior line origin coordinates (1/0 tells presence/absence). Glenoid's postero-anterior origin is the most posterior point of the glenoid's rim points

* shoulder_scapula_glenoid_posteroAnteriorLine_origin_x: Glenoid.posteroAnteriorLine.origin gives the x coordinates of glenoid's postero-anterior line origin

* shoulder_scapula_glenoid_posteroAnteriorLine_origin_y: Glenoid.posteroAnteriorLine.origin gives the y coordinates of glenoid's postero-anterior line origin

* shoulder_scapula_glenoid_posteroAnteriorLine_origin_z: Glenoid.posteroAnteriorLine.origin gives the z coordinates of glenoid's postero-anterior line origin

* shoulder_scapula_glenoid_posteroAnteriorLine_target: Glenoid.posteroAnteriorLine.target indicates the presence of glenoid's postero-anterior line target coordinates (1/0 tells presence/absence). Glenoid's postero-anterior target is the most anterior point of the glenoid's rim points

* shoulder_scapula_glenoid_posteroAnteriorLine_target_x: Glenoid.posteroAnteriorLine.target gives the x coordinates of glenoid's postero-anterior line target

* shoulder_scapula_glenoid_posteroAnteriorLine_target_y: Glenoid.posteroAnteriorLine.target gives the y coordinates of glenoid's postero-anterior line target

* shoulder_scapula_glenoid_posteroAnteriorLine_target_z: Glenoid.posteroAnteriorLine.target gives the z coordinates of glenoid's postero-anterior line target

* shoulder_scapula_glenoid_inferoSuperiorLine_origin: Glenoid.inferoSuperiorLine.origin indicates the presence of glenoid's infero-superior line origin coordinates (1/0 tells presence/absence). Glenoid's infero-superior origin is the most inferior point of the glenoid's rim points

* shoulder_scapula_glenoid_inferoSuperiorLine_origin_x: Glenoid.inferoSuperiorLine.origin gives the x coordinates of glenoid's infero-superior line origin

* shoulder_scapula_glenoid_inferoSuperiorLine_origin_y: Glenoid.inferoSuperiorLine.origin gives the y coordinates of glenoid's infero-superior line origin

* shoulder_scapula_glenoid_inferoSuperiorLine_origin_z: Glenoid.inferoSuperiorLine.origin gives the z coordinates of glenoid's infero-superior line origin

* shoulder_scapula_glenoid_inferoSuperiorLine_target: Glenoid.inferoSuperiorLine.target indicates the presence of glenoid's infero-superior line target coordinates (1/0 tells presence/absence). Glenoid's infero-superior target is the most superior point of the glenoid's rim points

* shoulder_scapula_glenoid_inferoSuperiorLine_target_x: Glenoid.inferoSuperiorLine.target gives the x coordinates of glenoid's infero-superior line target

* shoulder_scapula_glenoid_inferoSuperiorLine_target_y: Glenoid.inferoSuperiorLine.target gives the y coordinates of glenoid's infero-superior line target

* shoulder_scapula_glenoid_inferoSuperiorLine_target_z: Glenoid.inferoSuperiorLine.target gives the z coordinates of glenoid's infero-superior line target

* shoulder_scapula_glenoid_depth: Glenoid.depth is the span of glenoid's surface points projections on glenoid's center line

* shoulder_scapula_glenoid_width: Glenoid.width is the span of glenoid's surface points projected on the most antero-posterior Principal Component Analysis axis of glenoid's surface points

* shoulder_scapula_glenoid_height: Glenoid.height is the span of glenoid's surface points projected on the most infero-superior Principal Component Analysis axis of glenoid's surface points

* shoulder_scapula_glenoid_antero_superior_angle: Glenoid.anteroSuperiorAngle is the angle between the scapula infero-superior axis and the most infero-superior Principal Component Analysis axis of glenoid's surface points

* shoulder_scapula_glenoid_version_amplitude: Glenoid.versionAmplitude is the angle between the glenoid's center line and the scapular medio-lateral axis

* shoulder_scapula_glenoid_version_orientation: Glenoid.versionOrientation is the angle between the scapular antero-posterior axis and the glenoid's center line projected on the scapular sagittal plane (superior is positive)

* shoulder_scapula_glenoid_version: Glenoid.version is the angle between the scapular medio-lateral axis and the glenoid's center line projected on the scapular axial plane (anterior is positive)

* shoulder_scapula_glenoid_inclination: Glenoid.inclination is the angle between the scapular medio-lateral axis and the glenoid's center line projected on the scapular fontal plane (superior is positive)

* shoulder_scapula_glenoid_retroversion: Glenoid.retroversion is the angle between the glenoid postero-anterior line and the line perpendicular to the Friedman's line that goes through the most anterior glenoid rim point

* shoulder_scapula_glenoid_rim_inclination: Glenoid.rimInclination is the angle between the glenoid infero-superior line and the line that goes through Trigonum Spinae and Spino-Glenoid Notch

* shoulder_scapula_glenoid_beta: Glenoid.beta is the angle between the line fitted to the scapular groove points (supraspinatus fossa landmarks) and the glenoid's supero-inferior line

* shoulder_scapula_glenoid_density_1: Glenoid.density[0] is the glenoid bone density in the first volume of interest

* shoulder_scapula_glenoid_density_2: Glenoid.density[1] is the glenoid bone density in the second volume of interest

* shoulder_scapula_glenoid_density_3: Glenoid.density[2] is the glenoid bone density in the third volume of interest

* shoulder_scapula_glenoid_density_4: Glenoid.density[3] is the glenoid bone density in the fourth volume of interest

* shoulder_scapula_glenoid_density_5: Glenoid.density[4] is the glenoid bone density in the fifth volume of interest

* shoulder_scapula_glenoid_density_6: Glenoid.density[5] is the glenoid bone density in the sixth volume of interest

* shoulder_scapula_glenoid_walch: Glenoid.walch is the Walch index read in lbovenus/shoulder/data(Dev)/Excel/ShoulderDataBase.xlsx

* shoulder_scapula_glenoid_fittedSphere_center: Glenoid.fittedSphere.center indicates the presence of glenoid's fitted sphere center coordinates (1/0 tells presence/absence). Glenoid's fitted sphere center is the center of the sphere fitted on the glenoid's surface points

* shoulder_scapula_glenoid_fittedSphere_center_x: Glenoid.fittedSphere.center gives the x coordinates of glenoid's fitted sphere center

* shoulder_scapula_glenoid_fittedSphere_center_y: Glenoid.fittedSphere.center gives the y coordinates of glenoid's fitted sphere center

* shoulder_scapula_glenoid_fittedSphere_center_z: Glenoid.fittedSphere.center gives the z coordinates of glenoid's fitted sphere center

* shoulder_scapula_glenoid_fittedSphere_radius: Glenoid.fittedSphere.radius is the radius of the sphere fitted on the glenoid's surface points (Glenoid.radius DUPLICATE)

* shoulder_scapula_glenoid_fittedSphere_r2: Glenoid.fittedSphere.R2 is the coefficient of determination of the fitted points to the computed sphere

* shoulder_scapula_glenoid_fittedSphere_rmse: Glenoid.fittedSphere.RMSE is the Root Mean Square Error of the fitted points to the computed sphere

* shoulder_scapula_acromion_ai: Acromion.AI is the Acromion Index (doi:10.2106/JBJS.D.03042)

* shoulder_scapula_acromion_csa: Acromion.CSA is the Critical Shoulder Angle (doi:10.1302/0301-620X.95B7.31028)

* shoulder_scapula_acromion_psa: Acromion.PSA is the Posterior Slope Angle (10.1016/S1058-2746(05)80036-9)

* shoulder_scapula_acromion_psl: Acromion.PSL is the length of segment between AA and AC

* shoulder_scapula_acromion_aaa: Acromion.AAA is the angle between the scapular antero-posterior axis and the vector that starts at the scapular origin and ends at the Angulus Acromialis landmark

* shoulder_scapula_acromion_aal: Acromion.AAL is the norm of the vector that starts at the scapular origin and ends at the Angulus Acromialis landmark

* shoulder_scapula_acromion_a_ax: Acromion.AAx is the x coordinates of the Angulus Acromialis landmark expressed in the scapular coordinate system (along postero-anterior axis)

* shoulder_scapula_acromion_a_ay: Acromion.AAx is the y coordinates of the Angulus Acromialis landmark expressed in the scapular coordinate system (along infero-superior axis)

* shoulder_scapula_acromion_a_az: Acromion.AAx is the z coordinates of the Angulus Acromialis landmark expressed in the scapular coordinate system (along medio-lateral axis)

* shoulder_scapula_acromion_a_cx: Acromion.ACx is the x coordinates of the Acromio-Clavicular landmark expressed in the scapular coordinate system (along postero-anterior axis)

* shoulder_scapula_acromion_a_cy: Acromion.ACx is the y coordinates of the Acromio-Clavicular landmark expressed in the scapular coordinate system (along infero-superior axis)

* shoulder_scapula_acromion_a_cz: Acromion.ACx is the z coordinates of the Acromio-Clavicular landmark expressed in the scapular coordinate system (along medio-lateral axis)

* shoulder_scapula_acromion_comment: Acromion.comment DEADVAR

* shoulder_scapula_comment: Scapula.comment DEADVAR

* shoulder_rotatorCuff_SC_name: RotatorCuff.SC.name is the Muscle.name property of the RotatorCuff.SC object and is thus "SC" (DUPLICATE with variable name)

* shoulder_rotatorCuff_SC_segmentation_name: Muscle.segmentationName is the base name used to find segmentation related files

* shoulder_rotatorCuff_SC_slice_name: Muscle.sliceName is the base name used to find slice related files

* shoulder_rotatorCuff_SC_pcsa: Muscle.PCSA is the Physiological Cross Section Area of the muscle's segmentation (in cm^2). It's the area of the healthy muscle

* shoulder_rotatorCuff_SC_atrophy: Muscle.atrophy is the ratio of atrophied muscle area over healthy muscle area

* shoulder_rotatorCuff_SC_fat: Muscle.fat is the ratio of muscle area with fat infiltration over healthy muscle area

* shoulder_rotatorCuff_SC_osteochondroma: Muscle.osteochondroma is the ratio muscle area with osteochondroma over healthy muscle area

* shoulder_rotatorCuff_SC_degeneration: Muscle.degeneration is the sum of atrophy, fat, and osteochondroma ratios

* shoulder_rotatorCuff_SC_centroid: Muscle.centroid indicates the presence of muscle's segmentation centroid coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_SC_centroid_x: Muscle.centroid gives the x coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_SC_centroid_y: Muscle.centroid gives the y coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_SC_centroid_z: Muscle.centroid gives the z coordinates of muscle's segmentation centroid

* shoulder_rotatorCuff_SC_insertion: Muscle.insertion indicates the presence of muscle's insertion coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_SC_insertion_x: Muscle.insertion gives the x coordinates of muscle's insertion 

* shoulder_rotatorCuff_SC_insertion_y: Muscle.insertion gives the y coordinates of muscle's insertion 

* shoulder_rotatorCuff_SC_insertion_z: Muscle.insertion gives the z coordinates of muscle's insertion

* shoulder_rotatorCuff_SC_force_application_point: Muscle.forceApplicationPoint indicates the presence of muscle's force application point coordinates (1/0 tells presence/absence). Muscle's force application point is the contact point of the muscle's fibre on the humeral head sphere

* shoulder_rotatorCuff_SC_force_application_point_x: Muscle.forceApplicationPoint gives the x coordinates of muscle's force application point 

* shoulder_rotatorCuff_SC_force_application_point_y: Muscle.forceApplicationPoint gives the y coordinates of muscle's force application point 

* shoulder_rotatorCuff_SC_force_application_point_z: Muscle.forceApplicationPoint gives the z coordinates of muscle's force application point

* shoulder_rotatorCuff_SC_forceVector_origin: Muscle.forceVector indicates the presence of muscle's force vector origin coordinates (1/0 tells presence/absence). Muscle's force vector origin is muscle's force application point DUPLICATE

* shoulder_rotatorCuff_SC_forceVector_origin_x: Muscle.forceVector gives the x coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SC_forceVector_origin_y: Muscle.forceVector gives the y coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SC_forceVector_origin_z: Muscle.forceVector gives the z coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SC_forceVector_target: Muscle.forceVector indicates the presence of muscle's force vector target coordinates (1/0 tells presence/absence). Muscle's force vector target is towards muscle's centroid. The force vector norm is esqual to the muscle's PCSA multiplied by (1- muscle's degeneration) 

* shoulder_rotatorCuff_SC_forceVector_target_x: Muscle.forceVector gives the x coordinates of muscle's force vector target

* shoulder_rotatorCuff_SC_forceVector_target_y: Muscle.forceVector gives the y coordinates of muscle's force vector target

* shoulder_rotatorCuff_SC_forceVector_target_z: Muscle.forceVector gives the z coordinates of muscle's force vector target

* shoulder_rotatorCuff_SS_name: RotatorCuff.SS.name is the Muscle.name property of the RotatorCuff.SS object and is thus "SS" (DUPLICATE with variable name)

* shoulder_rotatorCuff_SS_segmentation_name: Muscle.segmentationName is the base name used to find segmentation related files

* shoulder_rotatorCuff_SS_slice_name: Muscle.sliceName is the base name used to find slice related files

* shoulder_rotatorCuff_SS_pcsa: Muscle.PCSA is the Physiological Cross Section Area of the muscle's segmentation (in cm^2). It's the area of the healthy muscle

* shoulder_rotatorCuff_SS_atrophy: Muscle.atrophy is the ratio of atrophied muscle area over healthy muscle area

* shoulder_rotatorCuff_SS_fat: Muscle.fat is the ratio of muscle area with fat infiltration over healthy muscle area

* shoulder_rotatorCuff_SS_osteochondroma: Muscle.osteochondroma is the ratio muscle area with osteochondroma over healthy muscle area

* shoulder_rotatorCuff_SS_degeneration: Muscle.degeneration is the sum of atrophy, fat, and osteochondroma ratios

* shoulder_rotatorCuff_SS_centroid: Muscle.centroid indicates the presence of muscle's segmentation centroid coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_SS_centroid_x: Muscle.centroid gives the x coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_SS_centroid_y: Muscle.centroid gives the y coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_SS_centroid_z: Muscle.centroid gives the z coordinates of muscle's segmentation centroid

* shoulder_rotatorCuff_SS_insertion: Muscle.insertion indicates the presence of muscle's insertion coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_SS_insertion_x: Muscle.insertion gives the x coordinates of muscle's insertion 

* shoulder_rotatorCuff_SS_insertion_y: Muscle.insertion gives the y coordinates of muscle's insertion 

* shoulder_rotatorCuff_SS_insertion_z: Muscle.insertion gives the z coordinates of muscle's insertion

* shoulder_rotatorCuff_SS_force_application_point: Muscle.forceApplicationPoint indicates the presence of muscle's force application point coordinates (1/0 tells presence/absence). Muscle's force application point is the contact point of the muscle's fibre on the humeral head sphere

* shoulder_rotatorCuff_SS_force_application_point_x: Muscle.forceApplicationPoint gives the x coordinates of muscle's force application point 

* shoulder_rotatorCuff_SS_force_application_point_y: Muscle.forceApplicationPoint gives the y coordinates of muscle's force application point 

* shoulder_rotatorCuff_SS_force_application_point_z: Muscle.forceApplicationPoint gives the z coordinates of muscle's force application point

* shoulder_rotatorCuff_SS_forceVector_origin: Muscle.forceVector indicates the presence of muscle's force vector origin coordinates (1/0 tells presence/absence). Muscle's force vector origin is muscle's force application point DUPLICATE

* shoulder_rotatorCuff_SS_forceVector_origin_x: Muscle.forceVector gives the x coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SS_forceVector_origin_y: Muscle.forceVector gives the y coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SS_forceVector_origin_z: Muscle.forceVector gives the z coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_SS_forceVector_target: Muscle.forceVector indicates the presence of muscle's force vector target coordinates (1/0 tells presence/absence). Muscle's force vector target is towards muscle's centroid. The force vector norm is esqual to the muscle's PCSA multiplied by (1- muscle's degeneration) 

* shoulder_rotatorCuff_SS_forceVector_target_x: Muscle.forceVector gives the x coordinates of muscle's force vector target

* shoulder_rotatorCuff_SS_forceVector_target_y: Muscle.forceVector gives the y coordinates of muscle's force vector target

* shoulder_rotatorCuff_SS_forceVector_target_z: Muscle.forceVector gives the z coordinates of muscle's force vector target

* shoulder_rotatorCuff_IS_name: RotatorCuff.IS.name is the Muscle.name property of the RotatorCuff.IS object and is thus "IS" (DUPLICATE with variable name)

* shoulder_rotatorCuff_IS_segmentation_name: Muscle.segmentationName is the base name used to find segmentation related files

* shoulder_rotatorCuff_IS_slice_name: Muscle.sliceName is the base name used to find slice related files

* shoulder_rotatorCuff_IS_pcsa: Muscle.PCSA is the Physiological Cross Section Area of the muscle's segmentation (in cm^2). It's the area of the healthy muscle

* shoulder_rotatorCuff_IS_atrophy: Muscle.atrophy is the ratio of atrophied muscle area over healthy muscle area

* shoulder_rotatorCuff_IS_fat: Muscle.fat is the ratio of muscle area with fat infiltration over healthy muscle area

* shoulder_rotatorCuff_IS_osteochondroma: Muscle.osteochondroma is the ratio muscle area with osteochondroma over healthy muscle area

* shoulder_rotatorCuff_IS_degeneration: Muscle.degeneration is the sum of atrophy, fat, and osteochondroma ratios

* shoulder_rotatorCuff_IS_centroid: Muscle.centroid indicates the presence of muscle's segmentation centroid coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_IS_centroid_x: Muscle.centroid gives the x coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_IS_centroid_y: Muscle.centroid gives the y coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_IS_centroid_z: Muscle.centroid gives the z coordinates of muscle's segmentation centroid

* shoulder_rotatorCuff_IS_insertion: Muscle.insertion indicates the presence of muscle's insertion coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_IS_insertion_x: Muscle.insertion gives the x coordinates of muscle's insertion 

* shoulder_rotatorCuff_IS_insertion_y: Muscle.insertion gives the y coordinates of muscle's insertion 

* shoulder_rotatorCuff_IS_insertion_z: Muscle.insertion gives the z coordinates of muscle's insertion

* shoulder_rotatorCuff_IS_force_application_point: Muscle.forceApplicationPoint indicates the presence of muscle's force application point coordinates (1/0 tells presence/absence). Muscle's force application point is the contact point of the muscle's fibre on the humeral head sphere

* shoulder_rotatorCuff_IS_force_application_point_x: Muscle.forceApplicationPoint gives the x coordinates of muscle's force application point 

* shoulder_rotatorCuff_IS_force_application_point_y: Muscle.forceApplicationPoint gives the y coordinates of muscle's force application point 

* shoulder_rotatorCuff_IS_force_application_point_z: Muscle.forceApplicationPoint gives the z coordinates of muscle's force application point

* shoulder_rotatorCuff_IS_forceVector_origin: Muscle.forceVector indicates the presence of muscle's force vector origin coordinates (1/0 tells presence/absence). Muscle's force vector origin is muscle's force application point DUPLICATE

* shoulder_rotatorCuff_IS_forceVector_origin_x: Muscle.forceVector gives the x coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_IS_forceVector_origin_y: Muscle.forceVector gives the y coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_IS_forceVector_origin_z: Muscle.forceVector gives the z coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_IS_forceVector_target: Muscle.forceVector indicates the presence of muscle's force vector target coordinates (1/0 tells presence/absence). Muscle's force vector target is towards muscle's centroid. The force vector norm is esqual to the muscle's PCSA multiplied by (1- muscle's degeneration) 

* shoulder_rotatorCuff_IS_forceVector_target_x: Muscle.forceVector gives the x coordinates of muscle's force vector target

* shoulder_rotatorCuff_IS_forceVector_target_y: Muscle.forceVector gives the y coordinates of muscle's force vector target

* shoulder_rotatorCuff_IS_forceVector_target_z: Muscle.forceVector gives the z coordinates of muscle's force vector target

* shoulder_rotatorCuff_TM_name: RotatorCuff.TM.name is the Muscle.name property of the RotatorCuff.TM object and is thus "TM" (DUPLICATE with variable name)

* shoulder_rotatorCuff_TM_segmentation_name: Muscle.segmentationName is the base name used to find segmentation related files

* shoulder_rotatorCuff_TM_slice_name: Muscle.sliceName is the base name used to find slice related files

* shoulder_rotatorCuff_TM_pcsa: Muscle.PCSA is the Physiological Cross Section Area of the muscle's segmentation (in cm^2). It's the area of the healthy muscle

* shoulder_rotatorCuff_TM_atrophy: Muscle.atrophy is the ratio of atrophied muscle area over healthy muscle area

* shoulder_rotatorCuff_TM_fat: Muscle.fat is the ratio of muscle area with fat infiltration over healthy muscle area

* shoulder_rotatorCuff_TM_osteochondroma: Muscle.osteochondroma is the ratio muscle area with osteochondroma over healthy muscle area

* shoulder_rotatorCuff_TM_degeneration: Muscle.degeneration is the sum of atrophy, fat, and osteochondroma ratios

* shoulder_rotatorCuff_TM_centroid: Muscle.centroid indicates the presence of muscle's segmentation centroid coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_TM_centroid_x: Muscle.centroid gives the x coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_TM_centroid_y: Muscle.centroid gives the y coordinates of muscle's segmentation centroid 

* shoulder_rotatorCuff_TM_centroid_z: Muscle.centroid gives the z coordinates of muscle's segmentation centroid

* shoulder_rotatorCuff_TM_insertion: Muscle.insertion indicates the presence of muscle's insertion coordinates (1/0 tells presence/absence)

* shoulder_rotatorCuff_TM_insertion_x: Muscle.insertion gives the x coordinates of muscle's insertion 

* shoulder_rotatorCuff_TM_insertion_y: Muscle.insertion gives the y coordinates of muscle's insertion 

* shoulder_rotatorCuff_TM_insertion_z: Muscle.insertion gives the z coordinates of muscle's insertion

* shoulder_rotatorCuff_TM_force_application_point: Muscle.forceApplicationPoint indicates the presence of muscle's force application point coordinates (1/0 tells presence/absence). Muscle's force application point is the contact point of the muscle's fibre on the humeral head sphere

* shoulder_rotatorCuff_TM_force_application_point_x: Muscle.forceApplicationPoint gives the x coordinates of muscle's force application point 

* shoulder_rotatorCuff_TM_force_application_point_y: Muscle.forceApplicationPoint gives the y coordinates of muscle's force application point 

* shoulder_rotatorCuff_TM_force_application_point_z: Muscle.forceApplicationPoint gives the z coordinates of muscle's force application point

* shoulder_rotatorCuff_TM_forceVector_origin: Muscle.forceVector indicates the presence of muscle's force vector origin coordinates (1/0 tells presence/absence). Muscle's force vector origin is muscle's force application point DUPLICATE

* shoulder_rotatorCuff_TM_forceVector_origin_x: Muscle.forceVector gives the x coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_TM_forceVector_origin_y: Muscle.forceVector gives the y coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_TM_forceVector_origin_z: Muscle.forceVector gives the z coordinates of muscle's force vector origin DUPLICATE

* shoulder_rotatorCuff_TM_forceVector_target: Muscle.forceVector indicates the presence of muscle's force vector target coordinates (1/0 tells presence/absence). Muscle's force vector target is towards muscle's centroid. The force vector norm is esqual to the muscle's PCSA multiplied by (1- muscle's degeneration) 

* shoulder_rotatorCuff_TM_forceVector_target_x: Muscle.forceVector gives the x coordinates of muscle's force vector target

* shoulder_rotatorCuff_TM_forceVector_target_y: Muscle.forceVector gives the y coordinates of muscle's force vector target

* shoulder_rotatorCuff_TM_forceVector_target_z: Muscle.forceVector gives the z coordinates of muscle's force vector target

* shoulder_rotatorCuff_imbalance_angle3_d: RotatorCuff.imbalanceAngle3D is the angle between the scapular medio-lateral axis and the force resultant of all the rotator cuff muscles' forces

* shoulder_rotatorCuff_imbalance_angle_orientation: RotatorCuff.imbalanceAngleOrientation is the angle between the scapular antero-posterior axis and the force resultant of the rotator cuff muscles projected on the scapular sagittal plane (superior is positive)

* shoulder_rotatorCuff_imbalance_angle_antero_posterior: RotatorCuff.imbalanceAngleAnteroPosterior is the angle between the scapular lateral-medial axis and the force resultant of the rotator cuff muscles projected on the scapular axial plane (posterior is positive)

* shoulder_rotatorCuff_imbalance_angle_infero_superior: RotatorCuff.imbalanceAngleAnteroPosterior is the angle between the scapular lateral-medial axis and the force resultant of the rotator cuff muscles projected on the scapular frontal plane (superior is positive)

* shoulder_humerus_insertionsRing_center: Humerus.insertionsRing.center indicates the presence of the coordinates of the center of the rotator cuff insertions ring on humerus (1/0 tells presence/absence). Rotator cuff insertions ring on humerus is measured with to insertions landmarks 

* shoulder_humerus_insertionsRing_center_x: Humerus.insertionsRing.center gives the x coordinates of the center of the rotator cuff insertions ring on humerus

* shoulder_humerus_insertionsRing_center_y: Humerus.insertionsRing.center gives the y coordinates of the center of the rotator cuff insertions ring on humerus

* shoulder_humerus_insertionsRing_center_z: Humerus.insertionsRing.center gives the z coordinates of the center of the rotator cuff insertions ring on humerus

* shoulder_humerus_insertionsRing_radius: Humerus.insertionsRing.radius is the radius of the rotator cuff insertions ring on humerus

* shoulder_humerus_insertionsRing_normal_origin: Humerus.insertionsRing.normal indicates the presence of the origin of the normal vector of the rotator cuff insertions plane (1/0 tells presence/absence). Rotator cuff insertions plane contains rotator cuff insertions ring

* shoulder_humerus_insertionsRing_normal_origin_x: Humerus.insertionsRing.normal gives the x coordinates of the origin of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_insertionsRing_normal_origin_y: Humerus.insertionsRing.normal gives the y coordinates of the origin of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_insertionsRing_normal_origin_z: Humerus.insertionsRing.normal gives the z coordinates of the origin of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_insertionsRing_normal_target: Humerus.insertionsRing.normal indicates the presence of the target of the normal vector of the rotator cuff insertions plane (1/0 tells presence/absence). Rotator cuff insertions plane contains rotator cuff insertions ring

* shoulder_humerus_insertionsRing_normal_target_x: Humerus.insertionsRing.normal gives the x coordinates of the target of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_insertionsRing_normal_target_y: Humerus.insertionsRing.normal gives the y coordinates of the target of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_insertionsRing_normal_target_z: Humerus.insertionsRing.normal gives the z coordinates of the target of the normal vector of the rotator cuff insertions plane

* shoulder_humerus_center: Humerus.center indicates the presence of the coordinates of the center of the humeral head (1/0 tells presence(absence). The humeral head is a sphere fitted to humeral head landmarks

* shoulder_humerus_center_x: Humerus.center gives the x coordinates of the center of the humeral head

* shoulder_humerus_center_y: Humerus.center gives the y coordinates of the center of the humeral head

* shoulder_humerus_center_z: Humerus.center gives the z coordinates of the center of the humeral head

* shoulder_humerus_radius: Humerus.radius is the radius of the humeral head. The humeral head is a sphere fitted to humeral head landmarks

* shoulder_humerus_shs_angle: Humerus.SHSAngle is the scapulo-humeral angle. It's the angle between the scapular medio-lateral axis and the vector that starts at glenoid center and ends at humerus center

* shoulder_humerus_shspa: Humerus.SHSPA is the angle between the scapular medio-lateral axis and projection on the scapular axial plane of the vector that starts at glenoid center and ends at humerus center (anterior is positive)

* shoulder_humerus_shsis: Humerus.SHSIS is the angle between the scapular medio-lateral axis and projection on the scapular frontal plane of the vector that starts at glenoid center and ends at humerus center (superior is positive)

* shoulder_humerus_shs_ampl: Humerus.SHSAmpl is the ratio of the norm of the projection on the scapular sagittal plane of the vector that starts at glenoid center and ends at humerus center over the humerus radius

* shoulder_humerus_shs_orient: Humerus.SHSOrient is the angle between the scapular antero-posterior axis and the projection on the scapular sagittal plane of the vector that starts at glenoid center and ends at humerus center

* shoulder_humerus_ghs_ampl: Humerus.GHSAmpl is the ratio of the norm of the vector from glenoid sphere center to glenoid center line (and perpendicular to it) over the humerus radius

* shoulder_humerus_ghs_orient: Humerus.GHSOrient is the angle between the scapular antero-posterior axis and the projection on the scapular sagittal plane of the vector from glenoid sphere center to glenoid center line (and perpendicular to it)

* shoulder_humerus_subluxation_index3_d: Humerus.subluxationIndex3D is the ratio of the sum of the norm of the projection on the glenoid postero-anterior line of the vector that starts at glenoid center and ends at humerus center plus the humerus radius over two times the humerus radius

## Documentation

Should be added here:
* datastructure design
* class diagram

## Files/Folders

### Folders

- **./anatomy** contains scripts used by Amira manual measurements and what looks like deprecated scripts that are now part of the Scapula class.
- **./ShoulderCase** contains the class and scripts used to update and access to the data. See here [how are organized class files and folders](https://ch.mathworks.com/help/matlab/matlab_oop/class-files-and-folders.html).
- **./ImportSCase** contains the GUI used to import new data into the database.
- **./Generated_Amira_TCL_Scripts** used to store TCL scripts. Not tracked by git.
- **./upsert** contains a script to update and insert data in a database. Looks like it's an external not LBO-made feature.

### Files

- *config.json* is mandatory.

## Version

Current version is 2.1.0

## Contributors

* Alexandre Terrier, 2012-now
* Julien Ston, 2012-2014
* Antoine Dewarrat, 2014-2015
* Raphael Obrist, 2015-2017
* Killian Cosendey, 2017
* Valérie Malfroy Camine, 2017
* Jorge Solana-Muñoz, 2018-2019
* Bharath Narayanan, 2018
* Paul Cimadomo, 2018
* Lore Hoffmann, 2019
* Nathan Donini, 2019
* Benoit Delecourt, 2019
* Amirsiavosh Bashardoust, 2020
* Matthieu Boubat, 2019-2021
* Pezhman Eghbalishamsabadi, 2021-now


## License

EPFL CC BY-NC-SA 4.0 Int.
