
from scipy import interpolate
import numpy as np
import pandas as pd

import opensim as osm


class OpenSimModelWrapper:

    def __init__(self, modelPath, motionPath):
        osm.Logger.setLevelString('error')
        self.model = osm.Model(str(modelPath))
        self.model.initSystem()

        # Create time-series table with coordinate values.
        self.table = osm.TimeSeriesTable(str(motionPath))
        tableProcessor = osm.TableProcessor(self.table)
        self.columnLabels = list(self.table.getColumnLabels())
        tableProcessor.append(osm.TabOpUseAbsoluteStateNames())
        self.time = np.asarray(self.table.getIndependentColumn())

        # Convert in radians.
        self.table = tableProcessor.processAndConvertToRadians(self.model)

        # Compute coordinate speeds and accelerations and add speeds to table.
        self.Qs = self.table.getMatrix().to_numpy()
        self.Qds = np.zeros(self.Qs.shape)
        self.Qdds = np.zeros(self.Qs.shape)
        columnAbsoluteLabels = list(self.table.getColumnLabels())
        for i, columnLabel in enumerate(columnAbsoluteLabels):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, self.Qs[:,i], k=3)
            # Coordinate speeds
            splineD1 = spline.derivative(n=1)
            self.Qds[:,i] = splineD1(self.time)
            # Coordinate accelerations.
            splineD2 = spline.derivative(n=2)
            self.Qdds[:,i] = splineD2(self.time)
            # Add coordinate speeds to table.
            columnLabel_speed = columnLabel[:-5] + 'speed'
            self.table.appendColumn(
                columnLabel_speed,
                osm.Vector(self.Qds[:,i].flatten().tolist()))

        # Append missing muscle states to table.
        # Needed for StatesTrajectory.
        stateVariableNames = self.model.getStateVariableNames()
        stateVariableNamesStr = [
            stateVariableNames.get(i) for i in range(
                stateVariableNames.getSize())]
        existingLabels = self.table.getColumnLabels()
        for stateVariableNameStr in stateVariableNamesStr:
            if not stateVariableNameStr in existingLabels:
                vec_0 = osm.Vector([0] * self.table.getNumRows())
                self.table.appendColumn(stateVariableNameStr, vec_0)

        # Set state trajectory
        self.stateTrajectory = osm.StatesTrajectory.createFromStatesTable(
            self.model, self.table)

        # Number of muscles.
        self.nMuscles = 0
        self.forceSet = self.model.getForceSet()
        for i in range(self.forceSet.getSize()):
            c_force_elt = self.forceSet.get(i)
            if 'Muscle' in c_force_elt.getConcreteClassName():
                self.nMuscles += 1

        # Coordinates.
        self.coordinateSet = self.model.getCoordinateSet()
        self.nCoordinates = self.coordinateSet.getSize()
        self.coordinates = [self.coordinateSet.get(i).getName()
                            for i in range(self.nCoordinates)]

        # TODO: hard coded
        # Translational coordinates.
        columnTrLabels = [
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']
        self.idxColumnTrLabels = [
            self.columnLabels.index(i) for i in columnTrLabels]
        self.idxColumnRotLabels = [
            self.columnLabels.index(i) for i in self.columnLabels
            if not i in columnTrLabels]

        # TODO: hard coded
        self.rootCoordinates = [
            'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
            'pelvis_tx', 'pelvis_ty', 'pelvis_tz']

        self.lumbarCoordinates = ['lumbar_extension', 'lumbar_bending',
                                  'lumbar_rotation']

        self.armCoordinates = ['arm_flex_r', 'arm_add_r', 'arm_rot_r',
                               'elbow_flex_r', 'pro_sup_r',
                               'arm_flex_l', 'arm_add_l', 'arm_rot_l',
                               'elbow_flex_l', 'pro_sup_l']

    def get_coordinate_values(self, in_degrees=True,
                              lowpass_cutoff_frequency=-1):

        # Convert to degrees.
        if in_degrees:
            Qs = np.zeros((self.Qs.shape))
            Qs[:, self.idxColumnTrLabels] = self.Qs[:, self.idxColumnTrLabels]
            Qs[:, self.idxColumnRotLabels] = (
                self.Qs[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qs = self.Qs

        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qs = lowPassFilter(self.time, Qs, lowpass_cutoff_frequency)
            if self.lowpass_cutoff_frequency_for_coordinate_values > 0:
                print("Warning: You are filtering the coordinate values a second time; coordinate values were filtered when creating your class object.")

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qs), axis=1)
        columns = ['time'] + self.columnLabels
        coordinate_values = pd.DataFrame(data=data, columns=columns)

        return coordinate_values

    def get_coordinate_speeds(self, in_degrees=True,
                              lowpass_cutoff_frequency=-1):

        # Convert to degrees.
        if in_degrees:
            Qds = np.zeros((self.Qds.shape))
            Qds[:, self.idxColumnTrLabels] = (
                self.Qds[:, self.idxColumnTrLabels])
            Qds[:, self.idxColumnRotLabels] = (
                self.Qds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qds = self.Qds

        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qds = lowPassFilter(self.time, Qds, lowpass_cutoff_frequency)

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qds), axis=1)
        columns = ['time'] + self.columnLabels
        coordinate_speeds = pd.DataFrame(data=data, columns=columns)

        return coordinate_speeds

    def get_coordinate_accelerations(self, in_degrees=True,
                                     lowpass_cutoff_frequency=-1):

        # Convert to degrees.
        if in_degrees:
            Qdds = np.zeros((self.Qdds.shape))
            Qdds[:, self.idxColumnTrLabels] = (
                self.Qdds[:, self.idxColumnTrLabels])
            Qdds[:, self.idxColumnRotLabels] = (
                self.Qdds[:, self.idxColumnRotLabels] * 180 / np.pi)
        else:
            Qdds = self.Qdds

        # Filter.
        if lowpass_cutoff_frequency > 0:
            Qdds = lowPassFilter(self.time, Qdds, lowpass_cutoff_frequency)

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), Qdds), axis=1)
        columns = ['time'] + self.columnLabels
        coordinate_accelerations = pd.DataFrame(data=data, columns=columns)

        return coordinate_accelerations

    def get_muscle_tendon_lengths(self, lowpass_cutoff_frequency=-1):

        # Compute muscle-tendon lengths.
        lMT = np.zeros((self.table.getNumRows(), self.nMuscles))
        for i in range(self.table.getNumRows()):
            self.model.realizePosition(self.stateTrajectory[i])
            if i == 0:
                muscleNames = []
            for m in range(self.forceSet.getSize()):
                c_force_elt = self.forceSet.get(m)
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    cObj = osm.Muscle.safeDownCast(c_force_elt)
                    lMT[i,m] = cObj.getLength(self.stateTrajectory[i])
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())

        # Filter.
        if lowpass_cutoff_frequency > 0:
            lMT = lowPassFilter(self.time, lMT, lowpass_cutoff_frequency)

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), lMT), axis=1)
        columns = ['time'] + muscleNames
        muscle_tendon_lengths = pd.DataFrame(data=data, columns=columns)

        return muscle_tendon_lengths

    def get_moment_arms(self, lowpass_cutoff_frequency=-1):

        # Compute moment arms.
        dM =  np.zeros((self.table.getNumRows(), self.nMuscles,
                        self.nCoordinates))
        for i in range(self.table.getNumRows()):
            self.model.realizePosition(self.stateTrajectory[i])
            if i == 0:
                muscleNames = []
            for m in range(self.forceSet.getSize()):
                c_force_elt = self.forceSet.get(m)
                if 'Muscle' in c_force_elt.getConcreteClassName():
                    muscleName = c_force_elt.getName()
                    cObj = osm.Muscle.safeDownCast(c_force_elt)
                    if i == 0:
                        muscleNames.append(c_force_elt.getName())
                    for c, coord in enumerate(self.coordinates):
                        # We use prior knowledge to improve computation speed;
                        # We do not want to compute moment arms that are not
                        # relevant, eg for a muscle of the left side with
                        # respect to a coordinate of the right side.
                        if muscleName[-2:] == '_l' and coord[-2:] == '_r':
                            dM[i, m, c] = 0
                        elif muscleName[-2:] == '_r' and coord[-2:] == '_l':
                            dM[i, m, c] = 0
                        elif (coord in self.rootCoordinates or
                              coord in self.lumbarCoordinates or
                              coord in self.armCoordinates):
                            dM[i, m, c] = 0
                        else:
                            coordinate = self.coordinateSet.get(
                                self.coordinates.index(coord))
                            dM[i, m, c] = cObj.computeMomentArm(
                                self.stateTrajectory[i], coordinate)

        # Clean numerical artefacts (ie, moment arms smaller than 1e-5 m).
        dM[np.abs(dM) < 1e-5] = 0

        # Filter.
        if lowpass_cutoff_frequency > 0:
            for c, coord in enumerate(self.coordinates):
                dM[:, :, c] = lowPassFilter(self.time, dM[:, :, c],
                                            lowpass_cutoff_frequency)

        # Return as DataFrame.
        moment_arms = {}
        for c, coord in enumerate(self.coordinates):
            data = np.concatenate(
                (np.expand_dims(self.time, axis=1), dM[:,:,c]), axis=1)
            columns = ['time'] + muscleNames
            moment_arms[coord] = pd.DataFrame(data=data, columns=columns)

        return moment_arms

    def compute_center_of_mass(self):

        # Compute center of mass position and velocity.
        self.com_values = np.zeros((self.table.getNumRows(),3))
        self.com_speeds = np.zeros((self.table.getNumRows(),3))
        for i in range(self.table.getNumRows()):
            self.model.realizeVelocity(self.stateTrajectory[i])
            self.com_values[i,:] = self.model.calcMassCenterPosition(
                self.stateTrajectory[i]).to_numpy()
            self.com_speeds[i,:] = self.model.calcMassCenterVelocity(
                self.stateTrajectory[i]).to_numpy()

    def get_center_of_mass_values(self):

        self.compute_center_of_mass()
        com_v = self.com_values

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_v), axis=1)
        columns = ['time'] + ['x','y','z']
        com_values = pd.DataFrame(data=data, columns=columns)

        return com_values

    def get_center_of_mass_speeds(self):

        self.compute_center_of_mass()
        com_s = self.com_speeds

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_s), axis=1)
        columns = ['time'] + ['x','y','z']
        com_speeds = pd.DataFrame(data=data, columns=columns)

        return com_speeds

    def get_center_of_mass_accelerations(self):

        self.compute_center_of_mass()
        com_s = self.com_speeds

        # Accelerations are first time derivative of speeds.
        com_a = np.zeros((com_s.shape))
        for i in range(com_s.shape[1]):
            spline = interpolate.InterpolatedUnivariateSpline(
                self.time, com_s[:,i], k=3)
            splineD1 = spline.derivative(n=1)
            com_a[:,i] = splineD1(self.time)

        # Return as DataFrame.
        data = np.concatenate(
            (np.expand_dims(self.time, axis=1), com_a), axis=1)
        columns = ['time'] + ['x','y','z']
        com_accelerations = pd.DataFrame(data=data, columns=columns)

        return com_accelerations


