
import pandas as pd
import numpy as np

from scipy.spatial import ConvexHull
from numpy.linalg import norm

from utilsKinematics import OpenSimModelWrapper


def center_of_mass(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_values()
    return df_com[['z', 'y', 'x']].values


def center_of_mass_vel(modelPath, motionPath):
    model = OpenSimModelWrapper(modelPath, motionPath)
    df_com = model.get_center_of_mass_speeds()
    return df_com[['z', 'y', 'x']].values


# from https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    # gets the angle between two vectors
    v1_u = v1 / norm(v1)
    v2_u = v2 / norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angle_between_all(s1, s2):
    # gets the angles between two vectors over time
    assert s1.shape == s2.shape
    out = np.empty(s1.shape[0])
    for i in range(s1.shape[0]):
        out[i] = angle_between(s1[i,:], s2[i,:])
    return out


def trc_arm_angles(xyz, markers):
    # shoulder, elbow, and wrist markers
    # rs = xyz[:,np.argmax(markers=='RShoulder'),:]
    # ls = xyz[:,np.argmax(markers=='LShoulder'),:]
    # re = xyz[:,np.argmax(markers=='RElbow'),:]
    # le = xyz[:,np.argmax(markers=='LElbow'),:]
    # rw = xyz[:,np.argmax(markers=='RWrist'),:]
    # lw = xyz[:,np.argmax(markers=='LWrist'),:]

    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    re = xyz[:,np.argmax(markers=='r_melbow_study'),:]
    le = xyz[:,np.argmax(markers=='L_melbow_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    # gravity vector
    grav = np.zeros_like(rs)
    grav[:,1] = -1

    # shoulder and elbow angles
    rsa = angle_between_all(re-rs, grav) * 180 / np.pi
    rea = angle_between_all(rw-re, re-rs) * 180 / np.pi
    lsa = angle_between_all(le-ls, grav) * 180 / np.pi
    lea = angle_between_all(lw-le, le-ls) * 180 / np.pi

    return rsa, rea, lsa, lea


def reachable_workspace(xyz, markers):
    rs = xyz[:,np.argmax(markers=='r_shoulder_study'),:]
    ls = xyz[:,np.argmax(markers=='L_shoulder_study'),:]
    rw = xyz[:,np.argmax(markers=='r_mwrist_study'),:]
    lw = xyz[:,np.argmax(markers=='L_mwrist_study'),:]

    rarm = rw-rs
    larm = lw-ls

    # arm length scaling
    # TODO use scaled model instead
    rarm /= np.median(norm(rarm, axis=1))
    larm /= np.median(norm(larm, axis=1))

    rw = ConvexHull(np.concatenate([rarm, larm + ls-rs])).volume

    return rw


def arm_rom_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)

    mean_sa = (rsa + lsa) / 2
    mean_ea = (rea + lea) / 2
    max_mean_sa = np.max(mean_sa)
    mean_ea_at_max_mean_sa = mean_ea[np.argmax(mean_sa)]

    min_sa = np.vstack([rsa, lsa]).min(0)
    max_min_sa = min_sa.max()

    max_ea = np.vstack([rea, lea]).max(0)
    max_ea_at_max_min_sa = max_ea[np.argmax(min_sa)]

    rw = reachable_workspace(xyz, markers)

    # TODO peak shoulder moment (SDU to fix shoulder model)

    return max_mean_sa, mean_ea_at_max_mean_sa, max_min_sa, max_ea_at_max_min_sa, rw


def arm_rom_mot_feats(df):
    # TODO max shoulder moment (SDU to fix shoulder model)

    # TODO review some videos to check sanity

    return None


def brooke_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)
    mean_sa = (rsa + lsa) / 2
    mean_ea = (rea + lea) / 2
    max_mean_sa = np.max(mean_sa)
    max_mean_ea = np.max(mean_ea)

    min_sa = np.vstack([rsa, lsa]).min(0)
    max_min_sa = min_sa.max()

    max_ea = np.vstack([rea, lea]).max(0)
    max_ea_at_max_min_sa = max_ea[np.argmax(min_sa)]

    # TODO peak shoulder moment (SDU to fix shoulder model)

    return max_mean_sa, max_mean_ea, max_min_sa, max_ea_at_max_min_sa


def brooke_mot_feats(df):
    # TODO max shoulder moment (SDU to fix shoulder model)

    return None


def curls_trc_feats(xyz, markers):
    rsa, rea, lsa, lea = trc_arm_angles(xyz, markers)
    max_rea = np.max(rea)
    max_lea = np.max(lea)
    mean_ea = (rea + lea) / 2
    max_mean_ea = np.max(mean_ea)

    # TODO wrist flexion range of motion
    # TODO wrist extension range of motion

    return max_mean_ea


def gait_trc_feats(xyz, markers, fps):
    com = xyz[:,np.argmax(markers == 'midHip'),:] # TODO uses actual CoM
    com -= com[-1,:]

    com_dist = norm(com[:,[0,2]], axis=-1)
    last_4m = np.argmax(com_dist < 4)
    time_4m = (xyz.shape[0] - last_4m)/fps
    time_10m = time_4m * 10 / 4
    speed = 4 / time_4m

    com_bob = com[last_4m:,1].ptp()

    com_xz = com[last_4m:,[0,2]].copy()
    direction = com_xz[0,:] - com_xz[-1,:]
    direction /= norm(direction)
    com_xz -= np.outer(com_xz @ direction, direction)
    com_sway = norm(com_xz, axis=1).ptp()

    # TODO scale bob and sway by height

    # TODO joint impedance? See Cavallo 2022

    return time_10m, speed, com_bob, com_sway, last_4m


def gait_mot_feats(df, last_4m):

    rha = df.hip_adduction_r.to_numpy()
    lha = df.hip_adduction_l.to_numpy()
    rka = df.knee_angle_r.to_numpy()
    lka = df.knee_angle_l.to_numpy()

    ptp_r_hip_add = rha.ptp()
    ptp_l_hip_add = lha.ptp()
    mean_ptp_hip_add = (ptp_r_hip_add + ptp_l_hip_add) / 2
    
    max_rka = rka.max()
    max_lka = lka.max()
    mean_max_ka = (max_rka + max_lka) / 2

    # TODO spatiotemporal features

    # replicate timed functional test
    # stride length (normalized by height?)
    # foot drop: max dorsiflexion angle during swing
    # hip circumduction (secondary to foot drop)
    # peak knee flexion angle during swing and stance

    # ankle plantar flexion moments - dynamic simulation
    # peak saggital moments and power at all three lower extremity joints

    return mean_ptp_hip_add, mean_max_ka


def tug_trc_feats(xyz, markers, fps):
    return None


def tug_mot_feats(df):
    # TODO replicate timing

    # TODO turning speed
    # max rate of change of pelvis_rotation

    # TODO whole-body angular momentum
    # use OpenSim BodyKinematics?

    return None


# def jump_trc_feats(xyz, markers, fps):
#     com = xyz[:,np.argmax(markers == 'midHip'),:] # TODO uses actual CoM
#     max_com_vel = np.max(np.diff(com[:,1])) * fps

#     return max_com_vel


def jump_trc_feats(com_xyz, fps):
    max_com_vel = np.max(np.diff(com_xyz[:,1])) * fps

    return max_com_vel


def toe_stand_trc_feats(xyz, markers, fps, com):
    # TODO find a participant who has low CoM y disp, see if they have x CoM disp

    start_win = int(fps*1)
    com_start = com[:start_win,:].mean(0)
    com -= com_start

    com_height = com_start[1]
    com /= com_height

    rc = xyz[:,np.argmax(markers=='r_calc_study'),:]
    lc = xyz[:,np.argmax(markers=='L_calc_study'),:]
    rc -= rc[:start_win,:].mean(0)
    lc -= lc[:start_win,:].mean(0)

    int_com_elev = np.sum(com[:,1])
    int_com_fwd = np.sum(com[:,2])
    # int_r_heel_elev = np.sum(rc[:,1])
    # int_l_heel_elev = np.sum(lc[:,1])
    int_mean_heel_elev = np.sum((rc[:,1] + lc[:,1])/2)

    # TODO normalize by height or foot length

    # or swap CoM vertical with peak knee flexion?

    # pick a steady state period
    # pick a 0.5 sec or so window where heels are high and mostly still

    # something about center of mass variance in 3D (teasing out balance)

    # kinematic trifecta
    # - com height
    # - knee flexion angle
    # - plantar flexion angle
    
    # muscle-driven simulation trifecta
    # gastroc-soleus ratio 
    # peak moment during steady state
    # integrated angle/height

    # TODO integral of plantar flexion torque

    return int_com_elev, int_com_fwd, int_mean_heel_elev


def sts_feats():
    # TODO rename function

    # replicate TFT time

    # peak sagittal moments for all three joints
    # or mean if peaks are too noisy

    # trunk lean - compute from markers, not kinematics, project onto sagittal plane
    # peak angular velocity of the above trunk lean thingy
    pass


def toe_stand_mot_feats(df):
    raa = df['ankle_angle_r'].values
    laa = df['ankle_angle_l'].values

    dt = df.time[1] - df.time[0]
    int_raa = np.sum(raa) * dt
    int_laa = np.sum(laa) * dt
    mean_int_aa = (int_raa + int_laa) / 2

    # TODO integral of plantar flexion torque

    return mean_int_aa



