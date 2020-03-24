################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################
#python project_laser_into_camera.py --image_dir /data/lyh/RobotCar_mono_left/2014-05-19-13-20-57/mono_left --laser_dir /data/lyh/RobotCar_lms_front/2014-05-19-13-20-57/lms_front --poses_file /data/lyh/RobotCar_gps_ins/2014-05-19-13-20-57/gps/ins.csv --models_dir /data/lyh/lab/robotcar-dataset-sdk/models --extrinsics_dir /data/lyh/lab/robotcar-dataset-sdk/extrinsics --image_idx 100
#python project_laser_into_camera.py --image_dir /data/lyh/RobotCar/mono_left/2014-05-19-13-20-57/mono_left --laser_dir /data/lyh/RobotCar/lms_front/2014-05-19-13-20-57/lms_front --poses_file /data/lyh/RobotCar/gps_ins/2014-05-19-13-20-57/gps/ins.csv --models_dir /data/lyh/lab/robotcar-dataset-sdk/models --extrinsics_dir /data/lyh/lab/robotcar-dataset-sdk/extrinsics --image_idx 100
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import argparse

from build_pointcloud import build_pointcloud
from transform import build_se3_transform
from image import load_image
from camera_model import CameraModel

parser = argparse.ArgumentParser(description='Project LIDAR data into camera image')
parser.add_argument('--image_dir', type=str, help='Directory containing images')
parser.add_argument('--laser_dir', type=str, help='Directory containing LIDAR scans')
parser.add_argument('--poses_file', type=str, help='File containing either INS or VO poses')
parser.add_argument('--models_dir', type=str, help='Directory containing camera models')
parser.add_argument('--extrinsics_dir', type=str, help='Directory containing sensor extrinsics')
parser.add_argument('--image_idx', type=int, help='Index of image to display')

args = parser.parse_args()

print(args.image_dir)
print(args.laser_dir)
print(args.poses_file)
print(args.models_dir)
print(args.extrinsics_dir)
print(args.image_idx)

model = CameraModel(args.models_dir, args.image_dir)
print("-----------------------1-------------------------")


extrinsics_path = os.path.join(args.extrinsics_dir, model.camera + '.txt')
print(extrinsics_path)
with open(extrinsics_path) as extrinsics_file:
    extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

print("-----------------------2-------------------------")

G_camera_vehicle = build_se3_transform(extrinsics)
G_camera_posesource = None

print("-----------------------3-------------------------")

poses_type = re.search('(vo|ins)\.csv', args.poses_file).group(1)
if poses_type == 'ins':
    print(os.path.join(args.extrinsics_dir, 'ins.txt'))
    with open(os.path.join(args.extrinsics_dir, 'ins.txt')) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])
else:
    # VO frame and vehicle frame are the same
    G_camera_posesource = G_camera_vehicle

print("-----------------------4-------------------------")

timestamps_path = os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps')
print(os.path.join(args.image_dir, os.pardir, model.camera + '.timestamps'))
if not os.path.isfile(timestamps_path):
    timestamps_path = os.path.join(args.image_dir, os.pardir, os.pardir, model.camera + '.timestamps')

print("-----------------------5-------------------------")

timestamp = 0
with open(timestamps_path) as timestamps_file:
    for i, line in enumerate(timestamps_file):
        if i == args.image_idx:
            timestamp = int(line.split(' ')[0])

print(timestamp)
print("-----------------------6-------------------------")

pointcloud, reflectance = build_pointcloud(args.laser_dir, args.poses_file, args.extrinsics_dir,
                                           timestamp - 1e7, timestamp + 1e7, timestamp)

print(pointcloud.shape)
pointcloud = np.dot(G_camera_posesource, pointcloud)
print(pointcloud.shape)
np.savetxt("let_me_look.txt",pointcloud.T,delimiter=",",fmt = "%.3f")

print("-----------------------7-------------------------")

image_path = os.path.join(args.image_dir, str(timestamp) + '.png')
image = load_image(image_path, model)

print("-----------------------8-------------------------")

uv, depth = model.project(pointcloud, image.shape)

plt.imshow(image)
plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, edgecolors='none', cmap='jet')
plt.xlim(0, image.shape[1])
plt.ylim(image.shape[0], 0)
plt.xticks([])
plt.yticks([])
plt.show()

