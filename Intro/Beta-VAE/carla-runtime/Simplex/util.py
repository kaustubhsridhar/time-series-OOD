#!/usr/bin/env python2

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
from collections import deque

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        self.transform = (self.world.get_map().get_spawn_points())
        spawn = self.transform[1]
        self.vehicle = self.world.spawn_actor(self.model_3, spawn)
        self.actor_list.append(self.vehicle)

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.10
        self.world.apply_settings(settings)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", '640')
        self.rgb_cam.set_attribute("image_size_y", '480')
        self.rgb_cam.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=10.5,y=-1.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # colsensor = self.blueprint_library.find("sensor.other.collision")
        # self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        # self.actor_list.append(self.colsensor)
        # self.colsensor.listen(lambda event: self.collision_data(event))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera

    # def collision_data(self, event):
    #     self.collision_hist.append(event)

    def Weather_Condition(self, c,p,s):
        weather = carla.WeatherParameters(
            cloudyness=c,
            precipitation=p,
            sun_altitude_angle=s)
        self.world.set_weather(weather)


    def process_img(self, image):
        i = np.array(image.raw_data)

        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3

    def step(self, action):
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.25, steer=action))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        return self.front_camera
