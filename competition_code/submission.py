"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
import math
import os

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

def distance_p_to_p(p1: roar_py_interface.RoarPyWaypoint, p2: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(p2.location[:2] - p1.location[:2])

class SpeedData:
    def __init__(self, distance_to_section: float, current_speed: float, target_speed: float, recommended_speed: float):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed

class ThrottlePlanner:
    def __init__(self):
        self.max_radius = 10000.0
        self.max_speed = 300.0
        self.intended_target_distance = [0, 30, 60, 90, 120, 140, 170]
        self.target_distance = [0, 30, 60, 90, 120, 150, 180]
        self.close_index = 0
        self.mid_index = 1
        self.far_index = 2
        self.tick_counter = 0
        self.previous_speed = 1.0
        self.brake_ticks = 0

    def run(self, waypoints: List[roar_py_interface.RoarPyWaypoint], current_location: np.ndarray, current_speed: float, current_section: int) -> Tuple[float, float, int]:
        self.tick_counter += 1
        throttle, brake = self._get_throttle_brake(current_location, current_speed, current_section, waypoints)
        gear = max(1, int(current_speed / 60.0))
        if throttle < 0:
            gear = -1
        self.previous_speed = current_speed
        if self.brake_ticks > 0 and brake > 0:
            self.brake_ticks -= 1
        return throttle, brake, gear

    def _get_throttle_brake(self, current_location: np.ndarray, current_speed: float, current_section: int, waypoints: List[roar_py_interface.RoarPyWaypoint]) -> Tuple[float, float]:
        nxt = self._get_next_waypoints(current_location, waypoints)
        r1 = self._radius(nxt[self.close_index:self.close_index+3])
        r2 = self._radius(nxt[self.mid_index:self.mid_index+3])
        r3 = self._radius(nxt[self.far_index:self.far_index+3])
        t1 = self._target_speed(r1, current_section)
        t2 = self._target_speed(r2, current_section)
        t3 = self._target_speed(r3, current_section)
        d1 = self.target_distance[self.close_index] + 3
        d2 = self.target_distance[self.mid_index]
        d3 = self.target_distance[self.far_index]
        sd: List[SpeedData] = []
        sd.append(self._speed_for_turn(d1, t1, current_speed))
        sd.append(self._speed_for_turn(d2, t2, current_speed))
        sd.append(self._speed_for_turn(d3, t3, current_speed))
        if current_speed > 100.0:
            if current_section != 9 and len(nxt) >= self.mid_index + 5:
                r4 = self._radius([nxt[self.mid_index], nxt[self.mid_index+2], nxt[self.mid_index+4]])
                t4 = self._target_speed(r4, current_section)
                sd.append(self._speed_for_turn(d1, t4, current_speed))
            if len(nxt) >= self.close_index + 7:
                r5 = self._radius([nxt[self.close_index], nxt[self.close_index+3], nxt[self.close_index+6]])
                t5 = self._target_speed(r5, current_section)
                sd.append(self._speed_for_turn(d1, t5, current_speed))
        upd = self._select_speed(sd)
        throttle, brake = self._speed_to_controls(upd)
        return throttle, brake

    def _speed_to_controls(self, sd: SpeedData) -> Tuple[float, float]:
        percent_of_max = sd.current_speed / max(1e-6, sd.recommended_speed_now)
        avg_speed_drop_kph = 2.4
        true_percent_drop_tick = avg_speed_drop_kph / max(1e-3, sd.current_speed)
        speed_up_threshold = 0.9
        throttle_dec_mult = 0.7
        throttle_inc_mult = 1.25
        brake_threshold_mult = 0.95
        percent_speed_change = 0.0
        if sd.current_speed + 1e-4 > 0.0:
            percent_speed_change = (sd.current_speed - self.previous_speed) / (self.previous_speed + 1e-4)
        speed_change = sd.current_speed - self.previous_speed
        if percent_of_max > 1.0:
            if percent_of_max > 1.0 + (brake_threshold_mult * true_percent_drop_tick):
                if self.brake_ticks > 0:
                    return -1.0, 1.0
                if self.brake_ticks <= 0 and speed_change < 2.5:
                    self.brake_ticks = max(1, int((sd.current_speed - sd.recommended_speed_now) / 3.0))
                    return -1.0, 1.0
                else:
                    self.brake_ticks = 0
                    return 1.0, 0.0
            else:
                if speed_change >= 2.5:
                    self.brake_ticks = 0
                    return 1.0, 0.0
                throttle_to_maintain = self._throttle_for_speed(sd.current_speed)
                if percent_of_max > 1.02 or percent_speed_change > (-true_percent_drop_tick / 2.0):
                    return throttle_to_maintain * throttle_dec_mult, 0.0
                else:
                    return throttle_to_maintain, 0.0
        else:
            self.brake_ticks = 0
            if speed_change >= 2.5:
                return 1.0, 0.0
            if percent_of_max < speed_up_threshold:
                return 1.0, 0.0
            throttle_to_maintain = self._throttle_for_speed(sd.current_speed)
            if percent_of_max < 0.98 or true_percent_drop_tick < -0.01:
                return throttle_to_maintain * throttle_inc_mult, 0.0
            else:
                return throttle_to_maintain, 0.0

    def _select_speed(self, sds: List[SpeedData]) -> SpeedData:
        best = sds[0]
        for s in sds:
            if s.recommended_speed_now < best.recommended_speed_now:
                best = s
        return best

    def _throttle_for_speed(self, current_speed: float) -> float:
        return 0.75 + current_speed / 500.0

    def _speed_for_turn(self, distance: float, target_speed: float, current_speed: float) -> SpeedData:
        d = (1.0 / 675.0) * (target_speed ** 2) + distance
        max_speed = math.sqrt(825.0 * d)
        return SpeedData(distance, current_speed, target_speed, max_speed)

    def _get_next_waypoints(self, current_location: np.ndarray, more_waypoints: List[roar_py_interface.RoarPyWaypoint]) -> List[roar_py_interface.RoarPyWaypoint]:
        points: List[roar_py_interface.RoarPyWaypoint] = []
        start = roar_py_interface.RoarPyWaypoint(current_location, np.ndarray([0, 0, 0]), 0.0)
        points.append(start)
        curr_dist = 0.0
        for p in more_waypoints:
            end = p
            curr_dist += distance_p_to_p(start, end)
            if curr_dist > self.intended_target_distance[len(points)]:
                self.target_distance[len(points)] = curr_dist
                points.append(end)
                start = end
            else:
                start = end
            if len(points) >= len(self.target_distance):
                break
        return points

    def _radius(self, wp: List[roar_py_interface.RoarPyWaypoint]) -> float:
        p1 = (wp[0].location[0], wp[0].location[1])
        p2 = (wp[1].location[0], wp[1].location[1])
        p3 = (wp[2].location[0], wp[2].location[1])
        l1 = round(math.dist(p1, p2), 3)
        l2 = round(math.dist(p2, p3), 3)
        l3 = round(math.dist(p1, p3), 3)
        small = 2.0
        if l1 < small or l2 < small or l3 < small:
            return self.max_radius
        sp = (l1 + l2 + l3) / 2.0
        area_sq = sp * (sp - l1) * (sp - l2) * (sp - l3)
        if area_sq < small:
            return self.max_radius
        r = (l1 * l2 * l3) / (4.0 * math.sqrt(area_sq))
        return r

    def _target_speed(self, radius: float, current_section: int) -> float:
        mu = 2.65
        if radius >= self.max_radius:
            return self.max_speed
        if current_section == 2:
            mu = 3.25
        if current_section == 3:
            mu = 3.2
        if current_section == 4:
            mu = 2.75
        if current_section == 6:
            mu = 3.2
        if current_section == 9:
            mu = 2.0
        ts = math.sqrt(mu * 9.81 * radius) * 3.6
        return max(20.0, min(ts, self.max_speed))

class LatController:
    def __init__(self, wheelbase: float = 4.7, steer_gain: float = 1.5):
        self.wheelbase = float(wheelbase)
        self.steer_gain = float(steer_gain)

    def run(self, vehicle_location: np.ndarray, vehicle_rotation: np.ndarray, next_waypoint: roar_py_interface.RoarPyWaypoint) -> float:
        waypoint_vector = next_waypoint.location - vehicle_location
        distance_to_waypoint = float(np.linalg.norm(waypoint_vector))
        if distance_to_waypoint == 0.0:
            return 0.0
        waypoint_vector_normalized = waypoint_vector / distance_to_waypoint
        alpha = normalize_rad(vehicle_rotation[2]) - normalize_rad(math.atan2(waypoint_vector_normalized[1], waypoint_vector_normalized[0]))
        steering_command = self.steer_gain * math.atan2(2.0 * self.wheelbase * math.sin(alpha) / distance_to_waypoint, 1.0)
        return float(steering_command)

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat = LatController()
        self.thr = ThrottlePlanner()
        self.num_sections = 10
        self.section_len = 1
        self.current_section = 0
        self.prev_steer = 0.0
    
    async def initialize(self) -> None:
        # TODO: You can do some initial computation here if you want to.
        # For example, you can compute the path to the first waypoint.

        vehicle_location = self.location_sensor.get_last_gym_observation()
        self.maneuverable_waypoints = (
            roar_py_interface.RoarPyWaypoint.load_waypoint_list(
                np.load(f"{os.path.dirname(__file__)}\\waypoints\\waypoints.npz")
            )[35:]
        )
        self.current_waypoint_idx = 0
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        self.section_len = max(1, len(self.maneuverable_waypoints) // self.num_sections)



    async def step(
        self
    ) -> None:
        """
        This function is called every world step.
        Note: You should not call receive_observation() on any sensor here, instead use get_last_observation() to get the last received observation.
        You can do whatever you want here, including apply_action() to the vehicle.
        """
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        v_ms = float(np.linalg.norm(vehicle_velocity))
        v_kmh = v_ms * 3.6
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
        self.current_section = int((self.current_waypoint_idx % len(self.maneuverable_waypoints)) // self.section_len)
        next_idx = self._get_lookahead_index(v_kmh)
        waypoint_to_follow = self._next_waypoint_smooth(v_kmh)
        steer = self.lat.run(vehicle_location, vehicle_rotation, waypoint_to_follow)
        steer_mult = max(1.0, v_kmh / 120.0)
        if self.current_section == 2:
            steer_mult *= 1.2
        if self.current_section == 3:
            steer_mult = np.clip(steer_mult * 1.5, 2.1, 3.1)
        if self.current_section == 4:
            steer_mult = min(1.45, steer_mult * 1.65)
        if self.current_section == 5:
            steer_mult *= 1.1
        if self.current_section == 6:
            steer_mult = np.clip(steer_mult * 5.5, 5.5, 7.0)
        if self.current_section == 7:
            steer_mult *= 2.0
        if self.current_section == 9:
            steer_mult = max(steer_mult, 1.6)
        waypoints_for_throttle = (self.maneuverable_waypoints * 2)[next_idx:next_idx+300]
        throttle, brake, gear = self.thr.run(waypoints_for_throttle, vehicle_location, v_kmh, self.current_section)
        if self.current_section == 3 and v_kmh > 220.0 and throttle > 0.0:
            throttle = min(throttle, 0.2)
        steer_cmd = float(np.clip(steer * steer_mult, -1.0, 1.0))
        if self.current_section == 3 and v_kmh > 120.0:
            max_delta = float(np.clip(0.22 - 0.10 * float(np.clip((v_kmh - 120.0) / 120.0, 0.0, 1.0)), 0.08, 0.22))
            steer_cmd = float(np.clip(steer_cmd, self.prev_steer - max_delta, self.prev_steer + max_delta))
        if brake <= 0.0 and abs(steer_cmd) < 0.045 and v_kmh < 303.0:
            throttle = 1.0
        self.prev_steer = steer_cmd
        control = {
            "throttle": float(np.clip(throttle, 0.0, 1.0)),
            "steer": steer_cmd,
            "brake": float(np.clip(brake, 0.0, 1.0)),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": gear
        }
        await self.vehicle.apply_action(control)
        return control

    def _lookahead_value(self, v_kmh: float) -> int:
        d = {
            90: 9,
            110: 11,
            130: 14,
            160: 18,
            180: 22,
            200: 26,
            250: 30,
            300: 35
        }
        for s, n in d.items():
            if v_kmh < s:
                return n
        return 8

    def _get_lookahead_index(self, v_kmh: float) -> int:
        n = self._lookahead_value(v_kmh)
        return int((self.current_waypoint_idx + n) % len(self.maneuverable_waypoints))

    def _next_waypoint_smooth(self, v_kmh: float) -> roar_py_interface.RoarPyWaypoint:
        if v_kmh > 70.0 and v_kmh < 300.0:
            return self._average_point(v_kmh)
        else:
            idx = self._get_lookahead_index(v_kmh)
            return self.maneuverable_waypoints[idx]

    def _average_point(self, v_kmh: float) -> roar_py_interface.RoarPyWaypoint:
        idx = self._get_lookahead_index(v_kmh)
        base = self._lookahead_value(v_kmh)
        num = base * 2
        if self.current_section == 0:
            num = int(round(base * 1.5))
        if self.current_section == 3:
            idx = self.current_waypoint_idx + 22
            num = 35
        if self.current_section == 4:
            num = base + 5
            idx = self.current_waypoint_idx + 24
        if self.current_section == 5:
            num = base
        if self.current_section == 6:
            num = 5
            idx = self.current_waypoint_idx + 28
        if self.current_section == 7:
            num = int(round(base * 1.25))
        if self.current_section == 9:
            num = 0
        start = (idx - (num // 2)) % len(self.maneuverable_waypoints)
        next_wp = self.maneuverable_waypoints[idx % len(self.maneuverable_waypoints)]
        next_loc = next_wp.location
        if num > 3:
            samples = [(start + i) % len(self.maneuverable_waypoints) for i in range(0, num)]
            loc_sum = None
            for i, s in enumerate(samples):
                if i == 0:
                    loc_sum = self.maneuverable_waypoints[s].location.copy()
                else:
                    loc_sum = loc_sum + self.maneuverable_waypoints[s].location
            cnt = len(samples)
            new_loc = loc_sum / cnt
            shift = float(np.linalg.norm(next_loc - new_loc))
            max_shift = 2.0
            if self.current_section == 1:
                max_shift = 0.2
            if shift > max_shift:
                uv = (new_loc - next_loc) / shift
                new_loc = next_loc + uv * max_shift
            return roar_py_interface.RoarPyWaypoint(new_loc, np.ndarray([0, 0, 0]), 0.0)
        else:
            return next_wp
