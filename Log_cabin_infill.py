#!/usr/bin/python

import re
import argparse
import os
import random
import math

# Slowdown method
SCALAR = False
AUTOMATIC = True

# Consider this an immutable class that stores all arguments
class ScriptConfig:
    def __init__(self, parser: argparse.ArgumentParser):
        args = parser.parse_args()
        nozzle_diameter = 0.4
        relative_e = True
        gcode_flavor = "marlin2"
        if "SLIC3R_NOZZLE_DIAMETER" in list(os.environ):
            # Multiple extruder support will be added later
            nozzle_diameter =\
                    float(os.environ["SLIC3R_NOZZLE_DIAMETER"].split(',')[0])
            relative_e =\
                    bool(int(os.environ["SLIC3R_USE_RELATIVE_E_DISTANCES"]))
            gcode_flavor = os.environ["SLIC3R_GCODE_FLAVOR"]
        else:
            with open(args.input_file, 'r') as input_lines:
                break_flags = [False] * 3
                for line in input_lines:
                    match = re.search(r"; nozzle_diameter = ([\.\d]+)", line)
                    if match:
                        nozzle_diameter = float(match.group(1))
                        break_flags[0] = True
                    
                    match = re.search(r"; use_relative_e_distances = (\d)",
                            line
                    )
                    if match:
                        relative_e = bool(int(match.group(1)))
                        break_flags[1] = True

                    match = re.search(r"^; gcode_flavor = (.+)$", line)
                    if match:
                        gcode_flavor = match.group(1)
                        break_flags[2] = True

                    if all(break_flags):
                        break

        minimum_length = args.minimum_length
        # Ensures the given string is of an allowed form with percentages
        # e.g. 50%, 12.5%, 0.45%
        match_percent = re.search(r"^(\d*(\.\d+)?)%$", minimum_length)
        match_float = re.search(r"^(\d*(\.\d+)?)$", minimum_length)
        if match_percent:
            minimum_length = nozzle_diameter * 0.01\
                    * float(match_percent.group(1))
        elif match_float:
            minimum_length = float(match_float.group(1))
        else:
            minimum_length = 0.5 * nozzle_diameter

        overhang_threshold = args.overhang_threshold
        match_percent = re.search(r"^(\d*(\.\d+)?)%$", overhang_threshold)
        match_float = re.search(r"^(\d*(\.\d+)?)$", overhang_threshold)
        if match_percent:
            overhang_threshold = nozzle_diameter * 0.01\
                    * float(match_percent.group(1))
        elif match_float:
            overhang_threshold = float(match_float.group(1))
        else:
            overhang_threshold = 0.5 * nozzle_diameter

        slowdown_speed = args.slowdown_speed
        slowdown_method = SCALAR
        slowdown_coefficient = 0
        match_percent = re.search(r"^(\d*(\.\d+)?)%$", slowdown_speed)
        match_float = re.search(r"^(\d*(\.\d+)?)$", slowdown_speed)
        if slowdown_speed == "-1":
            slowdown_method = AUTOMATIC
        elif match_percent:
            slowdown_coefficient = 0.01 * float(match_percent.group(1))
        elif match_float:
            slowdown_coefficient = float(match_float.group(1))
        
        # Do not depend on the object.
        self.input_file = args.input_file
        self.relative_e = relative_e
        
        # May depend on the object, therefore need getters and setters.
        self.minimum_length = {"default": minimum_length}
        self.overhang_threshold = {"default": overhang_threshold}
        self.slowdown_method = {"default": slowdown_method}
        self.slowdown_coefficient = {"default": slowdown_coefficient}

        # Get objects from g-code file
        self.objects = {}
        with open(args.input_file, 'r') as input_lines:
            slicer = None
            previous_line = ""
            index = 0
            for line in input_lines:
                if slicer == "SuperSlicer":
                    match = re.search(r'"id":"(.*?)"', line)
                    if match:
                        object_id = match.group(1)
                        self.objects[object_id] = index
                        # https://github.com/kageurufu/preprocess_cancellation/blob/main/preprocess_cancellation.py#L145
                        object_id_klipper =\
                                re.sub(r"\W+", "_", object_id).strip("_")
                        self.objects[object_id_klipper] = index
                        self.objects[index] = index
                        # ToDo: object id parsing for per object settings
                        index += 1
                    if line.startswith(";TYPE"):
                        # So as not to read the whole file
                        break
                elif slicer == "PrusaSlicer":
                    if line.startswith("; objects_info"):
                        object_names = re.findall(r'"name":"(.*?)"', line)
                        for index, object_name in enumerate(object_names):
                            self.objects[object_name] = index
                            object_id_klipper =\
                                    re.sub(r"\W+", "_", object_name).strip("_")
                            self.objects[object_id_klipper] = index
                            self.objects[index] = index
                            # ToDo: object id parsing for per object settings
                        break
                elif slicer == "OrcaSlicer":
                    object_id = None
                    match = re.search(r"M486 A(\w+)", line)
                    if match:
                        object_id = match.group(1)

                    match = re.search(r"NAME=(\w+)", line)
                    if match:
                        object_id = match.group(1)
                    
                    if object_id is not None:
                        self.objects[object_id] = index
                        self.objects[index] = index
                        index += 1
                    
                    if line.startswith(";TYPE"):
                        # So as not to read the whole file
                        break
                else:
                    match = re.search(r"(\w*Slicer)", line)
                    if match:
                        slicer = match.group(1)

    
    def set_and_get_current_object(self, current_object):
        if current_object not in list(self.objects):
            current_object = re.sub(r"\W+", "_", current_object).strip("_")
            if current_object not in list(self.objects):
                input("Something went wrong with object recognition. Press enter to exit.")
                raise Exception("Something went wrong with object recognition.")
        self.current_object = self.objects[current_object]
        return self.current_object
    
    def get_minimum_length(self) -> float:
        if self.current_object in list(self.minimum_length):
            return self.minimum_length[self.current_object]
        return self.minimum_length["default"]

    def get_overhang_threshold(self) -> float:
        if self.current_object in list(self.overhang_threshold):
            return self.overhang_threshold[self.current_object]
        return self.overhang_threshold["default"]
    
    def get_slowdown_method(self) -> float:
        if self.current_object in list(self.slowdown_method):
            return self.slowdown_method[self.current_object]
        return self.slowdown_method["default"]
    
    def get_slowdown_coefficient(self) -> float:
        if self.current_object in list(self.slowdown_coefficient):
            return self.slowdown_coefficient[self.current_object]
        return self.slowdown_coefficient["default"]


# An extrusion line is stored as this class. Stores start and end
# coordinates, extrusion width and length.
class LineSegment:
    def __init__(self, x0: float, y0: float, x1: float, y1: float,
            width: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = width
        self.length = math.sqrt((self.x1 - self.x0) ** 2
                + (self.y1 - self.y0) ** 2)


# For each object on the buildplate store all extrusion lines of the
# previous and current layer, layer heights and current maximum width
# encountered.
class GcodeObject:
    def __init__(self, layer_height=0, layer_z=0):
        self.previous_layer = []
        self.current_layer = []
        self.previous_layer_height = 0
        self.current_layer_height = layer_height
        self.layer_z = layer_z
        self.previous_z = layer_z - layer_height
        self.has_only_support = True
        
    def new_layer(self, layer_z):
        if layer_z > self.layer_z:
            if not self.has_only_support:
                self.previous_layer_height = self.current_layer_height
                self.previous_z = self.layer_z
                self.previous_layer = self.current_layer
                
            self.current_layer_height = layer_z - self.previous_z
            self.layer_z = layer_z
            self.has_only_support = True
            self.current_layer = []        

    def add_line_segment(self, line_segment: LineSegment):
        self.current_layer.append(line_segment)


# For a line in the current layer, find at which it is supported by line
# of the previous layer if the latter supports it at all. Return the
# interval that is supported relative to the total length based on the
# widths and intersection angle.
def intersect(segment_0: LineSegment, segment_1: LineSegment,
		script_config: ScriptConfig):
    epsilon = 1e-6
    # segment_0: in current layer
    # segment_1: in previous layer
    dx0 = segment_0.x1 - segment_0.x0
    dy0 = segment_0.y1 - segment_0.y0
    
    dx1 = segment_1.x1 - segment_1.x0
    dy1 = segment_1.y1 - segment_1.y0
    
    delta = dx1 * dy0 - dx0 * dy1
    # Lengths of both line segments
    r0 = segment_0.length
    r1 = segment_1.length

    dx = segment_1.x0 - segment_0.x0
    dy = segment_1.y0 - segment_0.y0
    
    # If both line segments are (nearly) parallel
    if abs(delta) < (r0 * r1) * math.sin(math.pi / 36):
        x = 0.5 * (segment_0.x0 + segment_0.x1)
        y = 0.5 * (segment_0.y0 + segment_0.y1)
        n0 = abs(dy1 * (x - segment_1.x0) - dx1 * (y - segment_1.y0)) / r1
        x = 0.5 * (segment_1.x0 + segment_1.x1)
        y = 0.5 * (segment_1.y0 + segment_1.y1)
        n1 = abs(dy0 * (x - segment_0.x0) - dx0 * (y - segment_0.y0)) / r0
        normal_distance = min(n0, n1)
        
        # If not supported or barely supported.
        if normal_distance > script_config.get_overhang_threshold():
            return False

        t0 = (dx0 * dx + dy0 * dy) / (r0 ** 2)

        dx = segment_1.x1 - segment_0.x0
        dy = segment_1.y1 - segment_0.y0
        t1 = (dx0 * dx + dy0 * dy) / (r0 ** 2)
        t_lower = min(1, max(0, min(t0, t1)))
        t_upper = min(1, max(0, max(t0, t1)))
        if t_lower != t_upper:
            return (t_lower, t_upper)
        return False
    
    # Calculate the length of subsegment that is supported based on the
    # angle between the supporting and supported line.
    csc_angle = (r0 * r1) / abs(delta)
    half_width = 0.5 * segment_1.width * csc_angle
    
    t0 = (dx1 * dy - dy1 * dx) / delta
    t1 = (dx0 * dy - dy0 * dx) / delta
    
    # Without the margin this algorithm finds the intersection of the
    # centerlines of an extrusion. By udding a margin value we can find
    # out whether the start or end point is supported by the previous
    # layer.
    margin = script_config.get_overhang_threshold() / r0
    
    if -margin <= t0 <= 1 + margin and -margin <= t1 <= 1 + margin:
        return (max(- margin, t0 - half_width / r0),\
                min(1 + margin, t0 + half_width / r0))
    return False


# Calculate the extrusion multiplier for segments that are not
# supported, by finding the areas of the cross sections of the current
# layer and previous layer for a given line width, and then dividing
# their sum by the cross sectional area of the current layer.
def get_extrusion_multiplier(line: LineSegment, current_layer_height: float,
        previous_layer_height: float) -> float:
    alpha = 1 - 0.25 * math.pi
    current_section = (line.width - alpha * current_layer_height)\
            * current_layer_height
    previous_section = (line.width - alpha * previous_layer_height)\
            * previous_layer_height
    #rect_section = line.width * (current_layer_height + previous_layer_height)
    #return 0.5 * (current_section + previous_section + rect_section)\
    #        / current_section
    return (current_section + previous_section) / current_section


# Find the result of the subtraction of an interval from a union of
# intervals.
# `increase_intervals` := A = U{[a_i, b_i]| 0 <= i <= n, b_{i-1} < a_i < b_i < a_{i+1}}
# `exclude_interval` := B = [a, b]
# C = A \ B
# D = U{c in C| |c| >= minimum_length}
# return D
def exclude_interval(increase_intervals: list, exclude_interval: tuple,
        segment_length: float, script_config: ScriptConfig) -> list:
    new_intervals = []
    start = exclude_interval[0]
    stop = exclude_interval[1]
    threshold = script_config.get_minimum_length() / segment_length
    for interval in increase_intervals:
        if stop <= interval[0]:
            new_intervals.append(interval)
            continue
        if start >= interval[1]:
            new_intervals.append(interval)
            continue
        if start <= interval[0] and stop <= interval[1]:
            if interval[1] - stop > threshold:
                new_intervals.append([stop, interval[1]])
            continue
        if start >= interval[0] and stop >= interval[1]:
            if start - interval[0] > threshold:
                new_intervals.append([interval[0], start])
            continue
        if start >= interval[0] and stop <= interval[1]:
            if interval[1] - stop > threshold:
                new_intervals.append([stop, interval[1]])
            if start - interval[0] > threshold:
                new_intervals.append([interval[0], start])
            continue
    return sorted(new_intervals, key=lambda x: x[0])
    

# The automatic method returns the average speed over a given distance
# with a certain initial speed, target speed and final speed, which is
# assumed to be the speed when extrusion is increased.
def get_speed(target_speed: float, previous_speed: float, slow_speed: float,
        acceleration: float, distance: float, script_config: ScriptConfig
) -> float:
    target_speed /= 60
    previous_speed /= 60
    slow_speed /= 60
    
    if script_config.get_slowdown_method() is SCALAR:
        return 60 * (script_config.get_slowdown_coefficient() * slow_speed\
                + (1 - script_config.get_slowdown_coefficient()) * target_speed)

    # Slowdown method is automatic
    if previous_speed >= target_speed:
        return 59 * target_speed + slow_speed
    if 3 * (previous_speed ** 2) + slow_speed ** 2\
            > 4 * previous_speed * slow_speed + acceleration * distance:
        return 30 * (slow_speed + previous_speed)
    t_dec = (previous_speed - slow_speed) / acceleration
    d_dec = previous_speed * t_dec - 0.5 * acceleration * (t_dec ** 2)
    d = distance - d_dec
    t_acc = (target_speed - previous_speed) / acceleration
    d_acc = 2 * previous_speed * t_acc + acceleration * (t_acc ** 2)
    if d < d_acc:
        t_acc = (math.sqrt(previous_speed ** 2 + acceleration * d)\
                - previous_speed) / acceleration
        t_total = t_dec + 2 * t_acc
        return 60 * distance / t_total
    d_coast = d - d_acc
    t_coast = d_coast / target_speed
    t_total = t_dec + 2 * t_acc + t_coast
    return 60 * distance / t_total


def process_g_code(script_config: ScriptConfig):
    internal_infill = ["Internal infill", "Sparse infill"]
    support_material = ["Support material", "Support material interface"]
    OTHER = 0
    INTERNAL_INFILL = 1
    
    INCREASE = 1
    DECREASE = -1

    objects = {}
    current_object = "Start G-code"
    current_x = 0
    current_y = 0
    current_z = 0
    previous_x = 0
    previous_y = 0
    width = 0
    layer_height = 0
    line_type = OTHER
    speed = 0
    previous_speed = 0
    acceleration = 0
    
    objects[current_object] = GcodeObject()
    
    # Open two files. Read a line from one, modify it if required, and
    # write to the second file.
    temp_file = re.sub(r"\.+", "_", script_config.input_file) + ".temp"
    with open(script_config.input_file, 'r') as input_lines,\
            open(temp_file, 'w') as temp_lines:
        for line in input_lines:
            additional_lines = []
            
            # Extract data per object in case different objects have
            # different layer heights, or if an object is sliced with
            # variable layer height
            match_object_start = []
            # Match OctoPrint labels
            match_object_start.append(re.search(r"; printing object ([^\n]*)", line))
            # Match Klipper labels
            match_object_start.append(re.search(r"EXCLUDE_OBJECT_START NAME='?(\w+)", line))
            # Match Marlin 2 labels
            match_object_start.append(re.search(r"M486 S(\d+)", line))
            if any(match_object_start):
                for match in match_object_start:
                    if match:
                        current_object = match.group(1)
                        current_object = script_config.set_and_get_current_object(current_object)
                        if current_object not in list(objects):
                            objects[current_object] = GcodeObject(layer_height, current_z)
                        else:
                            objects[current_object].new_layer(current_z)
                        temp_lines.write(line)
                        continue
            
            # Get current layer height
            match = re.search(r";Z:([\.\d]+)", line)
            if match:
                current_z = float(match.group(1))
                temp_lines.write(line)
                objects[current_object].new_layer(current_z)
                continue
            
            # Get X coordinate from G1 move
            match = re.search(r"G1 [^X\n]*X([-\d\.]+)", line)
            if match:
                previous_x = current_x
                current_x = float(match.group(1))
    
            # Get Y coordinate from G1 move
            match = re.search(r"G1 [^Y\n]*Y([-\d\.]+)", line)
            if match:
                previous_y = current_y
                current_y = float(match.group(1))
            
            # Get toolhead speed from G1 command
            match = re.search(r"G1 [^ZXYEF\n]*F([\d\.]+)", line)
            if match:
                previous_speed = speed
                speed = float(match.group(1))
                if line_type is not INTERNAL_INFILL:
                    temp_lines.write(line)
                continue
            
            # Get toolhead acceleration from M204 command
            match = re.search(r"M204 [^S\n]*S([\d\.]+)", line)
            if match:
                acceleration = float(match.group(1))
                temp_lines.write(line)
                continue
                
            # Get acceleration from SET_VELOCITY_LIMIT (OrcaSlicer)
            match = re.search(r"ACCEL=([\d\.]+)", line)
            if match:
                acceleration = float(match.group(1))
                temp_lines.write(line)
                continue
            
            # Get current line width
            match = re.search(r";WIDTH:([\.\d]+)", line)
            if match:
                width = float(match.group(1))
                temp_lines.write(line)
                continue

            # Get current line height
            match = re.search(r";HEIGHT:([\.\d]+)", line)
            if match:
                layer_height = float(match.group(1))
                temp_lines.write(line)
                continue
            
            # Detect perimeter types from G-code comments
            match = re.search(r";TYPE:([^\n]*)", line)
            if match:
                if match.group(1) in internal_infill:
                    line_type = INTERNAL_INFILL
                else:
                    line_type = OTHER
                    
                if match.group(1) not in support_material:
                    objects[current_object].has_only_support = False
                temp_lines.write(line)
                continue
            
            # Find extrusion lines inside internal infill and modify
            # them, increase extrusion and slow down toolhead speed for
            # unsupported segments.
            if line.startswith("G1") and ('X' in line or 'Y' in line)\
                    and 'E' in line and "E-" not in line:
                line_segment = LineSegment(previous_x, previous_y, current_x,
                        current_y, width)
                objects[current_object].add_line_segment(line_segment)
                
                if line_type is INTERNAL_INFILL:
                    increase_intervals = [[0, 1]]
                    for previous_layer_line in\
                            objects[current_object].previous_layer:
                        intersection = intersect(line_segment,
                                previous_layer_line, script_config)
                        if intersection:
                            increase_intervals = exclude_interval(
                                    increase_intervals, intersection,
                                    line_segment.length, script_config
                            )
                    
                    extrusion_multiplier = get_extrusion_multiplier(
                            line_segment,
                            objects[current_object].current_layer_height,
                            objects[current_object].previous_layer_height
                    )
                    e_value = float(re.search(r"E([\.\d]+)", line).group(1))
                    speed_increased_extrusion = speed / extrusion_multiplier
                    increased_height = objects[current_object].current_layer_height\
                            + objects[current_object].previous_layer_height
                    
                    if len(increase_intervals) > 0:
                        for i, interval in enumerate(increase_intervals):
                            if i == 0:
                                if interval[0] == 0:
                                    x = (1 - interval[1]) * line_segment.x0\
                                            + interval[1] * line_segment.x1
                                    y = (1 - interval[1]) * line_segment.y0\
                                            + interval[1] * line_segment.y1
                                    e = interval[1] * e_value\
                                            * extrusion_multiplier
                                    additional_lines.append(
                                            f";HEIGHT:{increased_height:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 F{speed_increased_extrusion:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                                    )
                                    previous_speed = speed_increased_extrusion
                                else:
                                    x = (1 - interval[0]) * line_segment.x0\
                                            + interval[0] * line_segment.x1
                                    y = (1 - interval[0]) * line_segment.y0\
                                            + interval[0] * line_segment.y1
                                    e = interval[0] * e_value
                                    distance = interval[0]\
                                            * line_segment.length
                                    slowdown_speed = get_speed(
                                            speed, previous_speed,
                                            speed_increased_extrusion,
                                            acceleration, distance,
                                            script_config
                                    )
                                    additional_lines.append(
                                            f";HEIGHT:{objects[current_object].current_layer_height:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 F{slowdown_speed:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                                    )
                                    x = (1 - interval[1]) * line_segment.x0\
                                            + interval[1] * line_segment.x1
                                    y = (1 - interval[1]) * line_segment.y0\
                                            + interval[1] * line_segment.y1
                                    e = (interval[1] - interval[0] ) * e_value\
                                            * extrusion_multiplier
                                    additional_lines.append(
                                            f";HEIGHT:{increased_height:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 F{speed_increased_extrusion:.3f}\n"
                                    )
                                    additional_lines.append(
                                            f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                                    )
                                    previous_speed = speed_increased_extrusion
                            else:
                                x = (1 - interval[0]) * line_segment.x0\
                                        + interval[0] * line_segment.x1
                                y = (1 - interval[0]) * line_segment.y0\
                                        + interval[0] * line_segment.y1
                                e = (interval[0] - increase_intervals[i - 1][1])\
                                        * e_value
                                distance = (interval[0] - increase_intervals[i - 1][1])\
                                        * line_segment.length
                                slowdown_speed = get_speed(
                                        speed, previous_speed,
                                        speed_increased_extrusion,
                                        acceleration, distance, script_config
                                )
                                additional_lines.append(
                                        f";HEIGHT:{objects[current_object].current_layer_height:.3f}\n"
                                )
                                additional_lines.append(
                                        f"G1 F{slowdown_speed:.3f}\n"
                                )
                                additional_lines.append(
                                        f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                                )
                                x = (1 - interval[1]) * line_segment.x0\
                                        + interval[1] * line_segment.x1
                                y = (1 - interval[1]) * line_segment.y0\
                                        + interval[1] * line_segment.y1
                                e = (interval[1] - interval[0] ) * e_value\
                                        * extrusion_multiplier
                                additional_lines.append(
                                        f";HEIGHT:{increased_height:.3f}\n"
                                )
                                additional_lines.append(
                                        f"G1 F{speed_increased_extrusion:.3f}\n"
                                )
                                additional_lines.append(
                                        f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                                )
                                previous_speed = speed_increased_extrusion
                        if increase_intervals[-1][1] < 1:
                            x = line_segment.x1
                            y = line_segment.y1
                            e = (1 - increase_intervals[-1][1]) * e_value
                            distance = (1 - increase_intervals[-1][1])\
                                    * line_segment.length
                            slowdown_speed = get_speed(
                                    speed, previous_speed, previous_speed,
                                    acceleration, distance, script_config
                            )
                            additional_lines.append(
                                    f";HEIGHT:{objects[current_object].current_layer_height:.3f}\n"
                            )
                            additional_lines.append(
                                    f"G1 F{slowdown_speed:.3f}\n"
                            )
                            additional_lines.append(
                                    f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                            )
                            previous_speed = slowdown_speed
                        additional_lines.append(
                                f";HEIGHT:{objects[current_object].current_layer_height:.3f}\n"
                        )
                    else:
                        distance = line_segment.length
                        slowdown_speed = get_speed(
                                            speed, previous_speed,
                                            previous_speed, acceleration,
                                            distance, script_config
                        )
                        additional_lines.append(f"G1 F{slowdown_speed:.3f}\n")
                        additional_lines.append(line)
                        previous_speed = slowdown_speed
                else:
                    if abs(previous_speed - speed) > 1e-6:
                        additional_lines.append(f"G1 F{speed:.3f}\n")
                        previous_speed = speed
                    additional_lines.append(line)
            else:
                additional_lines.append(line)

            temp_lines.writelines(additional_lines)

    # Copy G-code from the temporary file to the main file, and delete
    # the temporaryfile from the folder.
    with open(temp_file, 'r') as temp_lines,\
            open(script_config.input_file, 'w') as input_lines:
        for line in temp_lines:
            input_lines.write(line)
    os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
"""G-code post-processing script to increase flow of sparse infill lines
along intervals that are not supported by infill from the previous
layer.""")
    parser.add_argument("input_file",
            help="Path to the input G-code file.")
    parser.add_argument("--minimum_length", type=str, default="50%",
            help=
"""The script will not try to extrude into gaps smaller than this
distance. If expressed as a percentage it is calculated over the nozzle
diameter.
Default value: 50%.""")
    parser.add_argument("--overhang_threshold", type=str, default="50%",
            help=
"""The script will not try to increase extrusion when a line or
subsegment is supported by at least this distance. If expressed as a
percentage it is calculated over the nozzle diameter.
Default value: 50%.""")
    parser.add_argument("--slowdown_speed", type=str, default="0",
            help=
"""When set to zero, the script will slow down infill speed to maintain
a constant volumetric flow. This will result in jittering movements when
printing infill. With this setting you can reduce the print speed of the
fast segments of infill, since those will be on the smaller side, and
the target speed most likely won't be reached. A value between 0 and 1
will linearly interpolate between the default infill speed and the
reduced infill speed, with 0 - default speed and 1 - reduced infill
speed. Can be expressed as a percentage (50% = 0.5). If set to -1, the
script will create a dynamic slowdown loosely based on Klipper's
ACCEL_TO_DECEL algorithm.
Default value: 0.""")
    
    script_config = ScriptConfig(parser)

    if script_config.relative_e:
        process_g_code(script_config)
    else:
        print("Change to relative e distances.")
