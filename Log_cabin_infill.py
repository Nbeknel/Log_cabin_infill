import re
import argparse
import os
import random
import math


class LineSegment:
    def __init__(self, x0: float, y0: float, x1: float, y1: float,
            width: float):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.width = width
    
    @property
    def length(self):
        return math.sqrt((self.x1 - self.x0) ** 2 + (self.y1 - self.y0) ** 2)


class MyObject:
    def __init__(self, layer_height=0, layer_z=0):
        self.previous_layer = []
        self.current_layer = []
        self.previous_layer_height = 0
        self.current_layer_height = layer_height
        self.layer_z = layer_z
        self.max_width = 0
        
    def new_layer(self, layer_z):
        self.previous_layer = self.current_layer
        self.current_layer = []
        self.previous_layer_height = self.current_layer_height
        self.current_layer_height = layer_z - self.layer_z
        self.layer_z = layer_z

    def add_line_segment(self, line_segment: LineSegment):
        self.current_layer.append(line_segment)

    def set_width(self, width: float):
        self.width = max(self.width, width)


def intersect(segment_0: LineSegment, segment_1: LineSegment,
		overhang_threshold=0.5):
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
    if abs(delta) < (r0 * r1) * math.sin(math.pi / 180):
        x = 0.5 * (segment_0.x0 + segment_0.x1)
        y = 0.5 * (segment_0.y0 + segment_0.y1)
        n0 = abs(dy1 * (x - segment_1.x0) - dx1 * (y - segment_1.y0)) / r1
        x = 0.5 * (segment_1.x0 + segment_1.x1)
        y = 0.5 * (segment_1.y0 + segment_1.y1)
        n1 = abs(dy0 * (x - segment_0.x0) - dx0 * (y - segment_0.y0)) / r0
        normal_distance = min(n0, n1)
        if normal_distance > segment_0.width * overhang_threshold:
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
    
    csc_angle = (r0 * r1) / abs(delta)
    half_width = 0.5 * segment_1.width * csc_angle
    
    t0 = (dx1 * dy - dy1 * dx) / delta
    t1 = (dx0 * dy - dy0 * dx) / delta
    
    if 0 <= t0 <= 1 and 0 <= t1 <= 1:
        return (max(0, t0 - half_width / r0), min(1, t0 + half_width / r0))
    return False


def get_extrusion_multiplier(line: LineSegment, current_layer_height: float,
        previous_layer_height: float) -> float:
    alpha = 1 - 0.25 * math.pi
    current_section = (line.width - alpha * current_layer_height)\
            * current_layer_height
    previous_section = (line.width - alpha * previous_layer_height)\
            * previous_layer_height
    rect_section = line.width * (current_layer_height + previous_layer_height)
    return 0.5 * (current_section + previous_section + rect_section)\
            / current_section


def process_g_code(input_file: str, minimum_length: float = 0):
    internal_infill = ["Internal infill"]
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
    
    objects[current_object] = MyObject()
    
    temp_file = re.sub(r"\.+", "_", input_file) + ".temp"
    with open(input_file, 'r') as input_lines,\
            open(temp_file, 'w') as temp_lines:
        for line in input_lines:
            additional_lines = []
            
            # Extract data per object in case different objects have
            # different layer heights, or if an object is sliceed with
            # variable layer height
            match = re.search(r"; printing object ([^\n]*)", line)
            if match:
                current_object = match.group(1)
                if current_object in list(objects):
                    objects[current_object].new_layer(layer_z=current_z)
                else:
                    objects[current_object] = MyObject(layer_height, current_z)
                temp_lines.write(line)
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
            match = re.search(r"G1 [^XYEF\n]*F([-\d\.]+)", line)
            if match:
                speed = float(match.group(1))
                if line_type is not INTERNAL_INFILL:
                    temp_lines.write(line)
                continue
            
            # Get current layer height
            match = re.search(r";Z:([\.\d]+)", line)
            if match:
                current_z = float(match.group(1))
                temp_lines.write(line)
                continue
            
            # Get current line width
            match = re.search(r";WIDTH:([\.\d]+)", line)
            if match:
                width = float(match.group(1))
                temp_lines.write(line)
                objects[current_object].set_width(width)
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
                temp_lines.write(line)
                continue
            
            if line.startswith("G1") and ('X' in line or 'Y' in line)\
                    and 'E' in line and "E-" not in line:
                line_segment = LineSegment(previous_x, previous_y, current_x,
                        current_y, width)
                objects[current_object].add_line_segment(line_segment)
                
                if line_type is INTERNAL_INFILL\
                        and line_segment.length > line_segment.width:
                    if minimum_length == 0:
                        minimum_length = 0.5 * objects[current_object].width
                    increase_intervals = [[0, 1]]
                    for previous_layer_line in\
                            objects[current_object].previous_layer:
                        intersection = intersect(line_segment,
                                previous_layer_line)
                        if intersection:
                            new_intervals = []
                            start = intersection[0]
                            stop = intersection[0]
                            # TODO: create separate function, take segment length into account
                            for interval in increase_intervals:
                                if stop <= interval[0]:
                                    new_intervals.append(interval)
                                    continue
                                if start >= interval[1]:
                                    new_intervals.append(interval)
                                    continue
                                if start <= interval[0] and stop < interval[1]:
                                    if interval[1] - stop > minimum_length:
                                        new_intervals.append([stop, interval[1]])
                                    continue
                                if start > interval[0] and stop >= interval[1]:
                                    if start - interval[0] > minimum_length:
                                        new_intervals.append([interval[0], start])
                                    continue
                                if start > interval[0] and stop < interval[1]:
                                    if interval[1] - stop > minimum_length:
                                        new_intervals.append([stop, interval[1]])
                                    if start - interval[0] > minimum_length:
                                        new_intervals.append([interval[0], start])
                                    continue
                            increase_intervals = new_intervals
                    
                    extrusion_multiplier = get_extrusion_multiplier(line_segment,
                            objects[current_object].current_layer_height,
                            objects[current_object].previous_layer_height)
                    e_value = float(re.search(r"E([\.\d]+)", line).group(1))
                    speed_increased_extrusion = speed / extrusion_multiplier

                    # TODO!!!
                    for i, interval in enumerate(increase_intervals):
                        if i == 0:
                            if interval[0] == 0:
                                pass
                            else:
                                pass
                        else:
                            pass
                    t_current = 0
                    t_previous = 0
                    increase_flow = 1
                    for increase_point in increase_points:
                        t_previous = t_current
                        t_current = increase_point[0]
                        if increase_flow > 0:
                            if abs(t_current - t_previous) < 1e-6:
                                increase_flow += increase_point[1]
                                continue
                            additional_lines.append(
                                    f"G1 F{speed_increased_extrusion:.3f}\n"
                            )
                            x = (1 - t_current) * line_segment.x0\
                                    + t_current * line_segment.x1
                            y = (1 - t_current) * line_segment.y0\
                                    + t_current * line_segment.y1
                            e = (t_current - t_previous) * e_value\
                                    * extrusion_multiplier
                            additional_lines.append(
                                    f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                            )
                        else:
                            if abs(t_current - t_previous) < 1e-6:
                                increase_flow += increase_point[1]
                                continue
                            additional_lines.append(
                                    f"G1 F{speed:.3f}\n"
                            )
                            x = (1 - t_current) * line_segment.x0\
                                + t_current * line_segment.x1
                            y = (1 - t_current) * line_segment.y0\
                                + t_current * line_segment.y1
                            e = (t_current - t_previous) * e_value
                            additional_lines.append(
                                    f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                            )
                        increase_flow += increase_point[1]
                    if t_current < 1 or len(additional_lines) == 0:
                        if increase_flow > 0:
                            if abs(t_current - t_previous) < 1e-6:
                                continue
                            additional_lines.append(
                                    f"G1 F{speed_increased_flow:.3f}\n"
                            )
                            x = line_segment.x1
                            y = line_segment.y1
                            e = (1 - t_current) * e_value * flow_multiplier
                            additional_lines.append(
                                    f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                            )
                        else:
                            if abs(t_current - t_previous) < 1e-6:
                                continue
                            additional_lines.append(
                                    f"G1 F{speed:.3f}\n"
                            )
                            x = line_segment.x1
                            y = line_segment.y1
                            e = (1 - t_current) * e_value
                            additional_lines.append(
                                    f"G1 X{x:.3f} Y{y:.3f} E{e:.5f}\n"
                            )
                    #additional_lines.append(f"G1 F{speed:.3f}\n")
                else:
                    additional_lines.append(line)
            else:
                additional_lines.append(line)

            temp_lines.writelines(additional_lines)

    with open(temp_file, 'r') as temp_lines,\
            open(input_file, 'w') as input_lines:
        for line in temp_lines:
            input_lines.write(line)
    os.remove(temp_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-code post-processing "
            "script to increase flow of sparse infill lines along intervals "
            "that are not supported by infill from the previous layer.")
    parser.add_argument("input_file", help="Path to the input G-code file")
    args = parser.parse_args()
    
    process_g_code(args.input_file)
    pass