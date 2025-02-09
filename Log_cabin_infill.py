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


class MyObject:
    def __init__(self, layer_height=0, layer_z=0):
        self.previous_layer = []
        self.current_layer = []
        self.previous_layer_height = 0
        self.current_layer_height = layer_height
        self.layer_z
        
    def new_layer(self, layer_z):
        self.previous_layer = self.current_layer
        self.current_layer = []
        self.previous_layer_height = self.current_layer_height
        self.current_layer_height = layer_z - self.layer_z
        self.layer_z = layer_z

    def add_line_segment(self, line_segment: LineSegment):
        self.current_layer.append(line_segment)


def intersect(segment_0: LineSegment, segment_1: LineSegment,
		overhang_threshold=0.5):
    # segment_0: in current layer
    # segment_1: in previous layer
    dx0 = segment_0.x1 - segment_0.x0
    dy0 = segment_0.y1 - segment_0.y0
    
    dx1 = segment_1.x1 - segment_1.x0
    dy1 = segment_1.y1 - segment_1.y0
    
    delta = dx1 * dy0 - dx0 * dy1
    # Lengths of both line segments
    r0 = math.sqrt(dx0 ** 2 + dy0 ** 2)
    r1 = math.sqrt(dx1 ** 2 + dy1 ** 2)

    dx = segment_1.x0 - segment_0.x0
    dy = segment_1.y0 - segment_0.y0
    
    # If both line segments are (nearly) parallel
    if abs(delta) < 1e-6:
        normal_distance = abs(dx0 * dy - dy0 * dx) / r0
        if normal_distance > segment_0.width * overhang_threshold:
            return False

        t0 = (dx0 * dx + dy0 * dy) / (r0 ** 2)

        dx = segment_1.x1 - segment_0.x0
        dy = segment_1.y1 - segment_0.y0
        t1 = (dx0 * dx + dy0 * dy) / (r0 ** 2)
        # TODO
        if 0 <= t0 < 1 and 0 <= t1 < 1:
            pass
    
    csc_angle = (r0 * r1) / abs(delta)
    half_width = 0.5 * segment_1.width * csc_angle
    
    t0 = (dx1 * dy - dy1 * dx) / delta
    t1 = (dx0 * dy - dy0 * dx) / delta
    
    if 0 <= t0 < 1 and 0 <= t1 < 1:
        return (max(0, t0 - half_width / r0), min(1, t0 + half_width / r0))
    return False

def process_g_code(input_file: str):
    internal_infill = ["Internal infill"]
    OTHER = 0
    INTERNAL_INFILL = 1

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
    
    objects[current_object] = MyObject(current_x, current_y)
    
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
                    objects[current_object].new_layer(layer_height, current_z)
                else:
                    objects[current_object] = MyObject()
            
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
            
            # Get current layer height
            match = re.search(r";Z:([\.\d]+)", line)
            if match:
                current_z = float(match.group(1))
            
            # Get current line width
            match = re.search(r";WIDTH:([\.\d]+)", line)
            if match:
                width = float(match.group(1))

            # Get current line height
            match = re.search(r";HEIGHT:([\.\d]+)", line)
            if match:
                layer_height = float(match.group(1))
            
            # Detect perimeter types from G-code comments
            match = re.search(r";TYPE:([^\n]*)", line)
            if match:
                if match.group(1) in internal_infill:
                    line_type = INTERNAL_INFILL
                else:
                    line_type = OTHER
            
            if line.startswith("G1") and ('X' in line or 'Y' in line)\
                    and 'E' in line and "E-" not in line:
                line_segment = LineSegment(previous_x, previous_y, current_x,
                        current_y, layer_height, width)
                objects[current_object].add_line_segment(line_segment)
                
                if line_type is INTERNAL_INFILL:
                    for previous_layer_line in\
                            objects[current_object].previous_layer:
                        intersection = intersect(line_segment,
                                previous_layer_line)
                        pass
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