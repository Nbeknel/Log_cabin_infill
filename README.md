# Functionality

This script finds all segments in sparse infill that are not supported
by the previous layer. For these segments the script increase the amount
of filament to extrude so that it would fill the gap in the previous
layer. These segments are printed at a lower speed so as to maintain the
current flow. In theory, this should allow for bonding between the
current layer and the layer before the previous.

## Why not just increase infill width by a factor of two and decrease print speed by the same factor?

This script tries to maintain a constant line width, be it on top of
solid infill, internal infill, or when variable layer height is used.
Increasing the infill width will result in varying width in the print.
When using rectilinear infill, you might encounter a similar problem as
when printing grid or triangle infill, the infill might start hitting
the nozzle.

# Parameters

`--minimum_length`\
The script will not try to extrude into gaps smaller than this distance.
If expressed as a percentage it is calculated over the nozzle diameter.\
Default value: 50%.

`--overhang_threshold`\
The script will not try to increase extrusion when a line or subsegment
is supported by at least this distance. If expressed as a percentage it
is calculated over the nozzle diameter.\
Default value: 50%.

`--slowdown_speed`\
When set to zero, the script will slow down infill speed to maintain
a constant volumetric flow. This will result in jittering movements when
printing infill. With this setting you can reduce the print speed of the
fast segments of infill, since those will be on the smaller side, and
the target speed most likely won't be reached. A value between 0 and 1
will linearly interpolate between the default infill speed and the
reduced infill speed, with 0 - default speed and 1 - reduced infill
speed. Can be expressed as a percentage (50% = 0.5). If set to -1, the
script will create a dynamic slowdown loosely based on Klipper's
ACCEL_TO_DECEL algorithm.\
Default value: 0.

#
Currently supports only single extruder printers, or it will use the
nozzle diameter of the first extruder.\
Tested in SuperSlicer 2.5.59.13 and PrusaSlicer 2.9.0. For some reason
it won't work with OrcaSlicer, but since OrcaSlicer doesn't provide a
slicer with a console, I can't read out the error.
