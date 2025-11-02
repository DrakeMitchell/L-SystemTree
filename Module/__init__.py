# for now, put this at the top of every file created
import sys
import os
import bpy

blend_dir = os.path.dirname(bpy.data.filepath)
module_path = os.path.join(blend_dir, "Module")

if module_path not in sys.path:
    sys.path.append(module_path)
