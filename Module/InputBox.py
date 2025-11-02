import sys
import os
import bpy

blend_dir = os.path.dirname(bpy.data.filepath)

if blend_dir not in sys.path:
    sys.path.append(blend_dir)

from Module import TreeLSys

#TreeLSys.Tree("F[+F]/F[-F]*F^F!F[+F[+F]/F[-F]*F^F!F]/F[+F]/F[-F]*F^F!F[-F[+F]/F[-F]*F^F!F]*F^F!F[+F]/F[-F]*F^F!F[+F[+F]/F[-F]*F^F!F]/F[+F]/F[-F]*F^F!F[-F[+F]/F[-F]*F^F!F]*F^F!F[-F[+F]/F[-F]*F^F!F]/F[+F]/F[-F]*F^F!F[+F[+F]/F[-F]*F^F!F]/F[+F]/F[-F]*F^F!F[-F[+F]/F[-F]*F^F!F]*F^")


class L_SystemPopup(bpy.types.Operator):
    bl_idname = "wm.simple_popup"
    bl_label = "L System Input"

    my_string: bpy.props.StringProperty(name="Enter Text")

    def execute(self, context):
        TreeLSys.Tree(self.my_string)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class clearScreen(bpy.types.Operator):

    bl_idname = "wm.clearscreen"
    bl_label = "Clear Screen"
    
    
    def execute(self, context):
        collection = bpy.data.collections["Collection"]
        for obj in collection.all_objects:
            obj.select_set(True)
        bpy.ops.object.delete(use_global=False,confirm=False)
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

class L_SystemPanel(bpy.types.Panel):
    bl_label = "L-System"
    bl_idname = "VIEW3D_PT_simple_popup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = 'L-System'

    def draw(self, context):
        layout = self.layout
        layout.operator("wm.simple_popup")
        layout.operator("wm.clearscreen")

bpy.utils.register_class(L_SystemPopup)
bpy.utils.register_class(L_SystemPanel)
bpy.utils.register_class(clearScreen)
bpy.ops.wm.clearscreen("INVOKE_DEFAULT")
bpy.ops.VIEW3D_PT_simple_popup("INVOKE_DEFAULT")
bpy.ops.wm.simple_popup("INVOKE_DEFAULT")

