import os, mujoco, mujoco.viewer
m = mujoco.MjModel.from_xml_string('<mujoco></mujoco>')
d = mujoco.MjData(m)
has_display = 'DISPLAY' in os.environ
print("Has display:", has_display)
