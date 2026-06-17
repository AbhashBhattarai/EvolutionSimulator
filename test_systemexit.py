import mujoco, mujoco.viewer

def run():
    try:
        mujoco.viewer.launch_passive(mujoco.MjModel.from_xml_string('<mujoco></mujoco>'), mujoco.MjData(mujoco.MjModel.from_xml_string('<mujoco></mujoco>')))
    except Exception as e:
        print("Exception")
    except SystemExit:
        print("SystemExit")
run()
print("Alive")
