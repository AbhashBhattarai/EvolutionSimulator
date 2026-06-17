import time
import math
import mujoco
import mujoco.viewer

# 1. THE GENOME STRUCTURE
# A Python list of dictionaries representing a 'Genome'.
# Each item in the list represents a body segment.
genome = [
    {"length": 0.4, "radius": 0.08, "joint_axis": "0 1 0", "color": "0.8 0.2 0.2 1"},
    {"length": 0.3, "radius": 0.06, "joint_axis": "0 1 0", "color": "0.2 0.8 0.2 1"},
    {"length": 0.3, "radius": 0.05, "joint_axis": "1 0 0", "color": "0.2 0.2 0.8 1"},
    {"length": 0.2, "radius": 0.04, "joint_axis": "0 1 0", "color": "0.8 0.8 0.2 1"}
]

# 2. PROCEDURAL XML COMPILER
def generate_creature_mjcf(genome_specs):
    """
    Takes a genome and strings together a valid MuJoCo MJCF XML configuration.
    """
    xml = [
        '<mujoco model="procedural_creature">',
        '  <option gravity="0 0 -9.81"/>',
        '  <worldbody>',
        '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
        '    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>'
    ]
    
    actuators = []
    
    # Calculate starting height so the creature doesn't spawn stuck in the floor
    z_offset = sum(item["length"] for item in genome_specs) + 0.5

    # Loop through the genome to procedurally nest child <body> tags inside parent body tags
    for i, seg in enumerate(genome_specs):
        indent = "  " * (i + 2)
        # Position: The root is at z_offset. Child segments start at the end of the previous segment.
        pos = f"0 0 {z_offset}" if i == 0 else f"0 0 {-genome_specs[i-1]['length']}"
        
        xml.append(f'{indent}<body name="segment_{i}" pos="{pos}">')
        
        if i == 0:
            # The creature's root body should use a <freejoint/>
            xml.append(f'{indent}  <freejoint name="root"/>')
        else:
            # 1-DOF hinge joints based on the genome specs
            joint_name = f"joint_{i}"
            xml.append(f'{indent}  <joint name="{joint_name}" type="hinge" axis="{seg["joint_axis"]}"/>')
            # For every joint generated, add a corresponding torque-controlled <motor> actuator
            actuators.append(f'<motor name="motor_{i}" joint="{joint_name}" ctrlrange="-1 1" gear="20"/>')
            
        # Generate 3D capsule geoms
        xml.append(f'{indent}  <geom type="capsule" fromto="0 0 0 0 0 {-seg["length"]}" size="{seg["radius"]}" rgba="{seg["color"]}"/>')
        
    # Close all nested body tags
    for i in reversed(range(len(genome_specs))):
        indent = "  " * (i + 2)
        xml.append(f'{indent}</body>')
        
    xml.append('  </worldbody>')
    
    # Append actuators
    if actuators:
        xml.append('  <actuator>')
        for act in actuators:
            xml.append(f'    {act}')
        xml.append('  </actuator>')
        
    xml.append('</mujoco>')
    
    return '\n'.join(xml)


def main():
    # Generate the XML
    mjcf_string = generate_creature_mjcf(genome)
    print("Generated MJCF XML:")
    print(mjcf_string)
    
    # 3. THE RUNTIME GAME LOOP
    # Compile the XML string dynamically
    model = mujoco.MjModel.from_xml_string(mjcf_string)
    
    # Initialize the physics data
    data = mujoco.MjData(model)
    
    # Launch a passive 3D interactive rendering window
    print("\nLaunching MuJoCo viewer...")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Run the simulation loop
            while viewer.is_running():
                step_start = time.time()
                
                # Primitive 'brain' script:
                # Apply a simple oscillating sine wave with slightly offset phases to each motor actuator
                t = data.time
                for i in range(model.nu):
                    data.ctrl[i] = math.sin(t * 5.0 + i)
                    
                # Step the physics
                mujoco.mj_step(model, data)
                
                # Synchronize the frame with the viewer
                viewer.sync()
                
                # Maintain roughly realtime simulation speed
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        # If running in a headless environment without a display, this might fail.
        print("Viewer exited or failed to launch:", e)
        print("Running headless simulation for 1000 steps instead...")
        for _ in range(1000):
            t = data.time
            for i in range(model.nu):
                data.ctrl[i] = math.sin(t * 5.0 + i)
            mujoco.mj_step(model, data)
        print("Headless simulation complete.")

if __name__ == "__main__":
    main()