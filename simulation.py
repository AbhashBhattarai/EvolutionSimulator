import time
import math
import copy
import random
import mujoco
import mujoco.viewer

# 1. POPULATION AND GENOME STRUCTURE
# Maintain a population of 5 different individuals.
# Expand the Genome to include Morphological Traits and Control Traits.
base_genome = [
    {"length": 0.4, "radius": 0.08, "joint_axis": "0 1 0", "color": "0.8 0.2 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 0.0},
    {"length": 0.3, "radius": 0.06, "joint_axis": "0 1 0", "color": "0.2 0.8 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 1.0},
    {"length": 0.3, "radius": 0.05, "joint_axis": "1 0 0", "color": "0.2 0.2 0.8 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 2.0},
    {"length": 0.2, "radius": 0.04, "joint_axis": "0 1 0", "color": "0.8 0.8 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 3.0}
]

def mutate_genome(genome):
    """
    Clones the genome and applies small random mathematical variance (mutation) to its traits.
    """
    new_genome = copy.deepcopy(genome)
    for seg in new_genome:
        # Mutate morphological traits
        seg["length"] = max(0.05, seg["length"] + random.uniform(-0.05, 0.05))
        seg["radius"] = max(0.01, seg["radius"] + random.uniform(-0.01, 0.01))
        # Mutate control traits
        seg["amplitude"] = max(0.0, seg["amplitude"] + random.uniform(-0.2, 0.2))
        seg["frequency"] = max(0.1, seg["frequency"] + random.uniform(-1.0, 1.0))
        seg["phase_offset"] += random.uniform(-0.5, 0.5)
    return new_genome


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


def evaluate_creature(genome):
    """
    Compiles the creature's XML, resets physics, and runs the simulation loop for a fixed window.
    Calculates fitness as the absolute distance traveled along the X-axis.
    """
    mjcf_string = generate_creature_mjcf(genome)
    model = mujoco.MjModel.from_xml_string(mjcf_string)
    data = mujoco.MjData(model)
    
    steps = 400  # fixed simulation window (400 physics steps)
    
    try:
        # Launch passive viewer for the evaluation
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for step in range(steps):
                step_start = time.time()
                t = data.time
                
                # Apply brain control
                for i in range(model.nu):
                    seg = genome[i + 1] # motor_i corresponds to segment_i+1
                    data.ctrl[i] = seg["amplitude"] * math.sin(seg["frequency"] * t + seg["phase_offset"])
                    
                mujoco.mj_step(model, data)
                viewer.sync()
                
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except Exception as e:
        # Headless fallback if viewer fails to launch (e.g., in CI environments)
        for step in range(steps):
            t = data.time
            for i in range(model.nu):
                seg = genome[i + 1]
                data.ctrl[i] = seg["amplitude"] * math.sin(seg["frequency"] * t + seg["phase_offset"])
            mujoco.mj_step(model, data)

    # Calculate absolute distance traveled along the X-axis from start
    # data.qpos[0] is the X position of the root freejoint
    fitness = abs(data.qpos[0])
    return fitness


def main():
    population_size = 5
    
    # Initialize Population
    population = [copy.deepcopy(base_genome)]
    for _ in range(population_size - 1):
        population.append(mutate_genome(base_genome))
        
    generation = 1
    best_distance_ever = 0.0
    
    # 4. CONTINUOUS ENGINE: Infinite loop
    while True:
        print(f"\n{'='*40}")
        print(f"--- STARTING GENERATION {generation} ---")
        print(f"{'='*40}")
        
        fitness_scores = []
        
        # 2. SEQUENTIAL FITNESS EVALUATION
        for i, genome in enumerate(population):
            print(f"[Gen {generation}] Evaluating Creature {i} (Current Best: {best_distance_ever:.4f})...")
            fitness = evaluate_creature(genome)
            fitness_scores.append((fitness, genome, i))
            
        # 3. SELECTION AND MUTATION
        # Sort descending by fitness (Distance Traveled)
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        print("\n--- GENERATION SUMMARY ---")
        for rank, (fit, _, cid) in enumerate(fitness_scores):
            print(f"Rank {rank+1}: Creature {cid} | Distance: {fit:.4f}")
            
        elite_fitness, elite_genome, elite_id = fitness_scores[0]
        
        if elite_fitness > best_distance_ever:
            best_distance_ever = elite_fitness
            print(f">>> NEW BEST DISTANCE: {best_distance_ever:.4f} <<<")
            
        # Reproduction
        new_population = [copy.deepcopy(elite_genome)] # Save the elite
        for _ in range(population_size - 1):
            new_population.append(mutate_genome(elite_genome))
            
        population = new_population
        generation += 1

if __name__ == "__main__":
    main()
