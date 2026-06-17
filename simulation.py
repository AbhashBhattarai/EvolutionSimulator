import os
import time
import math
import copy
import random
import mujoco
import mujoco.viewer
import numpy as np

# 1. POPULATION AND GENOME STRUCTURE
base_genome = [
    {"length": 0.4, "radius": 0.08, "joint_axis": "0 1 0", "color": "0.8 0.2 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 0.0},
    {"length": 0.3, "radius": 0.06, "joint_axis": "0 1 0", "color": "0.2 0.8 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 1.0},
    {"length": 0.3, "radius": 0.05, "joint_axis": "1 0 0", "color": "0.2 0.2 0.8 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 2.0},
    {"length": 0.2, "radius": 0.04, "joint_axis": "0 1 0", "color": "0.8 0.8 0.2 1", "amplitude": 1.0, "frequency": 5.0, "phase_offset": 3.0}
]

def mutate_genome(genome):
    new_genome = copy.deepcopy(genome)
    for seg in new_genome:
        seg["length"] = max(0.05, seg["length"] + random.uniform(-0.05, 0.05))
        seg["radius"] = max(0.01, seg["radius"] + random.uniform(-0.01, 0.01))
        seg["amplitude"] = max(0.0, seg["amplitude"] + random.uniform(-0.2, 0.2))
        seg["frequency"] = max(0.1, seg["frequency"] + random.uniform(-1.0, 1.0))
        seg["phase_offset"] += random.uniform(-0.5, 0.5)
    return new_genome

# 2. PROCEDURAL XML COMPILER (Symmetrical and Horizontal)
def generate_creature_mjcf(genome_specs):
    xml = [
        '<mujoco model="procedural_creature">',
        '  <option gravity="0 0 -9.81"/>',
        '  <worldbody>',
        '    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>',
        '    <geom type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>'
    ]
    
    actuators = []
    
    root_seg = genome_specs[0]
    z_offset = root_seg["radius"] + 0.1 
    
    xml.append(f'    <body name="root_body" pos="0 0 {z_offset}">')
    xml.append(f'      <freejoint name="root"/>')
    hx = root_seg["length"] / 2.0
    xml.append(f'      <geom type="capsule" fromto="-{hx} 0 0 {hx} 0 0" size="{root_seg["radius"]}" rgba="{root_seg["color"]}"/>')
    
    indent = "      "
    xml_left = []
    for i in range(1, len(genome_specs)):
        seg = genome_specs[i]
        pos_y = root_seg["radius"] if i == 1 else genome_specs[i-1]["length"]
        xml_left.append(f'{indent}<body name="segment_L_{i}" pos="0 {pos_y} 0">')
        joint_name = f"joint_L_{i}"
        xml_left.append(f'{indent}  <joint name="{joint_name}" type="hinge" axis="{seg["joint_axis"]}"/>')
        xml_left.append(f'{indent}  <geom type="capsule" fromto="0 0 0 0 {seg["length"]} 0" size="{seg["radius"]}" rgba="{seg["color"]}"/>')
        actuators.append(f'<motor name="motor_L_{i}" joint="{joint_name}" ctrlrange="-1 1" gear="20"/>')
        indent += "  "
    for i in range(1, len(genome_specs)):
        indent = indent[:-2]
        xml_left.append(f'{indent}</body>')
        
    indent = "      "
    xml_right = []
    for i in range(1, len(genome_specs)):
        seg = genome_specs[i]
        pos_y = -root_seg["radius"] if i == 1 else -genome_specs[i-1]["length"]
        xml_right.append(f'{indent}<body name="segment_R_{i}" pos="0 {pos_y} 0">')
        joint_name = f"joint_R_{i}"
        xml_right.append(f'{indent}  <joint name="{joint_name}" type="hinge" axis="{seg["joint_axis"]}"/>')
        xml_right.append(f'{indent}  <geom type="capsule" fromto="0 0 0 0 {-seg["length"]} 0" size="{seg["radius"]}" rgba="{seg["color"]}"/>')
        actuators.append(f'<motor name="motor_R_{i}" joint="{joint_name}" ctrlrange="-1 1" gear="20"/>')
        indent += "  "
    for i in range(1, len(genome_specs)):
        indent = indent[:-2]
        xml_right.append(f'{indent}</body>')
        
    xml.extend(xml_left)
    xml.extend(xml_right)
    xml.append('    </body>')
    xml.append('  </worldbody>')
    
    if actuators:
        xml.append('  <actuator>')
        for act in actuators:
            xml.append(f'    {act}')
        xml.append('  </actuator>')
        
    xml.append('</mujoco>')
    return '\n'.join(xml)


def evaluate_creature(genome):
    mjcf_string = generate_creature_mjcf(genome)
    model = mujoco.MjModel.from_xml_string(mjcf_string)
    data = mujoco.MjData(model)
    
    steps = 400
    num_segments = len(genome) - 1
    exploded = False
    
    has_display = 'DISPLAY' in os.environ
    
    if has_display:
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                for step in range(steps):
                    step_start = time.time()
                    t = data.time
                    for i in range(num_segments):
                        seg = genome[i + 1]
                        ctrl_val = seg["amplitude"] * math.sin(seg["frequency"] * t + seg["phase_offset"])
                        data.ctrl[i] = ctrl_val
                        data.ctrl[i + num_segments] = ctrl_val
                        
                    mujoco.mj_step(model, data)
                    if np.isnan(data.qacc).any():
                        exploded = True
                        break
                    viewer.sync()
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except Exception:
            has_display = False

    if not has_display:
        for step in range(steps):
            t = data.time
            for i in range(num_segments):
                seg = genome[i + 1]
                ctrl_val = seg["amplitude"] * math.sin(seg["frequency"] * t + seg["phase_offset"])
                data.ctrl[i] = ctrl_val
                data.ctrl[i + num_segments] = ctrl_val
                
            mujoco.mj_step(model, data)
            if np.isnan(data.qacc).any():
                exploded = True
                break

    if exploded:
        return -999.0

    return abs(data.qpos[0])


def main():
    population_size = 5
    population = [copy.deepcopy(base_genome)]
    for _ in range(population_size - 1):
        population.append(mutate_genome(base_genome))
        
    generation = 1
    best_distance_ever = 0.0
    last_avg_fitness = None
    
    while True:
        fitness_scores = []
        
        for i, genome in enumerate(population):
            fitness = evaluate_creature(genome)
            fitness_scores.append((fitness, genome, i))
            
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        avg_fitness = sum(f[0] for f in fitness_scores) / population_size
        
        print("\n" + "="*50)
        print(f"       GENERATION {generation} SCOREBOARD")
        print("="*50)
        print(f"{'Rank':<6} | {'ID':<4} | {'Status':<10} | {'Fitness':<10}")
        print("-" * 50)
        
        for rank, (fit, _, cid) in enumerate(fitness_scores):
            if fit == -999.0:
                status = "DIED ☠️"
            elif rank == 0:
                status = "ELITE 👑"
            else:
                status = "SURVIVED"
            
            print(f"{rank+1:<6} | {cid:<4} | {status:<10} | {fit:.4f}")
            
        print("-" * 50)
        
        avg_diff_str = ""
        if last_avg_fitness is not None:
            diff = avg_fitness - last_avg_fitness
            sign = "+" if diff >= 0 else ""
            avg_diff_str = f" ({sign}{diff:.4f})"
            
        print(f"Average Fitness : {avg_fitness:.4f}{avg_diff_str}")
        
        elite_fitness, elite_genome, elite_id = fitness_scores[0]
        
        if elite_fitness > best_distance_ever and elite_fitness != -999.0:
            best_distance_ever = elite_fitness
            print(f"★ NEW ALL-TIME BEST: {best_distance_ever:.4f} ★")
            
        print("="*50 + "\n")
        
        last_avg_fitness = avg_fitness
        
        new_population = [copy.deepcopy(elite_genome)]
        for _ in range(population_size - 1):
            new_population.append(mutate_genome(elite_genome))
            
        population = new_population
        generation += 1

        if generation > 3:
             print("\nReached gen 3, breaking for headless CI safety.")
             break

if __name__ == "__main__":
    main()
