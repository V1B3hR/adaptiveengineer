# Swarm Robotics Capabilities

## Overview

The Adaptive Engineer swarm robotics system provides distributed coordination algorithms for physical autonomous robots. It implements bio-inspired behaviors (flocking, foraging) with realistic physical constraints (battery, communication range, movement costs).

## Architecture

### Core Component: SwarmRoboticsPlugin (`plugins/swarm_robotics.py`)

The plugin extends AliveLoopNode agents with swarm coordination capabilities:

#### Formation Types
- **CIRCLE**: Agents arrange in circular formation
- **LINE**: Linear formation for convoy/patrol
- **WEDGE**: V-shaped formation for navigation
- **GRID**: Grid pattern for area coverage

#### Swarm Behaviors
- **FLOCKING**: Cohesion, separation, alignment (boids algorithm)
- **FORAGING**: Distributed resource collection
- **PERIMETER_DEFENSE**: Coordinate to protect area
- **EMERGENCY_REGROUP**: Respond to threats collectively

## Physical Constraints

The system respects realistic robot limitations:

### Energy Model
- Battery capacity mapped from simulation energy
- Movement costs proportional to velocity
- Low battery triggers hardware failure detection
- Energy-aware task allocation

### Communication Range
- Agents can only coordinate with neighbors within perception radius
- Communication limited by `communication_range` parameter
- Distributed algorithms work with local information only

### Movement Costs
- Acceleration and velocity consume energy
- Higher speeds = higher energy consumption
- Formation maintenance balanced with energy conservation

### Sensor Limitations
- Perception radius defines sensor range
- Obstacle detection within `obstacle_avoidance_range`
- No global positioning assumed (relative positioning only)

## Algorithms

### 1. Flocking (Boids Algorithm)

Three behavioral rules combine to produce emergent flocking:

**Cohesion**: Move toward center of nearby agents
```python
center = mean([neighbor.position for neighbor in neighbors])
cohesion_force = center - agent.position
```

**Separation**: Avoid crowding neighbors
```python
for neighbor in neighbors:
    if distance < desired_spacing:
        separation_force += (agent.position - neighbor.position) / distance
```

**Alignment**: Match velocity with neighbors
```python
avg_velocity = mean([neighbor.velocity for neighbor in neighbors])
alignment_force = avg_velocity - agent.velocity
```

**Combined**:
```python
desired_velocity = (
    cohesion_weight * cohesion +
    separation_weight * separation +
    alignment_weight * alignment
)
```

### 2. Energy-Aware Task Allocation

Distributes tasks based on proximity and available energy:

```python
def allocate_tasks(agents, tasks):
    for task in sorted(tasks, key=lambda t: t.priority, reverse=True):
        best_agent = None
        best_score = -inf
        
        for agent in agents:
            if agent.energy < task.energy_requirement:
                continue  # Not enough energy
            
            distance = norm(agent.position - task.position)
            energy_ratio = agent.energy / agent.max_energy
            
            # Prefer closer agents with more energy
            score = energy_ratio * 2.0 - distance * 0.5
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        if best_agent:
            assign(task, best_agent)
```

### 3. Self-Healing Formations

When agents fail, the swarm adapts:

1. **Failure Detection**: Monitor energy levels and communication
2. **Failure Notification**: Failed agents identified via hardware monitoring
3. **Formation Adjustment**: Remaining agents redistribute positions
4. **Continuous Operation**: Swarm continues with reduced capacity

### 4. Obstacle Avoidance

Potential field approach for collision avoidance:

```python
def avoid_obstacles(agent, obstacles):
    avoidance = zeros_like(agent.velocity)
    
    for obstacle in obstacles:
        distance = norm(agent.position - obstacle.position)
        if distance < obstacle_avoidance_range:
            # Repulsive force, stronger when closer
            direction = (agent.position - obstacle.position) / distance
            strength = (obstacle_avoidance_range - distance) / obstacle_avoidance_range
            avoidance += direction * strength
    
    return obstacle_avoidance_weight * avoidance
```

## Physical Robot Integration

### Hardware Requirements

**Minimum**:
- 2D locomotion (wheeled, tracked, or legged)
- Distance sensors (ultrasonic, IR, or LiDAR)
- Wireless communication (WiFi, Zigbee, or LoRa)
- Rechargeable battery with voltage monitoring

**Recommended**:
- IMU for orientation tracking
- GPS for absolute positioning (outdoor)
- Camera for vision-based coordination
- Edge compute (Raspberry Pi or similar)

### Integration Example

```python
from adaptiveengineer import AliveLoopNode
from plugins.swarm_robotics import SwarmRoboticsPlugin, FormationType, SwarmBehavior

# Initialize robot agent
robot = AliveLoopNode(
    position=(x, y),  # From GPS or SLAM
    velocity=(vx, vy),  # From odometry
    initial_energy=battery_voltage_to_energy(voltage),
    node_id=robot_id
)

# Setup swarm plugin
swarm = SwarmRoboticsPlugin(
    formation_type=FormationType.CIRCLE,
    behavior_mode=SwarmBehavior.FLOCKING,
    perception_radius=sensor_range_meters
)
swarm.initialize(robot)

# Control loop
while True:
    # Sense
    neighbors = discover_neighbors(communication_range)
    obstacles = detect_obstacles(sensor_range)
    
    # Think
    swarm.update(robot, delta_time=0.1)
    
    # Act
    motor_left, motor_right = velocity_to_motor_commands(robot.velocity)
    set_motors(motor_left, motor_right)
    
    # Update energy from battery
    voltage = read_battery_voltage()
    robot.energy = battery_voltage_to_energy(voltage)
    
    time.sleep(0.1)
```

### Battery Mapping

Map battery voltage to simulation energy:

```python
def battery_voltage_to_energy(voltage, v_min=3.0, v_max=4.2, energy_max=10.0):
    """Map LiPo battery voltage to simulation energy"""
    voltage_range = v_max - v_min
    voltage_normalized = (voltage - v_min) / voltage_range
    return max(0.0, min(energy_max, voltage_normalized * energy_max))
```

### Communication Range

Determine perception radius based on wireless characteristics:

```python
def calculate_perception_radius(rssi_threshold=-80):
    """
    Calculate effective communication range from RSSI.
    
    Args:
        rssi_threshold: Minimum RSSI for reliable communication (dBm)
    
    Returns:
        Perception radius in meters
    """
    # Free space path loss: RSSI = Pt - 20*log10(d) - 20*log10(f) - 32.44
    # Approximate for 2.4GHz WiFi
    Pt = 20  # Transmit power (dBm)
    frequency_ghz = 2.4
    
    # Solve for distance
    path_loss = Pt - rssi_threshold
    distance = 10 ** ((path_loss - 20*math.log10(frequency_ghz) - 32.44) / 20)
    
    return distance
```

## Deployment Examples

### Example 1: Warehouse Robots

```python
# Initialize swarm for warehouse patrol
num_robots = 10
robots = []

for i in range(num_robots):
    robot = AliveLoopNode(
        position=(random.uniform(0, 50), random.uniform(0, 50)),
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=i
    )
    robots.append(robot)

# Configure for foraging behavior
swarm = SwarmRoboticsPlugin(
    formation_type=FormationType.GRID,
    behavior_mode=SwarmBehavior.FORAGING,
    perception_radius=5.0
)

# Add collection tasks
for shelf_location in warehouse_shelves:
    task = SwarmTask(
        task_id=f"collect_{shelf_location}",
        task_type="pick_item",
        energy_requirement=0.5,
        priority=0.8,
        position=shelf_location
    )
    swarm.add_task(task)

# Allocate and execute
allocations = swarm.allocate_tasks(robots)
```

### Example 2: Agricultural Drones

```python
# Initialize drone swarm for crop monitoring
drones = []
for i in range(6):
    drone = AliveLoopNode(
        position=(field_x + i*10, field_y),
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=i
    )
    drones.append(drone)

# Line formation for sweep coverage
swarm = SwarmRoboticsPlugin(
    formation_type=FormationType.LINE,
    behavior_mode=SwarmBehavior.FLOCKING,
    perception_radius=15.0  # Longer range for drones
)

# Sweep across field
swarm.set_swarm_center(field_center)
```

### Example 3: Search and Rescue

```python
# Initialize rescue robots
rescue_team = []
for i in range(8):
    robot = AliveLoopNode(
        position=base_position,
        velocity=(0, 0),
        initial_energy=10.0,
        node_id=i
    )
    rescue_team.append(robot)

# Circle formation for area search
swarm = SwarmRoboticsPlugin(
    formation_type=FormationType.CIRCLE,
    behavior_mode=SwarmBehavior.PERIMETER_DEFENSE,
    perception_radius=8.0
)

# Expand search radius over time
for t in range(simulation_steps):
    search_radius = initial_radius + t * expansion_rate
    swarm.set_swarm_center(search_center)
    # Update robots...
```

## Performance Metrics

### Formation Quality
- **Coherence**: How well formation is maintained (0-1)
- **Convergence Time**: Time to reach desired formation
- **Formation Error**: RMS distance from ideal positions

### Task Efficiency
- **Completion Rate**: Percentage of tasks completed
- **Energy Efficiency**: Tasks per unit energy consumed
- **Load Balance**: Variance in tasks per robot

### Failure Resilience
- **Recovery Time**: Time to reform after agent failure
- **Degradation**: Performance loss per failed agent
- **Failure Detection Latency**: Time to detect failures

## Demonstration

See `example/example_swarm_robotics.py` for complete demonstration:

```bash
python3 example/example_swarm_robotics.py
```

Expected output:
```
🤖 SWARM ROBOTICS DEMONSTRATION 🤖

Phase 1: Formation Control
✓ Agents forming circular formation...

Phase 2: Energy-Aware Task Allocation
✓ Switched to FORAGING mode
✓ All tasks completed at step 23!

Phase 3: Self-Healing Formation
⚠️  Simulating failure of Agent 2...
✓ Activated EMERGENCY_REGROUP behavior

📊 Swarm Performance:
   Completion Rate: 100.0%
   Hardware Failures: 1
```

## Best Practices

1. **Energy Budgeting**: Always maintain 20% battery reserve
2. **Communication Testing**: Verify range in deployment environment
3. **Formation Tuning**: Adjust weights based on robot dynamics
4. **Failure Redundancy**: Plan for 20% failure rate
5. **Sensor Calibration**: Regular calibration of distance sensors
6. **Safety Margins**: Separation distance > robot diameter + safety margin
7. **Graceful Degradation**: Test with progressively fewer agents

## Hardware Failure Handling

The system automatically detects and handles:

### Low Battery
```python
if robot.energy < low_battery_threshold:
    # Detected automatically
    # Robot excluded from task allocation
    # Formation adapts without this robot
```

### Communication Failure
```python
if len(robot.signal_history) == 0:
    # No recent communication
    # Marked as failed
    # Other robots continue
```

### Recovery
```python
if robot.energy > normal_threshold:
    # Removed from failure list
    # Reintegrated into formation
```

## Limitations

1. **2D Assumption**: Current implementation assumes planar movement
2. **Simple Dynamics**: Does not model acceleration limits or slip
3. **No Global Positioning**: Requires relative positioning system
4. **Homogeneous Agents**: Assumes similar capabilities
5. **Static Obstacles**: Dynamic obstacle avoidance not fully implemented

## Future Enhancements

- 3D formations for aerial/underwater swarms
- Heterogeneous agent support (different capabilities)
- Dynamic obstacle avoidance with prediction
- Vision-based coordination
- Multi-swarm coordination
- ROS integration for direct hardware control

## Hybrid Defense Swarm

For combining swarm robotics with cyber-defense, see `example/example_hybrid_defense_swarm.py`:

- Mobile defense agents patrol infrastructure
- Swarm repositions based on threat intelligence
- Coordinated response to distributed attacks
- Self-healing under attack

## References

- Boids Algorithm: Reynolds (1987)
- Swarm Robotics: Şahin (2005)
- Multi-Robot Task Allocation: Gerkey & Matarić (2004)
- Bio-Inspired Robotics: Bar-Cohen & Breazeal (2003)

## Support

For issues and questions:
- GitHub Issues: https://github.com/V1B3hR/adaptiveengineer/issues
- Examples: `example/example_swarm_robotics.py`
- Tests: `tests/test_swarm_robotics.py`
