# Autonomous Vehicle Platform - Project Notes

**Project Duration**: 2 Months  
**Date Created**: October 27, 2025  
**Platform**: CARLA Simulator + Python

---

## Project Overview

Building a complete autonomous vehicle system in simulation that can:
- Navigate city streets autonomously
- Detect and avoid obstacles
- Follow traffic rules (lights, signs, lanes)
- Handle intersections
- Park autonomously

---

## Core Architecture (4 Main Components)

### 1. PERCEPTION 👁️ - "What do I see?"
**Purpose**: Car's vision system to understand the environment

**Components**:
- Lane detection (OpenCV)
- Object detection (YOLOv8 - pre-trained)
- Traffic light/sign recognition
- Obstacle identification

**Output**: "I see 2 cars ahead, red traffic light, I'm in right lane"

---

### 2. LOCALIZATION 📍 - "Where am I?"
**Purpose**: Determine exact position and orientation

**Components**:
- GPS for position
- IMU for movement tracking (gyroscope/accelerometer)
- Sensor fusion for accuracy

**Output**: "I'm at coordinates (X,Y), heading north at 30 mph"

---

### 3. PLANNING 🧠 - "What should I do?"
**Purpose**: Decision-making and path planning

**Sub-components**:
- **Route Planning**: A* algorithm for path from A to B
- **Behavioral Planning**: Rule-based decisions (stop/go/turn)
- **Motion Planning**: Frenet frame for trajectory generation
- **Collision Avoidance**: Cost-based path selection

**Output**: "Slow down, stop at red light, then turn left when green"

---

### 4. CONTROL 🎮 - "How do I execute?"
**Purpose**: Translate decisions into vehicle commands

**Components**:
- PID controller for speed
- Pure Pursuit algorithm for steering
- Emergency braking system

**Output**: "Turn wheel 15° left, press gas 30%, no brake"

---

## Complete System Loop

```
PERCEPTION → Captures environment
    ↓
LOCALIZATION → Determines position
    ↓
PLANNING → Decides action
    ↓
CONTROL → Executes command
    ↓
Vehicle moves → Loop repeats (20+ times/second)
```

---

## 2-Month Timeline

### Week 1-2: Foundation & Perception
**Goals**:
- CARLA setup and familiarization
- Camera-based lane detection
- Object detection (vehicles, pedestrians)
- Data collection pipeline

**Deliverables**:
- Lane detection working
- YOLO detecting objects
- Recording system functional

---

### Week 3-4: Localization & Basic Control
**Goals**:
- GPS + IMU fusion
- Waypoint following
- PID controller implementation
- Basic path planning

**Deliverables**:
- Vehicle follows route A→B
- Smooth steering/speed control
- Stays in lane consistently

**MVP CHECKPOINT**: Basic autonomous driving working

---

### Week 5-6: Planning & Decision Making
**Goals**:
- Behavioral planning (traffic rules)
- Collision avoidance
- Dynamic obstacle handling
- Intersection navigation

**Deliverables**:
- Stops at red lights
- Avoids moving vehicles
- Navigates intersections
- Emergency braking

---

### Week 7: Advanced Features
**Goals**:
- Parking scenarios (parallel/perpendicular)
- Multi-scenario testing
- Edge case handling
- System integration

**Deliverables**:
- Complete urban driving
- 5+ test scenarios working
- Robust error handling

---

### Week 8: Polish & Documentation
**Goals**:
- Performance optimization
- Comprehensive documentation
- Demo video creation
- Code cleanup

**Deliverables**:
- Professional README
- Demo video (3-5 min)
- Performance metrics
- Clean codebase

---

## Technology Stack

### Core Platform
- **CARLA Simulator 0.9.15**: Professional AV simulator
- **Python 3.10**: Main programming language
- **No ROS**: Simplified architecture for faster development

### Key Libraries
- **OpenCV**: Computer vision and image processing
- **Ultralytics YOLOv8**: Pre-trained object detection
- **NumPy/SciPy**: Mathematical operations
- **Matplotlib**: Visualization and debugging

### Optional Tools
- **Pygame**: Custom UI/debugging interface
- **Docker**: Environment consistency
- **Jupyter**: Prototyping and analysis

---

## Data Sources

### All Data is FREE from CARLA! 🎉

#### 1. Real-Time Sensor Data
**Camera Images**:
- RGB cameras (color images)
- Semantic segmentation (labeled pixels)
- Depth cameras (distance info)
- 1280x720 @ 30 FPS

**GPS/IMU Data**:
- Latitude, longitude, altitude
- Acceleration, angular velocity
- Real-time positioning

**LiDAR** (optional):
- 3D point clouds
- Distance measurements

#### 2. Ground Truth Data
CARLA provides perfect labels for validation:
- All vehicle positions
- All pedestrian locations
- Traffic light states
- Exact vehicle position/rotation

**Why this matters**: Compare your AI output vs. perfect truth for debugging

#### 3. Pre-Trained Models (No Training Needed!)
**YOLOv8**:
- Pre-trained on COCO dataset (330K images)
- Detects: cars, trucks, buses, pedestrians, traffic lights, stop signs
- Download: `pip install ultralytics` (6MB, automatic)

**Lane Detection**:
- Use traditional OpenCV (no training needed)
- OR use Ultra-Fast-Lane-Detection (pre-trained)

#### 4. Maps & Routes
CARLA includes built-in maps:
- Town01-12: Various city layouts
- Highway scenarios
- Urban crossroads
- Downtown areas

**Cost Breakdown**: $0 total!

---

## Project Structure

```
autonomous-vehicle-platform/
├── src/
│   ├── perception/
│   │   ├── lane_detection.py
│   │   ├── object_detection.py
│   │   └── traffic_light_detection.py
│   ├── planning/
│   │   ├── behavioral_planner.py
│   │   ├── local_planner.py
│   │   └── path_planner.py
│   ├── control/
│   │   ├── pid_controller.py
│   │   └── vehicle_controller.py
│   ├── localization/
│   │   └── gps_imu_fusion.py
│   └── utils/
│       ├── carla_utils.py
│       └── visualization.py
├── tests/
│   └── test_scenarios.py
├── scenarios/
│   └── urban_driving.py
├── docs/
│   └── architecture.md
├── requirements.txt
└── README.md
```

---

## Team Roles (If Dividing Work)

### Option A: 4-Person Team
1. **Perception Engineer**: Lane/object/traffic light detection
2. **Planning Engineer**: Path planning, behavioral decisions
3. **Control Engineer**: Steering, speed control, vehicle dynamics
4. **Integration Lead**: Combine modules, testing, documentation

### Option B: 2-Person Team
- Person 1: Perception + Control
- Person 2: Planning + Integration

---

## Success Metrics

By end of 2 months, the system should:
- ✅ Complete 3+ different urban routes successfully
- ✅ 80%+ success rate in test scenarios
- ✅ Handle 5+ edge cases (pedestrian crossing, vehicle cutting in, etc.)
- ✅ Run in real-time (20+ FPS)
- ✅ Well-documented codebase with architecture diagrams
- ✅ Impressive demo video (3-5 minutes)

---

## Demo Scenarios to Showcase

1. **Urban Navigation**: Drive from A to B through city streets
2. **Traffic Light Compliance**: Stop at red, go on green
3. **Obstacle Avoidance**: Slow down for cars ahead, avoid pedestrians
4. **Intersection Handling**: Navigate complex intersections safely
5. **Parking**: Parallel and perpendicular parking
6. **Edge Cases**: 
   - Jaywalking pedestrian
   - Vehicle suddenly cutting in
   - Emergency vehicle (bonus)
   - Rain/fog conditions

---

## Risk Mitigation

| Risk | Solution |
|------|----------|
| CARLA setup issues | Use Docker image, pre-built binaries |
| Stuck on complex algorithms | Use proven libraries, don't reinvent wheel |
| Scope creep | Stick to core features, track with checklist |
| Integration problems | Test components independently first |
| Time constraints | Have MVP ready by Week 4 |

---

## Week 1 Action Items

### Day 1-2: Environment Setup
- [ ] Install CARLA simulator
- [ ] Verify CARLA runs properly
- [ ] Complete basic CARLA tutorials
- [ ] Set up Python environment

### Day 3-4: First Components
- [ ] Implement camera capture
- [ ] Basic lane detection working
- [ ] Test on multiple CARLA scenarios

### Day 5-7: Foundation Complete
- [ ] Integrate object detection (YOLOv8)
- [ ] First vehicle movement control
- [ ] Set up Git repository
- [ ] Initial project structure

---

## Important Implementation Notes

### What to SKIP (Save Time)
❌ Custom SLAM implementation (use CARLA ground truth)
❌ Deep reinforcement learning (too time-consuming)
❌ Custom sensor fusion (use existing libraries)
❌ Real hardware integration
❌ End-to-end neural networks (use modular approach)

### What to FOCUS ON
✅ Working end-to-end system quickly
✅ Robust core scenario performance
✅ Clean, modular code architecture
✅ Excellent documentation
✅ Impressive demo video

---

## Installation Quick Start

```bash
# 1. Install CARLA
# Download from: https://github.com/carla-simulator/carla/releases
# OR use: pip install carla

# 2. Install Python dependencies
pip install ultralytics opencv-python numpy scipy matplotlib

# 3. First test
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

# 4. Clone starter code (if available)
git clone <repository-url>
cd autonomous-vehicle-platform
```

---

## Key Algorithms Reference

### Lane Detection
- **Canny Edge Detection**: Find edges in image
- **Hough Transform**: Detect lines
- **ROI Masking**: Focus on road area

### Object Detection
- **YOLOv8**: Real-time object detection
- **Non-Max Suppression**: Remove duplicate detections

### Planning
- **A* Algorithm**: Find shortest path
- **Frenet Coordinates**: Path planning in road-relative frame
- **Cost Functions**: Evaluate path safety/efficiency

### Control
- **PID Controller**: Proportional-Integral-Derivative control
- **Pure Pursuit**: Follow path with look-ahead point
- **MPC** (optional): Model Predictive Control for optimization

---

## Resources & References

### Documentation
- CARLA Docs: https://carla.readthedocs.io/
- YOLOv8 Docs: https://docs.ultralytics.com/
- OpenCV Tutorials: https://docs.opencv.org/

### Learning Materials
- Udacity Self-Driving Car Nanodegree
- Apollo/Baidu autonomous driving courses
- "Probabilistic Robotics" by Thrun
- "Planning Algorithms" by LaValle

### Research Papers
- Follow CVPR, ICCV, ICRA, IROS conferences
- Check arXiv for latest AV research

---

## 30-Second Elevator Pitch

"We're building an autonomous driving system that navigates city streets in a realistic simulator. The car uses cameras to see, AI to understand its environment, smart algorithms to plan safe routes, and control systems to drive. In 2 months, we'll demo a car that drives itself through complex scenarios - stopping at lights, avoiding obstacles, and parking - all while following traffic rules."

---

## Why This Project Stands Out

✅ **Comprehensive**: Multiple CS domains (AI, vision, robotics, control)
✅ **Relevant**: Hot industry topic with real-world applications
✅ **Impressive**: Visual demo that's easy to understand
✅ **Educational**: Learn cutting-edge technologies
✅ **Portfolio-worthy**: Excellent for resumes and interviews
✅ **Achievable**: Clear 2-month roadmap with checkpoints

---

## Notes & Updates

### Week 1 Progress
- [ ] Setup completed
- [ ] First component working
- [ ] Issues encountered:

### Week 2 Progress
- [ ] Perception pipeline complete
- [ ] Challenges:

### Week 3 Progress
- [ ] Basic control implemented
- [ ] Notes:

### Week 4 Progress - MVP CHECKPOINT
- [ ] End-to-end system working
- [ ] Demo ready:

(Continue for remaining weeks...)

---

## Contact & Collaboration

**Team Members**:
- Member 1: [Role]
- Member 2: [Role]
- Member 3: [Role]
- Member 4: [Role]

**Meeting Schedule**: [Add times]

**Communication**: [Slack/Discord/Email]

**Repository**: [GitHub URL]

---

## Final Checklist (Week 8)

### Documentation
- [ ] README with setup instructions
- [ ] Architecture diagrams
- [ ] API documentation
- [ ] User guide

### Code Quality
- [ ] Clean, commented code
- [ ] Consistent style
- [ ] Error handling
- [ ] Performance optimized

### Testing
- [ ] All scenarios pass
- [ ] Edge cases handled
- [ ] Performance metrics recorded

### Demo
- [ ] Video recorded (3-5 min)
- [ ] Screenshots captured
- [ ] Presentation slides ready

### Submission
- [ ] All files committed
- [ ] Repository organized
- [ ] Documentation complete
- [ ] Ready to present!

---

**Last Updated**: October 27, 2025
**Status**: Planning Phase
**Next Milestone**: Week 1 Setup Complete