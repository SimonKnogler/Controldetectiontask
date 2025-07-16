# Experimental Design Specifications

**Technical Report #2**  
*Control Detection Task (CDT) Experiment*

---

## Executive Summary

This document provides comprehensive specifications for the Control Detection Task (CDT) experimental design. The CDT is a psychophysical experiment designed to measure agency perception and control detection using personalized motion trajectories with adaptive staircase procedures.

## 1. Theoretical Framework

### 1.1 Research Questions
1. **Primary**: How do individuals detect and rate their sense of agency over visual stimuli?
2. **Secondary**: What role do expectation and motion processing modes play in agency perception?
3. **Tertiary**: How does personalized motion matching affect control detection sensitivity?

### 1.2 Conceptual Model
The CDT is based on the **Predictive Processing Theory of Agency**:
- Agency emerges from predictive models of action-outcome relationships
- Prediction errors drive agency perception updates
- Individual movement styles influence prediction accuracy

### 1.3 Experimental Hypothesis
**H1**: Participants will show higher agency ratings when control proportion matches their expectation level  
**H2**: Motion processing mode (prediction vs. regularity) will modulate agency sensitivity  
**H3**: Personalized motion matching will improve control detection accuracy

## 2. Experimental Design Overview

### 2.1 Design Structure
- **Design Type**: 2 × 2 × 7 within-subjects factorial design
- **Factor 1**: Processing Mode (0° prediction-based, 90° regularity-based)
- **Factor 2**: Expectation Level (low precision 85% target, high precision 55% target)
- **Factor 3**: Control Proportion (7 levels via adaptive staircase)

### 2.2 Trial Structure
```
Demo Trial → Main Trials (4 blocks × 15 trials) → Break → Continue
```

### 2.3 Dependent Variables
- **Primary**: Agency ratings (1-7 Likert scale)
- **Secondary**: Confidence ratings (1-7 scale)
- **Exploratory**: Response times, mouse kinematics, staircase convergence

## 3. Demo Trial Protocol

### 3.1 Purpose
- Familiarize participant with task
- Collect movement style data for personalization
- Establish baseline control level (80% participant control)

### 3.2 Procedure
1. **Instructions**: "DEMO TRIAL – Get familiar with the task"
2. **Fixation**: 1-second white fixation cross
3. **Stimulus Presentation**: Square and dot appear at fixed positions
4. **Movement Detection**: Wait for participant to initiate movement
5. **Trial Duration**: 3 seconds of active movement
6. **Control Mixing**: 80% participant + 20% trajectory snippet
7. **Final Position**: Shapes freeze for 0.5 seconds

### 3.3 Movement Analysis
```python
# Feature extraction from demo trial
demo_features = {
    'mean_speed': np.mean(movement_speeds),
    'mean_turn': np.mean(turning_angles), 
    'straightness': net_displacement / path_length
}

# Cluster assignment (2 closest clusters)
participant_clusters = assign_participant_clusters(demo_features)
```

## 4. Main Trial Structure

### 4.1 Trial Timeline
```
Expectation Cue (2s) → Fixation (1s) → Movement (3s) → 
Agency Rating → Confidence Rating → ITI (1s)
```

### 4.2 Expectation Cueing
- **Low Precision (Yellow)**: "Control level will vary widely"
- **High Precision (Blue)**: "Control level will be consistent"
- **Duration**: 2 seconds
- **Purpose**: Establish prediction context

### 4.3 Stimulus Presentation
- **Shapes**: Square (40×40 px) and dot (diameter 40 px)
- **Colors**: Match expectation cue color
- **Initial Positions**: ±150 px from center (randomized left/right)
- **Movement Boundary**: Circular confinement (radius 250 px)

### 4.4 Movement Control Implementation
```python
# Target shape (participant controls)
target_velocity = control_prop * mouse_velocity + (1 - control_prop) * snippet_velocity

# Distractor shape (pure trajectory)
distractor_velocity = snippet_velocity

# Low-pass filtering for smooth movement
filtered_velocity = LOWPASS * previous_velocity + (1 - LOWPASS) * current_velocity
```

### 4.5 Response Collection
- **Agency Rating**: "How much control did you have? (1=none, 7=complete)"
- **Confidence Rating**: "How confident are you? (1=not at all, 7=very confident)"
- **Response Method**: Keyboard input (1-7 keys)
- **No Time Limit**: Self-paced responses

## 5. Adaptive Staircase Procedures

### 5.1 Staircase Configuration
- **Method**: 1-up/2-down adaptive procedure
- **Convergence Target**: 70.7% accuracy threshold
- **Initial Values**: Based on expectation level
- **Step Sizes**: Asymmetric for up/down adjustments

### 5.2 Target Accuracy Levels
```python
TARGET_ACCURACY = {
    "low": 0.85,    # 85% accuracy (easier detection)
    "high": 0.55    # 55% accuracy (harder detection)
}
```

### 5.3 Step Size Calculation
```python
BASE_STEP = 0.05
STEP_DOWN = {expectation: BASE_STEP for expectation in ["low", "high"]}
STEP_UP = {exp: (TARGET[exp]/(1-TARGET[exp])) * BASE_STEP for exp in ["low", "high"]}
```

### 5.4 Logit-Space Midpoint Calculation
For test trials, control proportion calculated as logit-space midpoint:
```python
def prop_to_logit_midpoint(prop_low, prop_high):
    logit_low = np.log(prop_low / (1 - prop_low))
    logit_high = np.log(prop_high / (1 - prop_high))
    logit_mid = (logit_low + logit_high) / 2
    return 1 / (1 + np.exp(-logit_mid))
```

## 6. Motion Trajectory Management

### 6.1 Trajectory Selection Algorithm
1. **Primary Method**: `find_matched_trajectory_pair()`
   - Matches trajectories based on movement signatures
   - Ensures similar speed, variability, and path characteristics
   
2. **Fallback Method**: `sample_from_participant_clusters()`
   - Uses participant's assigned movement clusters
   - Samples from 2 closest clusters based on demo trial
   
3. **Final Fallback**: Random selection from valid trajectories

### 6.2 Trajectory Preprocessing
- **Quality Control**: Pre-filtered during library creation
- **Consistent Smoothing**: Applied to target/distractor pairs
- **Magnitude Normalization**: Matched to participant movement scale
- **Velocity Mixing**: Real-time combination with participant input

### 6.3 Movement Signature Matching
```python
def get_trajectory_signature(trajectory):
    return {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }
```

## 7. Data Collection Specifications

### 7.1 Main Data File (CSV)
**Filename**: `CDT_v2_{participant_id}.csv`

**Key Variables**:
```python
trial_data = {
    'participant': participant_id,
    'trial_number': trial_index,
    'processing_mode': [0, 90],  # degrees
    'expectation_level': ['low', 'high'],
    'control_proportion': float,  # actual control level
    'agency_rating': int,  # 1-7 scale
    'confidence': int,  # 1-7 scale
    'target_snippet_idx': int,  # motion library index
    'participant_cluster': list,  # assigned clusters
    'response_time_agency': float,  # milliseconds
    'response_time_confidence': float
}
```

### 7.2 Kinematics Data File (CSV)
**Filename**: `CDT_v2_{participant_id}_kinematics.csv`

**Frame-by-frame tracking**:
```python
kinematics_data = {
    'trial_number': int,
    'frame': int,  # 0-149 (150 frames total)
    'timestamp': float,  # relative to trial start
    'mouse_x': float,  # participant mouse position
    'mouse_y': float,
    'target_x': float,  # target shape position
    'target_y': float,
    'distractor_x': float,  # distractor shape position
    'distractor_y': float,
    'control_proportion': float  # instantaneous control level
}
```

### 7.3 Data Storage
- **Location**: `Main Experiment/data/`
- **Format**: CSV files with UTF-8 encoding
- **Backup**: Automatic overwrite protection with versioning
- **File Size**: ~50KB per participant (main data), ~2MB (kinematics)

## 8. Experimental Parameters

### 8.1 Timing Parameters
```python
TIMING = {
    'expectation_cue': 2.0,    # seconds
    'fixation': 1.0,           # seconds
    'movement_phase': 3.0,     # seconds
    'freeze_duration': 0.5,    # seconds
    'inter_trial_interval': 1.0 # seconds
}
```

### 8.2 Visual Parameters
```python
VISUAL = {
    'window_size': (1920, 1080),
    'fullscreen': True,
    'background_color': [0.5, 0.5, 0.5],  # gray
    'shape_size': 40,          # pixels
    'offset_distance': 150,    # pixels from center
    'boundary_radius': 250,    # pixels
    'fixation_size': 60        # pixels
}
```

### 8.3 Control Parameters
```python
CONTROL = {
    'demo_proportion': 0.80,   # participant control in demo
    'lowpass_filter': 0.8,     # movement smoothing
    'break_frequency': 15,     # trials between breaks
    'total_trials': 60,        # 4 conditions × 15 trials
    'cluster_assignment': 2    # number of clusters per participant
}
```

## 9. Quality Control Measures

### 9.1 Real-time Validation
- **Movement Detection**: Ensures participant initiates movement
- **Boundary Enforcement**: Confines shapes within circular area
- **Response Validation**: Checks for valid key presses (1-7)
- **Timing Accuracy**: Maintains precise trial timing

### 9.2 Data Integrity Checks
- **File Protection**: Prevents accidental overwriting
- **Complete Trials**: Ensures all responses collected
- **Kinematics Sync**: Verifies frame-by-frame alignment
- **Cluster Validity**: Confirms participant assignment success

### 9.3 Participant Monitoring
- **Break Reminders**: Automatic breaks every 15 trials
- **Escape Option**: Safe experiment termination
- **Progress Tracking**: Visual feedback on completion
- **Error Handling**: Graceful failure recovery

## 10. Statistical Considerations

### 10.1 Power Analysis
- **Effect Size**: Expected η² = 0.15 for main effects
- **Power**: 80% with α = 0.05
- **Sample Size**: N = 24 participants minimum
- **Repeated Measures**: Within-subjects design increases power

### 10.2 Analysis Plan
1. **Descriptive Statistics**: Agency ratings by condition
2. **ANOVA**: 2×2 repeated measures (Mode × Expectation)
3. **Signal Detection**: d-prime and response bias calculation
4. **Mixed Models**: Account for individual differences
5. **Staircase Analysis**: Convergence and threshold estimation

### 10.3 Planned Contrasts
- **Expectation Effect**: Low vs. High precision conditions
- **Processing Mode**: 0° vs. 90° motion processing
- **Interaction**: Mode × Expectation interaction
- **Individual Differences**: Cluster-based grouping analysis

## 11. Technical Implementation

### 11.1 Software Dependencies
- **PsychoPy**: Version 2023.2+ for stimulus presentation
- **NumPy**: Array processing and numerical computation
- **Pandas**: Data manipulation and CSV handling
- **JSON**: Parameter and metadata storage

### 11.2 Hardware Requirements
- **Display**: 1920×1080 minimum resolution
- **Input**: Standard computer mouse
- **Processor**: 2GHz+ for real-time processing
- **Memory**: 4GB+ RAM for motion library loading

### 11.3 Platform Compatibility
- **Primary**: Windows 10/11, macOS 10.15+
- **Testing**: Ubuntu 20.04+ (Linux support)
- **Browser**: Chrome/Firefox for web deployment
- **Mobile**: Not supported (requires precise mouse input)

