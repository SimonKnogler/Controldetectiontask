# Control Detection Task (CDT) - Motion Library Experiment

This repository contains a complete experimental pipeline for a **Control Detection Task (CDT)**, a psychophysics experiment that uses traget and distractor movement trajectories to study agency perception and control detection.

## 🗂️ Repository Structure

### `Main Experiment/`
**Core experimental implementations using PsychoPy**

- **`CDT_mac.py`** - Main experimental script (macOS version)
  - Implements the full CDT with personalized motion library
  - Features participant cluster assignment based on demo trial movement
  - Includes trajectory matching, quality control, and adaptive staircases
  - Saves data to `Main Experiment/data/` directory
  - Supports simulation mode for testing
  - Optimized for macOS with native path handling

- **`CDT_windows.py`** - Windows-compatible version of the main experiment
  - Cross-platform path handling for Windows systems
  - Same functionality as macOS version
  - Automatic Python environment detection

- **`CDT_eyetracking.py`** - Eye tracking version with gaze data collection
  - Supports Tobii and EyeLink eye trackers
  - Records gaze coordinates, pupil diameter, and fixation data
  - Synchronized eye tracking and kinematics data
  - Automatic eye tracker detection and calibration

- **`data/`** - Experimental data output directory
  - Contains participant CSV files (`CDT_mac_{participant_id}.csv`, `CDT_windows_{participant_id}.csv`, `CDT_eyetracking_{participant_id}.csv`)
  - Contains kinematics data (`CDT_*_{participant_id}_kinematics.csv`)
  - Contains eye tracking data (`CDT_eyetracking_{participant_id}_eyetracking.csv`)

### `Motion Library/`
**Motion snippet library and creation tools**

#### Core Library Files:
- **`core_pool.npy`** - Main motion library (240 trajectories × 150 frames × 2D coordinates)
- **`core_pool_feats.npy`** - Feature vectors for each trajectory (6D features per snippet)
- **`core_pool_labels.npy`** - Cluster assignments for each trajectory (4 clusters: 0-3)
- **`scaler_params.json`** - Normalization parameters (mean/std for feature scaling)
- **`cluster_centroids.json`** - K-means cluster centers for participant matching

#### Library Creation Scripts:
- **`motionlib_create_filtered.py`** - Main library creation pipeline
  - Processes raw recordings from Pavlovia experiment
  - Applies quality filtering and trajectory preprocessing
  - Performs K-means clustering (4 clusters)
  - Generates all core library files

#### Experiment Data Collection:
- **`Experiment Pavlovia/`** - Web-based data collection experiment
  - **`buildermolib.psyexp`** - PsychoPy experiment file
  - **`index.html`** - Web deployment files
  - **`data/`** - Raw participant recordings used to build motion library

#### Analysis and Utilities:
- **`motion_analysis.py`** - Library analysis and quality metrics
- **`motion_analysis.R`** - R script for statistical analysis of motion patterns

### `Scripts/`
**Analysis and utility scripts**

- **`analysis.py`** - Python data analysis pipeline
- **`lib_create.py`** - Library creation utilities
- **`MLP.py`** - Machine learning pipeline for motion classification
- **`Plots/`** - Generated analysis plots
  - `agency_plot.png` - Agency rating distributions
  - `conf_plot.png` - Confidence analysis
  - `dprime_plot.png` - Signal detection analysis

### `Technical Reports/`
**Documentation and technical specifications**

*This folder will contain detailed technical documentation about:*
- Motion library creation methodology
- Experimental design specifications  
- Data processing pipelines
- Quality control procedures
- Clustering and personalization algorithms

### `Trial Demo/`
**Demonstration materials**

- **`Demo_trial_vid.mp4`** - Video demonstration of the experimental task
- **`trial_vid.mov`** - Alternative format demonstration video

#### 🎥 **Video Demonstration**
Watch the experimental task in action:

![Control Detection Task Demo](Trial%20Demo/trial_recording.gif)

*This GIF shows a complete trial of the Control Detection Task, demonstrating the mouse-controlled movement, trajectory matching, and agency rating process. The animation plays automatically on GitHub.*

### `tests/`
**Test suite for trajectory processing**

- **`test_assign_cluster.py`** - Unit tests for cluster assignment functionality

## 🔬 How the Experiment Works

### 1. **Demo Trial & Participant Profiling**
- Participant performs a demo trial with high control (80%)
- System analyzes movement features:
  - **Mean speed** - Average movement velocity
  - **Mean turn** - Average turning angles
  - **Straightness** - Path efficiency (net displacement / total path length)
- Participant is assigned to 2 closest movement style clusters

### 2. **Personalized Motion Selection**
- Main trials use trajectory matching algorithm:
  1. **Primary**: `find_matched_trajectory_pair()` - finds similar trajectories
  2. **Fallback**: `sample_from_participant_clusters()` - uses participant's clusters
  3. **Final**: Random selection from valid trajectories
- All trajectories undergo quality control and normalization

### 3. **Adaptive Control Manipulation**
- Two processing modes: 0° (prediction-based) and 90° (regularity-based)
- Two expectation levels: low precision (85% target) and high precision (55% target)
- Staircase procedures adapt control proportion based on participant responses

### 4. **Data Collection**
- 7-point Likert scale for agency ratings
- Confidence ratings on agency judgments
- Complete kinematics tracking (mouse positions, velocities)
- Trial-by-trial metadata (control proportions, trajectories used, etc.)

## 🧬 Motion Library Details

### Library Composition
- **325** original trajectories collected from 13 participants
- **240** high-quality trajectories after filtering
- **4** movement style clusters:
  - Cluster 0: Typical movement (58% of trajectories)
  - Cluster 1: Fast, jerky movement
  - Cluster 2: Slow, smooth movement  
  - Cluster 3: Variable movement patterns

### Feature Engineering
Each trajectory is characterized by 6 features:
1. Mean speed
2. Speed variability
3. Mean turning angle
4. Turn variability  
5. Path straightness
6. Movement consistency

### Quality Control
- Minimum speed thresholds
- Maximum jitter/noise ratios
- Smoothness requirements
- Trajectory length validation

## 🚀 Quick Start

### Running the Experiment

**macOS:**
```bash
cd "Main Experiment"
python CDT_mac.py
```

**Windows:**
```bash
cd "Main Experiment"
python CDT_windows.py
```

**With Eye Tracking:**
```bash
cd "Main Experiment"
python CDT_eyetracking.py
```

### Creating a New Motion Library
```bash
cd "Motion Library"
python motionlib_create_filtered.py
```

### Data Analysis
```bash
cd Scripts
python analysis.py
```

### Simulation Mode
Enable "Simulate" checkbox in startup dialog for automated testing without human participant.

## 📊 Data Output

### Main Data Files
- **`CDT_mac_{participant_id}.csv`** - Trial-by-trial responses and metadata (macOS)
- **`CDT_windows_{participant_id}.csv`** - Trial-by-trial responses and metadata (Windows)
- **`CDT_eyetracking_{participant_id}.csv`** - Trial-by-trial responses and metadata (Eye tracking)
- **`CDT_*_{participant_id}_kinematics.csv`** - Complete movement tracking data
- **`CDT_eyetracking_{participant_id}_eyetracking.csv`** - Eye tracking data (gaze, pupil, etc.)

### Key Variables
- `agency_rating` - 1-7 Likert scale agency perception
- `confidence` - Confidence in agency judgment
- `control_proportion` - Actual control level in trial
- `processing_mode` - 0° (prediction) or 90° (regularity)
- `expectation_level` - "low" or "high" precision condition
- `participant_cluster` - Assigned movement style cluster(s)
- `target_snippet_idx` - Motion library trajectory used

## 🔧 Configuration

### Key Parameters
- **`SIMULATE`** - Enable/disable simulation mode
- **`TOTAL_SNIPS`** - Number of trajectories in motion pool
- **`SNIP_LEN`** - Trajectory length (150 frames = 3 seconds at 50 Hz)
- **`K_CLUST`** - Number of movement clusters (4)
- **`BREAK_EVERY`** - Trials between breaks (15)

### File Paths
All paths are relative to script locations - no manual configuration needed.

## 🧪 Testing

Run the test suite:
```bash
cd tests
python -m pytest test_assign_cluster.py -v
```

## 📝 Dependencies

- **PsychoPy** - Experimental presentation
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation  
- **Scikit-learn** - Machine learning (clustering)
- **Matplotlib** - Plotting and visualization

## 📄 Citation

If you use this code or motion library in your research, please cite the associated publication [details to be added].

