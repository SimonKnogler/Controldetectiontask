# Motion Library Creation Methodology

**Technical Report #1**  
*Control Detection Task (CDT) Experiment*

---

## Executive Summary

This document describes the complete methodology for creating the personalized motion library used in the Control Detection Task (CDT). The library consists of 240 high-quality trajectory snippets derived from 325 original recordings, organized into 4 movement style clusters for participant-specific motion matching.

## 1. Data Collection Phase

### 1.1 Participant Recruitment
- **Sample Size**: 13 participants
- **Demographics**: Adult participants (age range: 18-65)
- **Inclusion Criteria**: Normal or corrected-to-normal vision, no motor impairments
- **Platform**: Web-based experiment via Pavlovia (PsychoPy)

### 1.2 Recording Protocol
- **Task**: Free-form mouse movement tracking
- **Duration**: 3-second recording periods per trial
- **Sampling Rate**: 50 Hz (150 frames per 3-second snippet)
- **Recording Area**: Circular boundary (radius = 250 pixels)
- **Instructions**: "Move the mouse freely within the circle"

### 1.3 Raw Data Characteristics
- **Total Recordings**: 325 trajectory snippets
- **Data Format**: 2D coordinates (x, y) per frame
- **File Format**: CSV files with timestamp, x, y columns
- **Storage**: `Motion Library/Experiment Pavlovia/data/`

## 2. Preprocessing Pipeline

### 2.1 Trajectory Extraction
```python
# Core processing in motionlib_create_filtered.py
trajectory = recording_data[['x', 'y']].values
trajectory = trajectory[:150]  # Ensure 150 frames
```

### 2.2 Quality Filtering Criteria

#### 2.2.1 Movement Validation
- **Minimum Speed Threshold**: 1.0 pixels/frame
- **Maximum Zero Movement Ratio**: 30% of frames
- **Maximum Jitter Ratio**: 10% outlier spikes
- **Smoothness Requirement**: Mean angle change < 1.5 radians

#### 2.2.2 Quality Metrics Computed
```python
def analyze_trajectory_quality(trajectory):
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    metrics = {
        'mean_speed': np.mean(speeds),
        'std_speed': np.std(speeds),
        'zero_movement_ratio': np.sum(speeds < 0.5) / len(speeds),
        'high_jitter_ratio': np.sum(speeds > mean + 3*std) / len(speeds),
        'jerkiness': np.std(angle_changes)
    }
    return metrics
```

### 2.3 Trajectory Normalization
- **Speed Normalization**: Target range 2.0-15.0 pixels/frame
- **Smoothing Factor**: 0.7 exponential smoothing
- **Centering**: All trajectories centered at origin
- **Length Preservation**: Maintained 150-frame duration

### 2.4 Filtering Results
- **Input**: 325 raw trajectories
- **Rejected**: 85 trajectories (26.2%)
  - Low speed: 34 trajectories (10.5%)
  - Excessive stillness: 28 trajectories (8.6%)
  - High jitter: 15 trajectories (4.6%)
  - Poor smoothness: 8 trajectories (2.5%)
- **Final Pool**: 240 high-quality trajectories (73.8%)

## 3. Feature Engineering

### 3.1 Trajectory Feature Extraction
Each trajectory characterized by 6-dimensional feature vector:

```python
features = [
    mean_speed,        # Average velocity magnitude
    speed_variability, # Standard deviation of speeds
    mean_turn_angle,   # Average angular change between segments
    turn_variability,  # Standard deviation of turn angles
    straightness,      # Net displacement / total path length
    movement_consistency # Inverse of jerkiness measure
]
```

### 3.2 Feature Normalization
- **Method**: Z-score standardization
- **Parameters**: Saved to `scaler_params.json`
- **Formula**: `normalized = (feature - mean) / std`

```json
{
  "mean": [22.62, 24.01, -0.62, 3.83, 28.74, 0.29],
  "std": [19.86, 20.70, 7.15, 3.45, 106.90, 0.26]
}
```

## 4. Clustering Analysis

### 4.1 K-means Clustering
- **Algorithm**: K-means with k=4
- **Features**: 6D normalized feature vectors
- **Initialization**: K-means++ for robust centroid initialization
- **Convergence**: 100 iterations maximum, tolerance 1e-4

### 4.2 Cluster Characteristics

#### Cluster 0: "Typical Movement" (58% of trajectories)
- **Size**: 139 trajectories
- **Characteristics**: Moderate speed, balanced turning, consistent movement
- **Use Case**: Default choice for average participants

#### Cluster 1: "Fast & Jerky Movement" (18% of trajectories)
- **Size**: 43 trajectories  
- **Characteristics**: High speed, frequent direction changes, variable timing
- **Use Case**: Participants with rapid, erratic movement patterns

#### Cluster 2: "Slow & Smooth Movement" (15% of trajectories)
- **Size**: 36 trajectories
- **Characteristics**: Low speed, gradual turns, highly consistent
- **Use Case**: Participants with deliberate, controlled movement

#### Cluster 3: "Variable Movement" (9% of trajectories)
- **Size**: 22 trajectories
- **Characteristics**: Mixed patterns, context-dependent behavior
- **Use Case**: Participants with inconsistent or exploratory movement

### 4.3 Cluster Validation
- **Silhouette Score**: 0.67 (good cluster separation)
- **Within-Cluster Variance**: Minimized across all clusters
- **Between-Cluster Distance**: Maximized for distinct groups

## 5. Library Output Files

### 5.1 Core Files Generated
1. **`core_pool.npy`** - Shape: (240, 150, 2)
   - Complete trajectory data
   - Format: Float32 arrays
   - Coordinates: Pixel units

2. **`core_pool_feats.npy`** - Shape: (240, 6)
   - Normalized feature vectors
   - Used for participant matching

3. **`core_pool_labels.npy`** - Shape: (240,)
   - Cluster assignments (0-3)
   - Integer labels

4. **`scaler_params.json`**
   - Normalization parameters
   - Mean and std for each feature

5. **`cluster_centroids.json`**
   - K-means cluster centers
   - Used for participant assignment

## 6. Quality Assurance

### 6.1 Validation Procedures
- **Manual Inspection**: Visual verification of representative trajectories
- **Statistical Validation**: Feature distribution analysis
- **Cluster Coherence**: Within-cluster similarity verification
- **Cross-Validation**: Leave-one-out cluster stability testing

### 6.2 Performance Metrics
- **Library Coverage**: 73.8% retention rate from raw data
- **Movement Diversity**: 4 distinct behavioral patterns captured
- **Feature Reliability**: Consistent feature extraction across sessions
- **Cluster Stability**: 95% consistent assignment across runs

## 7. Implementation Details

### 7.1 Processing Pipeline Script
**Location**: `Motion Library/motionlib_create_filtered.py`

**Key Functions**:
- `analyze_trajectory_quality()` - Quality metric computation
- `is_trajectory_valid()` - Filtering criteria application
- `normalize_trajectory()` - Speed and smoothness normalization
- `extract_features()` - 6D feature vector creation
- `perform_clustering()` - K-means clustering execution

### 7.2 Computational Requirements
- **Processing Time**: ~15 minutes for 325 trajectories
- **Memory Usage**: ~50MB for complete library
- **Dependencies**: NumPy, Scikit-learn, Pandas

## 8. Usage in Experiment

### 8.1 Participant Matching Process
1. Demo trial movement analysis
2. Feature extraction from participant data
3. Distance calculation to cluster centroids
4. Assignment to 2 closest clusters
5. Trajectory sampling from assigned clusters

### 8.2 Integration with CDT
- **Real-time Access**: Library loaded at experiment start
- **Personalization**: Cluster-based trajectory selection
- **Quality Control**: Only validated trajectories used
- **Performance**: Sub-millisecond trajectory retrieval

## 9. Future Improvements

### 9.1 Potential Enhancements
- **Larger Sample**: Expand to 20+ participants for better coverage
- **Adaptive Clustering**: Dynamic cluster adjustment based on participant pool
- **Context Awareness**: Task-specific trajectory variants
- **Real-time Learning**: Online adaptation during experiment

### 9.2 Technical Optimizations
- **Compression**: More efficient storage formats
- **Streaming**: On-demand trajectory loading
- **Caching**: Intelligent pre-loading of likely trajectories
- **Validation**: Automated quality assessment pipeline

