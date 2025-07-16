# Data Processing Pipeline

**Technical Report #3**  
*Control Detection Task (CDT) Experiment*

---

## Executive Summary

This document describes the complete data processing pipeline for the Control Detection Task (CDT), covering real-time data collection, post-experiment processing, quality control measures, and analysis workflows. The pipeline handles both behavioral responses and high-frequency kinematics data with comprehensive validation procedures.

## 1. Real-Time Data Collection

### 1.1 Data Acquisition Framework
- **Sampling Rate**: 50 Hz for movement tracking
- **Data Types**: Mouse coordinates, stimulus positions, timing events
- **Storage Format**: In-memory buffers → CSV files
- **Synchronization**: Frame-locked data collection

### 1.2 Movement Tracking System
```python
class MovementTracker:
    def __init__(self, sampling_rate=50):
        self.sampling_rate = sampling_rate
        self.frame_duration = 1.0 / sampling_rate
        self.data_buffer = []
        
    def record_frame(self, trial_number, frame_index, timestamp):
        frame_data = {
            'trial_number': trial_number,
            'frame': frame_index,
            'timestamp': timestamp,
            'mouse_x': mouse.getPos()[0],
            'mouse_y': mouse.getPos()[1],
            'target_x': target_shape.pos[0],
            'target_y': target_shape.pos[1],
            'distractor_x': distractor_shape.pos[0],
            'distractor_y': distractor_shape.pos[1],
            'control_proportion': current_control_level
        }
        self.data_buffer.append(frame_data)
```

### 1.3 Response Collection
- **Agency Ratings**: Keyboard input (1-7 keys)
- **Confidence Ratings**: Keyboard input (1-7 keys)
- **Response Times**: Millisecond precision timing
- **Validation**: Real-time input verification

## 2. Data Storage Architecture

### 2.1 File Structure
```
Main Experiment/data/
├── CDT_v2_{participant_id}.csv          # Main behavioral data
├── CDT_v2_{participant_id}_kinematics.csv # Movement tracking data
└── backup/
    ├── CDT_v2_{participant_id}_1.csv    # Automatic backups
    └── CDT_v2_{participant_id}_1_kinematics.csv
```

### 2.2 Main Data Schema
**File**: `CDT_v2_{participant_id}.csv`

| Column | Type | Description |
|--------|------|-------------|
| participant | string | Unique participant identifier |
| trial_number | int | Sequential trial index (0-59) |
| processing_mode | int | Motion processing angle (0°/90°) |
| expectation_level | string | Precision condition ('low'/'high') |
| control_proportion | float | Actual control level (0.0-1.0) |
| agency_rating | int | Participant's agency judgment (1-7) |
| confidence | int | Confidence in rating (1-7) |
| target_snippet_idx | int | Motion library trajectory index |
| distractor_snippet_idx | int | Distractor trajectory index |
| participant_cluster | string | Assigned movement clusters |
| response_time_agency | float | Agency response latency (ms) |
| response_time_confidence | float | Confidence response latency (ms) |
| trial_start_time | float | Trial onset timestamp |
| staircase_direction | string | Staircase adjustment ('up'/'down') |

### 2.3 Kinematics Data Schema
**File**: `CDT_v2_{participant_id}_kinematics.csv`

| Column | Type | Description |
|--------|------|-------------|
| trial_number | int | Trial identifier |
| frame | int | Frame index within trial (0-149) |
| timestamp | float | Time relative to trial start (seconds) |
| mouse_x | float | Participant mouse X coordinate |
| mouse_y | float | Participant mouse Y coordinate |
| target_x | float | Target shape X position |
| target_y | float | Target shape Y position |
| distractor_x | float | Distractor shape X position |
| distractor_y | float | Distractor shape Y position |
| control_proportion | float | Instantaneous control level |
| mouse_velocity_x | float | Mouse velocity X component |
| mouse_velocity_y | float | Mouse velocity Y component |
| target_velocity_x | float | Target velocity X component |
| target_velocity_y | float | Target velocity Y component |

## 3. Real-Time Processing

### 3.1 Movement Analysis Pipeline
```python
def process_movement_frame(mouse_pos, previous_pos, dt):
    # Calculate instantaneous velocity
    velocity = (mouse_pos - previous_pos) / dt
    
    # Apply low-pass filter for smoothing
    filtered_velocity = LOWPASS * prev_velocity + (1 - LOWPASS) * velocity
    
    # Trajectory mixing with motion library
    snippet_velocity = get_snippet_velocity(current_frame)
    mixed_velocity = control_prop * filtered_velocity + (1 - control_prop) * snippet_velocity
    
    # Boundary enforcement
    new_position = enforce_boundary(current_pos + mixed_velocity * dt)
    
    return new_position, filtered_velocity
```

### 3.2 Quality Control During Collection
- **Movement Validation**: Continuous speed monitoring
- **Boundary Checking**: Real-time confinement enforcement
- **Data Completeness**: Frame-by-frame validation
- **Response Verification**: Valid key press confirmation

### 3.3 Adaptive Staircase Updates
```python
def update_staircase(response, current_level, expectation):
    if response >= 5:  # High agency rating (correct detection)
        # Make task harder (reduce control)
        new_level = current_level - STEP_DOWN[expectation]
    else:  # Low agency rating (missed detection)
        # Make task easier (increase control)
        new_level = current_level + STEP_UP[expectation]
    
    # Enforce bounds [0.0, 1.0]
    return np.clip(new_level, 0.0, 1.0)
```

## 4. Post-Experiment Processing

### 4.1 Data Validation Pipeline
```python
def validate_experiment_data(participant_id):
    main_data = load_main_data(participant_id)
    kinematics_data = load_kinematics_data(participant_id)
    
    validation_results = {
        'complete_trials': check_trial_completeness(main_data),
        'response_validity': check_response_ranges(main_data),
        'kinematics_sync': check_frame_alignment(kinematics_data),
        'timing_consistency': check_timing_accuracy(kinematics_data),
        'movement_quality': assess_movement_patterns(kinematics_data)
    }
    
    return validation_results
```

### 4.2 Derived Variable Calculation
```python
def calculate_derived_variables(kinematics_data):
    # Movement characteristics
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    accelerations = np.diff(speeds)
    
    # Path analysis
    total_path_length = np.sum(speeds)
    net_displacement = np.linalg.norm(positions[-1] - positions[0])
    straightness_index = net_displacement / total_path_length
    
    # Temporal dynamics
    pause_count = np.sum(speeds < MOVEMENT_THRESHOLD)
    movement_efficiency = np.mean(speeds[speeds > MOVEMENT_THRESHOLD])
    
    # Control analysis
    control_variability = np.std(control_proportions)
    effective_control = np.mean(control_proportions)
    
    return derived_metrics
```

### 4.3 Movement Feature Extraction
```python
def extract_movement_features(trial_kinematics):
    mouse_trajectory = trial_kinematics[['mouse_x', 'mouse_y']].values
    
    features = {
        # Speed characteristics
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'max_speed': np.max(speeds),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75, 90]),
        
        # Path characteristics
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(mouse_trajectory[-1] - mouse_trajectory[0]),
        'straightness': net_displacement / path_length,
        'area_covered': calculate_convex_hull_area(mouse_trajectory),
        
        # Temporal characteristics
        'movement_duration': calculate_active_movement_time(speeds),
        'pause_frequency': count_movement_pauses(speeds),
        'acceleration_changes': count_acceleration_reversals(accelerations),
        
        # Smoothness characteristics
        'jerkiness': calculate_path_jerkiness(mouse_trajectory),
        'angular_velocity': np.mean(np.abs(angular_changes)),
        'curvature': calculate_path_curvature(mouse_trajectory)
    }
    
    return features
```

## 5. Quality Control Procedures

### 5.1 Participant-Level Validation
```python
def validate_participant_performance(participant_data):
    validation_checks = {
        # Response patterns
        'response_variance': check_response_variability(participant_data['agency_rating']),
        'ceiling_floor_effects': check_rating_extremes(participant_data['agency_rating']),
        'response_consistency': check_confidence_alignment(participant_data),
        
        # Movement patterns
        'movement_engagement': check_movement_amount(kinematics_data),
        'boundary_compliance': check_area_violations(kinematics_data),
        'movement_consistency': check_speed_patterns(kinematics_data),
        
        # Staircase convergence
        'staircase_stability': check_convergence_patterns(participant_data),
        'learning_effects': check_temporal_trends(participant_data),
        'condition_balance': check_trial_distribution(participant_data)
    }
    
    return validation_checks
```

### 5.2 Trial-Level Quality Metrics
```python
def assess_trial_quality(trial_data, trial_kinematics):
    quality_metrics = {
        # Movement quality
        'sufficient_movement': np.mean(speeds) > MIN_MOVEMENT_THRESHOLD,
        'smooth_trajectory': jerkiness_score < MAX_JERKINESS,
        'boundary_violations': count_boundary_exits(positions),
        'temporal_consistency': check_frame_timing(timestamps),
        
        # Response quality
        'valid_responses': all(1 <= rating <= 7 for rating in responses),
        'reasonable_timing': MIN_RT < response_time < MAX_RT,
        'engagement_indicators': response_variability > MIN_VARIANCE,
        
        # Technical quality
        'data_completeness': len(trial_kinematics) == EXPECTED_FRAMES,
        'sync_accuracy': check_timestamp_consistency(trial_kinematics),
        'processing_errors': check_for_anomalies(trial_data)
    }
    
    return quality_metrics
```

### 5.3 Data Exclusion Criteria
```python
EXCLUSION_CRITERIA = {
    'participant_level': {
        'min_movement_engagement': 0.5,    # 50% of trials with sufficient movement
        'max_ceiling_proportion': 0.8,     # <80% maximum ratings
        'min_response_variance': 1.0,      # Minimum response variability
        'max_missing_trials': 0.1          # <10% incomplete trials
    },
    
    'trial_level': {
        'min_movement_distance': 100,      # Minimum total movement (pixels)
        'max_boundary_violations': 10,     # Maximum boundary exits
        'min_response_time': 200,          # Minimum response time (ms)
        'max_response_time': 30000         # Maximum response time (ms)
    }
}
```

## 6. Analysis Workflows

### 6.1 Behavioral Data Analysis
```python
def analyze_behavioral_data(all_participants_data):
    # Aggregate across participants
    summary_stats = calculate_descriptive_statistics(all_participants_data)
    
    # Main analyses
    anova_results = perform_repeated_measures_anova(
        dv='agency_rating',
        factors=['processing_mode', 'expectation_level'],
        subject='participant',
        data=all_participants_data
    )
    
    # Signal detection analysis
    sdt_results = calculate_signal_detection_metrics(all_participants_data)
    
    # Individual differences
    cluster_analysis = analyze_movement_clusters(all_participants_data)
    
    return {
        'descriptives': summary_stats,
        'anova': anova_results,
        'signal_detection': sdt_results,
        'individual_differences': cluster_analysis
    }
```

### 6.2 Movement Analysis Pipeline
```python
def analyze_movement_patterns(all_kinematics_data):
    # Extract movement features for each participant
    participant_features = {}
    for participant in participants:
        participant_kinematics = filter_by_participant(all_kinematics_data, participant)
        features = extract_comprehensive_features(participant_kinematics)
        participant_features[participant] = features
    
    # Cluster analysis
    movement_clusters = perform_movement_clustering(participant_features)
    
    # Relationship to behavior
    feature_behavior_correlation = correlate_movement_with_ratings(
        participant_features, behavioral_data
    )
    
    return {
        'movement_features': participant_features,
        'clusters': movement_clusters,
        'behavior_correlations': feature_behavior_correlation
    }
```

### 6.3 Staircase Analysis
```python
def analyze_staircase_performance(participant_data):
    staircase_metrics = {}
    
    for condition in ['0_low', '0_high', '90_low', '90_high']:
        condition_data = filter_by_condition(participant_data, condition)
        
        metrics = {
            'threshold_estimate': calculate_threshold(condition_data),
            'convergence_rate': assess_convergence(condition_data),
            'stability_index': calculate_stability(condition_data),
            'learning_trend': detect_learning_effects(condition_data)
        }
        
        staircase_metrics[condition] = metrics
    
    return staircase_metrics
```

## 7. Data Export and Sharing

### 7.1 Standardized Output Formats
```python
def export_analysis_ready_data(participant_data, output_format='csv'):
    """Export cleaned, validated data for analysis"""
    
    if output_format == 'csv':
        # Behavioral data
        behavioral_df = prepare_behavioral_dataframe(participant_data)
        behavioral_df.to_csv('CDT_behavioral_data.csv', index=False)
        
        # Movement summary
        movement_df = prepare_movement_summary(participant_data)
        movement_df.to_csv('CDT_movement_features.csv', index=False)
        
    elif output_format == 'r':
        # R-compatible format
        export_for_r_analysis(participant_data, 'CDT_data_for_R.RData')
        
    elif output_format == 'spss':
        # SPSS format
        export_for_spss(participant_data, 'CDT_data.sav')
```

### 7.2 Metadata Documentation
```python
def generate_data_documentation(participant_data):
    documentation = {
        'study_info': {
            'experiment_name': 'Control Detection Task',
            'version': 'CDT_v2',
            'collection_date_range': get_date_range(participant_data),
            'total_participants': len(participant_data),
            'total_trials': count_total_trials(participant_data)
        },
        
        'variable_definitions': get_variable_codebook(),
        'processing_log': get_processing_history(),
        'quality_metrics': get_overall_quality_summary(participant_data),
        'exclusion_log': get_exclusion_summary(participant_data)
    }
    
    save_json(documentation, 'CDT_data_documentation.json')
```

## 8. Error Handling and Recovery

### 8.1 Real-Time Error Recovery
```python
def handle_collection_errors(error_type, context):
    if error_type == 'movement_tracking_failure':
        # Fallback to simplified tracking
        enable_backup_movement_tracking()
        log_error('Movement tracking degraded', context)
        
    elif error_type == 'file_write_failure':
        # Switch to backup location
        redirect_to_backup_storage()
        log_error('Primary storage failed', context)
        
    elif error_type == 'trajectory_loading_failure':
        # Use fallback trajectories
        switch_to_backup_trajectories()
        log_error('Motion library access failed', context)
```

### 8.2 Data Recovery Procedures
```python
def recover_incomplete_data(participant_id):
    """Attempt to recover partially collected data"""
    
    # Check for temporary files
    temp_files = find_temporary_data_files(participant_id)
    
    # Reconstruct from backup
    backup_data = load_backup_data(participant_id)
    
    # Merge and validate
    recovered_data = merge_data_sources(temp_files, backup_data)
    recovery_success = validate_recovered_data(recovered_data)
    
    return recovered_data, recovery_success
```

