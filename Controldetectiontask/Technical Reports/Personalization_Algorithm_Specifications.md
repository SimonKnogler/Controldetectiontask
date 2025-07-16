# Personalization Algorithm Specifications

**Technical Report #4**  
*Control Detection Task (CDT) Experiment*

---

## Executive Summary

This document provides detailed specifications for the personalization algorithms used in the Control Detection Task (CDT). The personalization system adapts motion trajectories to individual participant movement styles through cluster-based assignment, trajectory matching, and real-time selection algorithms to enhance agency perception sensitivity.

## 1. Personalization Framework Overview

### 1.1 Theoretical Motivation
- **Individual Differences**: Participants have distinct movement signatures that affect agency perception
- **Prediction Accuracy**: Personalized motion improves predictive model accuracy
- **Ecological Validity**: Matching participant style increases task naturalism
- **Sensitivity Enhancement**: Personalization improves control detection sensitivity

### 1.2 System Architecture
```
Demo Trial → Feature Extraction → Cluster Assignment → Motion Selection → Real-time Adaptation
```

### 1.3 Personalization Components
1. **Movement Style Profiling**: Demo trial analysis and feature extraction
2. **Cluster Assignment**: Mapping participants to movement clusters
3. **Trajectory Matching**: Real-time selection of appropriate motion snippets
4. **Adaptive Selection**: Dynamic adjustment based on trial context

## 2. Movement Style Profiling

### 2.1 Demo Trial Analysis
The personalization process begins with a high-control demo trial (80% participant control) designed to capture authentic movement patterns.

```python
def analyze_demo_movement(demo_positions, demo_timestamps):
    """Extract movement characteristics from demo trial"""
    
    # Calculate frame-by-frame movement metrics
    deltas = np.diff(demo_positions, axis=0)
    speeds = np.linalg.norm(deltas, axis=1)
    
    # Core movement features
    demo_mean_speed = float(np.mean(speeds))
    
    # Angular analysis
    unit_vectors = deltas / (speeds.reshape(-1, 1) + 1e-9)
    dot_products = np.einsum('ij,ij->i', unit_vectors[:-1], unit_vectors[1:])
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    demo_mean_turn = float(np.mean(angles))
    
    # Path efficiency
    net_displacement = demo_positions[-1] - demo_positions[0]
    net_magnitude = np.linalg.norm(net_displacement)
    path_length = np.sum(speeds)
    demo_straightness = float(net_magnitude / (path_length + 1e-9))
    
    return demo_mean_speed, demo_mean_turn, demo_straightness
```

### 2.2 Feature Vector Construction
The participant's movement style is encoded as a 6-dimensional feature vector compatible with the motion library's feature space:

```python
def construct_participant_feature_vector(demo_speed, demo_turn, demo_straightness):
    """Create feature vector matching motion library format"""
    
    # Match the 6D structure used in motion library
    # [speed, speed_var, turn, turn_var, straightness, consistency]
    demo_vector = np.array([
        demo_speed,      # Mean movement speed
        0.0,             # Speed variability (not calculated from single demo)
        demo_turn,       # Mean turning angle
        0.0,             # Turn variability (not calculated from single demo)
        demo_straightness, # Path straightness measure
        0.0              # Movement consistency (not calculated from single demo)
    ], dtype=np.float32)
    
    return demo_vector
```

### 2.3 Feature Normalization
Participant features are normalized using the same scaler parameters from motion library creation:

```python
def normalize_participant_features(demo_vector, scaler_mean, scaler_std):
    """Apply z-score normalization using motion library parameters"""
    
    normalized_vector = (demo_vector - scaler_mean) / (scaler_std + 1e-9)
    return normalized_vector
```

## 3. Cluster Assignment Algorithm

### 3.1 Distance-Based Assignment
Participants are assigned to movement clusters based on Euclidean distance to cluster centroids in normalized feature space:

```python
def assign_participant_clusters(demo_mean_speed, demo_mean_turn, demo_straightness):
    """Assign participant to 2 closest movement clusters"""
    
    # Construct and normalize feature vector
    demo_vector = construct_participant_feature_vector(
        demo_mean_speed, demo_mean_turn, demo_straightness
    )
    demo_scaled = normalize_participant_features(demo_vector, scaler_mean, scaler_std)
    
    # Calculate distances to all cluster centroids
    distances = np.linalg.norm(
        CLUSTER_CENTROIDS - demo_scaled.reshape(1, -1), 
        axis=1
    )
    
    # Select 2 closest clusters
    closest_indices = np.argsort(distances)[:2]
    
    return closest_indices.tolist()
```

### 3.2 Cluster Characteristics and Assignment Logic

#### Cluster 0: "Typical Movement" (58% of library)
- **Characteristics**: Moderate speed, balanced turning, consistent patterns
- **Target Population**: Average movement patterns, most participants
- **Selection Criteria**: Balanced scores across all dimensions

#### Cluster 1: "Fast & Jerky Movement" (18% of library)
- **Characteristics**: High speed, frequent direction changes, variable timing
- **Target Population**: Rapid, erratic movers
- **Selection Criteria**: High speed (>75th percentile) + high turn variability

#### Cluster 2: "Slow & Smooth Movement" (15% of library)
- **Characteristics**: Low speed, gradual turns, highly consistent
- **Target Population**: Deliberate, controlled movers
- **Selection Criteria**: Low speed (<25th percentile) + high straightness

#### Cluster 3: "Variable Movement" (9% of library)
- **Characteristics**: Mixed patterns, context-dependent behavior
- **Target Population**: Inconsistent or exploratory movers
- **Selection Criteria**: High variability across multiple dimensions

### 3.3 Assignment Validation
```python
def validate_cluster_assignment(participant_clusters, demo_features):
    """Validate cluster assignment quality"""
    
    validation_metrics = {
        'assignment_confidence': calculate_assignment_confidence(distances),
        'feature_alignment': check_feature_cluster_alignment(demo_features, participant_clusters),
        'cluster_coverage': assess_cluster_representation(participant_clusters),
        'fallback_needed': len(participant_clusters) < 2
    }
    
    return validation_metrics
```

## 4. Trajectory Matching Algorithms

### 4.1 Primary Matching: Movement Signature Similarity
The primary trajectory selection method finds pairs of trajectories with similar movement characteristics:

```python
def find_matched_trajectory_pair():
    """Find trajectories with similar movement signatures for target/distractor"""
    
    # Sample from valid trajectories for efficiency
    sample_size = min(100, len(valid_snippet_indices))
    candidate_indices = rng.choice(valid_snippet_indices, size=sample_size, replace=False)
    
    # Extract movement signatures
    signatures = []
    for idx in candidate_indices:
        trajectory = motion_pool[idx]
        signature = get_trajectory_signature(trajectory)
        signatures.append((idx, signature))
    
    # Find best matching pair
    best_score = float('inf')
    best_pair = (None, None)
    
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            similarity_score = calculate_signature_similarity(signatures[i][1], signatures[j][1])
            
            if similarity_score < best_score:
                best_score = similarity_score
                best_pair = (signatures[i][0], signatures[j][0])
    
    return best_pair
```

### 4.2 Movement Signature Extraction
```python
def get_trajectory_signature(trajectory):
    """Extract movement characteristics for trajectory matching"""
    
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    signature = {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }
    
    return signature
```

### 4.3 Signature Similarity Calculation
```python
def calculate_signature_similarity(sig1, sig2):
    """Calculate similarity score between movement signatures (lower = more similar)"""
    
    # Normalized differences for each characteristic
    speed_diff = abs(sig1['mean_speed'] - sig2['mean_speed']) / max(sig1['mean_speed'], sig2['mean_speed'])
    var_diff = abs(sig1['speed_variability'] - sig2['speed_variability']) / max(sig1['speed_variability'], sig2['speed_variability'])
    length_diff = abs(sig1['path_length'] - sig2['path_length']) / max(sig1['path_length'], sig2['path_length'])
    disp_diff = abs(sig1['net_displacement'] - sig2['net_displacement']) / max(sig1['net_displacement'], sig2['net_displacement'])
    
    # Weighted combination
    similarity_score = (
        0.3 * speed_diff +      # Speed is most important
        0.2 * var_diff +        # Variability patterns
        0.2 * length_diff +     # Overall movement amount
        0.15 * disp_diff +      # Path efficiency
        0.15 * np.mean([abs(p1 - p2) for p1, p2 in zip(sig1['speed_percentiles'], sig2['speed_percentiles'])])
    )
    
    return similarity_score
```

## 5. Cluster-Based Selection Algorithm

### 5.1 Fallback Selection Method
When signature matching fails, the system falls back to cluster-based selection:

```python
def sample_from_participant_clusters(n_samples_per_cluster=10):
    """Sample trajectories from participant's assigned clusters"""
    
    if participant_clusters is None:
        return []
    
    selected_snippets = []
    
    for cluster_id in participant_clusters:
        # Find trajectories in this cluster
        cluster_mask = snippet_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        # Filter to valid trajectories only
        valid_cluster_indices = np.intersect1d(cluster_indices, valid_snippet_indices)
        
        # Sample without replacement
        n_available = len(valid_cluster_indices)
        n_to_sample = min(n_samples_per_cluster, n_available)
        
        if n_to_sample > 0:
            sampled = rng.choice(valid_cluster_indices, size=n_to_sample, replace=False)
            selected_snippets.extend(sampled.tolist())
    
    return selected_snippets
```

### 5.2 Selection Priority Hierarchy
```python
def select_trial_trajectories():
    """Hierarchical trajectory selection with multiple fallbacks"""
    
    # Primary: Movement signature matching
    target_idx, distractor_idx = find_matched_trajectory_pair()
    
    if target_idx is not None and distractor_idx is not None:
        selection_method = "signature_matching"
        return target_idx, distractor_idx, selection_method
    
    # Secondary: Cluster-based selection
    available_snippets = sample_from_participant_clusters(n_samples_per_cluster=20)
    
    if len(available_snippets) >= 2:
        selected = rng.choice(available_snippets, size=2, replace=False)
        target_idx, distractor_idx = selected[0], selected[1]
        selection_method = "cluster_based"
        return target_idx, distractor_idx, selection_method
    
    # Tertiary: Random valid trajectories
    if len(valid_snippet_indices) >= 2:
        selected = rng.choice(valid_snippet_indices, size=2, replace=False)
        target_idx, distractor_idx = selected[0], selected[1]
        selection_method = "random_valid"
        return target_idx, distractor_idx, selection_method
    
    # Final fallback: Any trajectories
    target_idx = rng.integers(0, TOTAL_SNIPS)
    distractor_idx = rng.integers(0, TOTAL_SNIPS)
    selection_method = "random_any"
    
    return target_idx, distractor_idx, selection_method
```

## 6. Trajectory Preprocessing and Adaptation

### 6.1 Consistent Smoothing Application
Selected trajectory pairs undergo consistent smoothing to ensure similar temporal characteristics:

```python
def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to trajectory pair"""
    
    def smooth_trajectory(traj, smoothing_factor=0.7):
        smoothed = np.copy(traj)
        for i in range(1, len(traj)):
            smoothed[i] = smoothing_factor * smoothed[i-1] + (1 - smoothing_factor) * traj[i]
        return smoothed
    
    # Apply same smoothing to both trajectories
    smoothed_traj1 = smooth_trajectory(trajectory1)
    smoothed_traj2 = smooth_trajectory(trajectory2)
    
    return smoothed_traj1, smoothed_traj2
```

### 6.2 Real-Time Magnitude Normalization
During trials, trajectory velocities are normalized to match participant movement scale:

```python
def normalize_trajectory_to_participant(trajectory_velocity, participant_velocity):
    """Normalize trajectory velocity to match participant scale"""
    
    # Calculate magnitudes
    traj_magnitude = np.linalg.norm(trajectory_velocity)
    participant_magnitude = np.linalg.norm(participant_velocity)
    
    # Normalize if trajectory has movement
    if traj_magnitude > 0:
        normalized_velocity = trajectory_velocity * (participant_magnitude / traj_magnitude)
    else:
        normalized_velocity = trajectory_velocity
    
    return normalized_velocity
```

## 7. Personalization Effectiveness Metrics

### 7.1 Assignment Quality Measures
```python
def evaluate_personalization_quality(participant_data):
    """Assess effectiveness of personalization algorithms"""
    
    metrics = {
        # Cluster assignment quality
        'cluster_distance_ratio': calculate_cluster_distance_ratio(participant_data),
        'assignment_stability': assess_assignment_consistency(participant_data),
        'cluster_representation': check_cluster_coverage(participant_data),
        
        # Trajectory matching quality
        'matching_success_rate': calculate_matching_success_rate(participant_data),
        'signature_similarity': assess_average_signature_similarity(participant_data),
        'fallback_frequency': calculate_fallback_usage(participant_data),
        
        # Behavioral effectiveness
        'sensitivity_improvement': measure_sensitivity_gain(participant_data),
        'response_consistency': assess_rating_consistency(participant_data),
        'engagement_indicators': measure_task_engagement(participant_data)
    }
    
    return metrics
```

### 7.2 Personalization Impact Analysis
```python
def analyze_personalization_impact(personalized_data, control_data):
    """Compare personalized vs. non-personalized performance"""
    
    impact_analysis = {
        # Agency perception effects
        'agency_rating_variance': compare_rating_variance(personalized_data, control_data),
        'confidence_improvements': compare_confidence_levels(personalized_data, control_data),
        'response_time_effects': compare_response_times(personalized_data, control_data),
        
        # Sensitivity measures
        'control_detection_accuracy': compare_detection_accuracy(personalized_data, control_data),
        'staircase_convergence': compare_convergence_rates(personalized_data, control_data),
        'threshold_stability': compare_threshold_estimates(personalized_data, control_data),
        
        # Individual differences
        'between_subject_variance': compare_individual_differences(personalized_data, control_data),
        'cluster_effect_sizes': calculate_cluster_specific_effects(personalized_data),
        'adaptation_quality': assess_adaptation_effectiveness(personalized_data)
    }
    
    return impact_analysis
```

## 8. Algorithm Performance Optimization

### 8.1 Computational Efficiency
```python
# Pre-computed trajectory signatures for faster matching
trajectory_signatures = {}
for idx in valid_snippet_indices:
    trajectory_signatures[idx] = get_trajectory_signature(motion_pool[idx])

# Efficient cluster membership lookup
cluster_membership = {
    cluster_id: np.where(snippet_labels == cluster_id)[0] 
    for cluster_id in range(K_CLUST)
}

# Cached distance calculations
def get_cached_cluster_distances(demo_features):
    """Use pre-computed centroids for faster assignment"""
    return np.linalg.norm(CLUSTER_CENTROIDS - demo_features.reshape(1, -1), axis=1)
```

### 8.2 Memory Management
```python
def optimize_trajectory_storage():
    """Optimize memory usage for trajectory data"""
    
    # Use memory-mapped arrays for large trajectory data
    motion_pool_mmap = np.memmap('motion_pool.dat', dtype='float32', mode='r', 
                                shape=(TOTAL_SNIPS, SNIP_LEN, 2))
    
    # Cache frequently accessed trajectories
    trajectory_cache = {}
    cache_size = 50  # Maximum cached trajectories
    
    def get_trajectory_with_cache(idx):
        if idx not in trajectory_cache:
            if len(trajectory_cache) >= cache_size:
                # Remove oldest entry
                oldest_key = next(iter(trajectory_cache))
                del trajectory_cache[oldest_key]
            trajectory_cache[idx] = motion_pool_mmap[idx].copy()
        return trajectory_cache[idx]
```

## 9. Personalization Quality Assurance

### 9.1 Runtime Validation
```python
def validate_personalization_runtime(participant_clusters, selected_trajectories):
    """Real-time validation of personalization decisions"""
    
    validation_checks = {
        'valid_cluster_assignment': len(participant_clusters) >= 1,
        'cluster_ids_valid': all(0 <= cid < K_CLUST for cid in participant_clusters),
        'trajectory_selection_success': selected_trajectories[0] is not None,
        'trajectory_indices_valid': all(0 <= idx < TOTAL_SNIPS for idx in selected_trajectories if idx is not None),
        'fallback_appropriate': check_fallback_usage_appropriate(selected_trajectories)
    }
    
    return validation_checks
```

### 9.2 Post-Trial Assessment
```python
def assess_trial_personalization_quality(trial_data, kinematics_data):
    """Evaluate personalization effectiveness for individual trials"""
    
    assessment = {
        # Trajectory appropriateness
        'movement_scale_match': assess_velocity_scale_matching(kinematics_data),
        'smoothness_consistency': evaluate_smoothness_preservation(kinematics_data),
        'temporal_alignment': check_temporal_synchronization(kinematics_data),
        
        # Behavioral indicators
        'engagement_quality': measure_movement_engagement(kinematics_data),
        'response_confidence': evaluate_response_patterns(trial_data),
        'task_understanding': assess_rating_appropriateness(trial_data)
    }
    
    return assessment
```

