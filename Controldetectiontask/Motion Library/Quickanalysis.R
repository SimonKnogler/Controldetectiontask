#!/usr/bin/env Rscript
# ——————————————————————————————————————————————
# Quick QC for MotionLibrary data - Multi-file Analysis
# Updated to handle both old and new data formats
# ——————————————————————————————————————————————

# 1) Load necessary libraries
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
if (!requireNamespace("readr", quietly = TRUE)) {
  install.packages("readr")
}
library(dplyr)
library(ggplot2)
library(readr)

# 2) User settings: path to the data directory
data_directory <- "/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/data"

# 3) Constants (must match your experiment)
MAIN_TRIALS        <- 10               # number of trials per participant
DURATION_SEC       <- 5                # seconds per trial snippet
FPS                <- 60               # frames per second sampling
SAMPLES_PER_SNIPPET <- FPS * DURATION_SEC
RADIUS             <- 250              # max allowed distance from center

# 4) Get all CSV files in the directory (exclude analysis files)
all_csv_files <- list.files(data_directory, pattern = "\\.csv$", full.names = TRUE)
# Filter out analysis files that were created by this script
csv_files <- all_csv_files[!grepl("analysis|trial_details", basename(all_csv_files))]
cat("Found", length(csv_files), "participant CSV files to analyze\n\n")

# 5) Function to detect data format and analyze a single file
analyze_file <- function(file_path) {
  cat("Analyzing:", basename(file_path), "\n")
  
  # Check if file is empty
  file_size <- file.size(file_path)
  if (file_size == 0) {
    cat("⚠️  Skipping", basename(file_path), "- file is empty\n")
    return(NULL)
  }
  
  # Read the data with error handling
  tryCatch({
    dat <- read.csv(file_path, stringsAsFactors = FALSE)
  }, error = function(e) {
    cat("⚠️  Skipping", basename(file_path), "- error reading file:", e$message, "\n")
    return(NULL)
  })
  
  # Check if data is empty after reading
  if (nrow(dat) == 0) {
    cat("⚠️  Skipping", basename(file_path), "- no data rows found\n")
    return(NULL)
  }
  
  # Check if this is a participant data file by looking for key columns
  has_trial_col <- "trial" %in% names(dat)
  has_isPractice_col <- "isPractice" %in% names(dat)
  has_frame_col <- "frame" %in% names(dat)
  has_x_col <- "x" %in% names(dat)
  has_y_col <- "y" %in% names(dat)
  
  if (!has_x_col || !has_y_col) {
    cat("⚠️  Skipping", basename(file_path), "- missing x/y coordinates\n")
    return(NULL)
  }
  
  # Determine data format and handle accordingly
  if (has_trial_col && has_frame_col) {
    # New format with trial and frame columns
    cat("  Format: New (trial + frame columns)\n")
    data_format <- "New (trial + frame)"
    dat$trial <- as.numeric(dat$trial)
    dat$frame <- as.numeric(dat$frame)
    
    # Create sampleIndex from frame (assuming 300 frames per trial)
    dat$sampleIndex <- dat$frame %% 300 + 1
    dat$sampleTime <- (dat$sampleIndex - 1) * (1000 / FPS)  # milliseconds
    
  } else if (has_trial_col && has_x_col && has_y_col) {
    # New format with trial, sampleIndex, sampleTime, x, y columns
    cat("  Format: New (trial + sampleIndex + sampleTime + x + y columns)\n")
    data_format <- "New (trial + sampleIndex + sampleTime + x + y)"
    dat$trial <- as.numeric(dat$trial)
    dat$sampleIndex <- as.numeric(dat$sampleIndex)
    dat$sampleTime <- as.numeric(dat$sampleTime)
    dat$x <- as.numeric(dat$x)
    dat$y <- as.numeric(dat$y)
    
  } else if (has_isPractice_col && has_frame_col) {
    # New format with isPractice and frame columns
    cat("  Format: New (isPractice + frame columns)\n")
    data_format <- "New (isPractice + frame)"
    dat$isPractice <- as.numeric(dat$isPractice)
    dat$frame <- as.numeric(dat$frame)
    
    # Create trial and sampleIndex from frame
    dat$trial <- floor(dat$frame / 300)
    dat$sampleIndex <- dat$frame %% 300 + 1
    dat$sampleTime <- (dat$sampleIndex - 1) * (1000 / FPS)  # milliseconds
    
  } else {
    cat("  Format: Unknown - skipping\n")
    return(NULL)
  }
  
  # Extract participant info with error handling
  participant_id <- ifelse(length(unique(dat$participant_id)) > 0 && !is.na(unique(dat$participant_id)[1]), 
                         unique(dat$participant_id)[1], 
                         "Unknown")
  participant_num <- ifelse(length(unique(dat$Participant)) > 0 && !is.na(unique(dat$Participant)[1]), 
                          unique(dat$Participant)[1], 
                          "Unknown")
  
  # Filter out practice trials (trial = 0) for main analysis
  main_trials_dat <- dat %>% filter(trial > 0)
  
  if (nrow(main_trials_dat) == 0) {
    cat("⚠️  Skipping", basename(file_path), "- no main trials found\n")
    return(NULL)
  }
  
  # Calculate trial statistics
  trials_completed <- length(unique(dat$trial))
  trials_expected <- 10  # Fixed expectation for 10 trials (0-9)
  completion_rate <- (trials_completed / trials_expected) * 100
  
  # Calculate motion and speed scores
  motion_score <- mean(sqrt(diff(dat$x)^2 + diff(dat$y)^2), na.rm = TRUE)
  speed_score <- mean(sqrt(diff(dat$x)^2 + diff(dat$y)^2) / (1000 / FPS), na.rm = TRUE)
  
  # Check for out-of-bounds samples (assuming radius of 250)
  out_of_bounds <- sum(sqrt(dat$x^2 + dat$y^2) > 250, na.rm = TRUE)
  
  # Determine if participant had adequate movement (motion score > 0.1)
  adequate_movement <- motion_score > 0.1
  
  # Create result data frame
  result <- data.frame(
    File = basename(file_path),
    Trials_Completed = trials_completed,
    Trials_Expected = trials_expected,
    Completion_Rate = round(completion_rate, 1),
    Motion_Score = round(motion_score, 3),
    Speed_Score = round(speed_score, 3),
    Out_of_Bounds_Samples = out_of_bounds,
    Adequate_Movement = adequate_movement,
    Format = data_format,
    stringsAsFactors = FALSE
  )
  return(result)
}

# 6) Analyze all files
cat("Analyzing", length(csv_files), "files...\n")
results <- lapply(csv_files, analyze_file)
results <- Filter(Negate(is.null), results)  # Remove NULLs

if (length(results) > 0) {
  summary_df <- do.call(rbind, results)
  # Write summary to CSV
  write.csv(summary_df, file = file.path(data_directory, "motion_library_analysis_summary.csv"), row.names = FALSE)
  cat("\nSummary written to:", file.path(data_directory, "motion_library_analysis_summary.csv"), "\n")
} else {
  cat("No valid participant files found.\n")
}

# Create trial-level analysis
trial_details <- do.call(rbind, lapply(results, function(r) {
  if (!is.null(r$trial_metrics)) {
    r$trial_metrics$file <- r$file
    r$trial_metrics$participant_id <- r$participant_id
    r$trial_metrics
  }
}))

if (nrow(trial_details) > 0) {
  trial_output_file <- file.path(data_directory, "motion_library_trial_details.csv")
  write.csv(trial_details, trial_output_file, row.names = FALSE)
  cat("📊 Trial-level details saved to:", trial_output_file, "\n\n")
}

cat("✅ Analysis complete!\n")
