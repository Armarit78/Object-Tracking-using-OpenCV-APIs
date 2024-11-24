
# Object Tracking using OpenCV APIs

## Overview

This project demonstrates object tracking using various OpenCV APIs, focusing on selecting and evaluating the best tracking algorithm. The study is based on practical tests designed to assess algorithm performance in different scenarios, including varying frame rates, object sizes, and tracking challenges like pedestrian and vehicle movements.

## Project Goals

1. Reproduce an open-sourced object tracking algorithm using OpenCV APIs.
2. Implement tracking on both video input and real-time video streams.
3. Evaluate the performance of different algorithms to determine the best choice for object tracking.

---

## Tests Conducted

### 1. Pedestrian Tracking at Different FPS

- **Objective:** Evaluate the ability of each tracker to handle varying frame rates.
- **Methodology:**
  - Videos of pedestrians were tested at different FPS values (e.g., 15 FPS, 30 FPS, 60 FPS, 120 FPS).
  - **Metrics collected:**
    - Average FPS during tracking.
    - CPU usage.
    - Detection stability.

---

### 2. Minimum and Maximum Object Sizes

- **Objective:** Test the tracker's ability to handle objects of varying sizes.
- **Methodology:**
  - Two videos were used:
    - Small object tracking (e.g., faces).
    - Large object tracking (e.g., entire bodies).
  - Each tracker was tested for its success in maintaining a lock on the object without losing precision.

---

### 3. Vehicle Tracking

- **Objective:** Test real-world scenarios involving cars in motion.
- **Methodology:**
  - A video of moving cars was used.
  - The focus was on stability during high-speed movements and object overlaps.

---

## Chosen Algorithm

Based on the tests conducted, the **CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)** tracker was selected as the best-performing algorithm for this project due to its:

- Robustness to object size variations.
- Stability across different FPS.
- High precision in maintaining the tracking box during occlusions and fast movements.






   
   
