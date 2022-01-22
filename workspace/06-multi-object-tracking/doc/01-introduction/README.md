# Prelogue

---

## Introduction

### Problem Definition

`Multi-Object Tracking` is the sequential processing of noisy sensor measurements to determine:

* the number of dynamic objects

* each dynamic object's state

At each timestamp, the number of measurements per object depends on the sensor, the detector and the object characteristics.

### Overview

#### Workflow

<img src="images/01-tracking-after-detection-workflow.png" alt="Tracking-After-Detection Workflow" width="100%">

#### Case Analysis:

<img src="images/02-case-analysis--camera.png" alt="Workflow--Camera" width="100%">

### Different Types of Tracking

#### Point Object Tracking

<img src="images/03-a--point-object-tracking.png" alt="Point Object Tracking" width="100%">

#### Extended Object Tracking

<img src="images/03-b--extended-object-tracking.png" alt="Extended Object Tracking" width="100%">

#### Group Object Tracking

<img src="images/03-c--group-object-tracking.png" alt="Group Object Tracking" width="100%">

#### Tracking with Multi-Path Propagation

<img src="images/03-d--tracking-with-multi-path-propagation.png" alt="Tracking with Multi-Path Propagation" width="100%">

#### Tracking with Unresolved Objects

<img src="images/03-e--tracking-with-unresolved-objects.png" alt="Tracking with Unresolved Objects" width="100%">

### Challenges

<img src="images/04-challenges.png" alt="Challenges in MOT" width="100%">

The challenges in `Multi-Object Tracking` can be summarized as follows:

* restricted field-of-view(FOV).

* stochastic object birth-death in FOV.

* sensor occlusion.

* detector error.

* data association error.

<img src="images/05-detector-error--a-miss-detection.png" alt="Miss Detection" width="100%">

<img src="images/05-detector-error--b-false-detection.png" alt="False Detection" width="100%">

<img src="images/05-detector-error--c-data-association.png" alt="Data Association" width="100%">

---

## Bayesian Filtering

### Overview

Process `the sequential measurements` and estimate `the posterior density of the state of interest`.

<img src="images/06-bayesian-filtering-a-workflow.png" alt="Bayesian Filtering: Workflow" width="100%">

### Motion Models

<img src="images/06-bayesian-filtering-b-problem-statement-1-motion-models.png" alt="Bayesian Filtering: Motion Models" width="100%">

Example: Vehicle Agent Motion Model

<img src="images/06-bayesian-filtering-b-problem-statement-1-vehicle-agent-model.png" alt="Vehicle Agent Motion Model" width="100%">

### Measurement Models

<img src="images/06-bayesian-filtering-b-problem-statement-2-measurement-models.png" alt="Bayesian Filtering: Measurement Models" width="100%">

Example: Radar & Lidar Measurement Model

<img src="images/06-bayesian-filtering-b-problem-statement-2-radar-model.png" alt="Range-Bear Measurement Model" width="100%">

### Sequential Processing Workflow

<img src="images/06-bayesian-filtering-c-workflow.png" alt="Sequential Processing Workflow" width="100%">

Multi-Object Tracking Sequential Processing Workflow

* Chapman-Kolmogorov Prediction

* Predicted Measurement Likelihood for Association

* Bayesian Measurement Update

### Output

<img src="images/06-bayesian-filtering-d-output.png" alt="Module Output" width="100%">

### Evaluation

<img src="images/06-bayesian-filtering-e-evaluation.png" alt="Performance Evaluation" width="100%">

---

## Kalman Filtering

### Introduction

`Kalman Filter` is the minimum mean squared error (MMSE) when:

* `motion` & `measurement` models are both linear

* noises are `Gaussian`

* initial state is `Gaussian`

<img src="images/07-kalman-filter-a-introduction.png" alt="Kalman Filter: MMSE Estimator for Gaussian Noise" width="100%">

### Processing Workflow

<img src="images/07-kalman-filter-b-processing-workflow.png" alt="Prediction - Measurement Likelihood - Update Workflow" width="100%">

### Non-Linear Models

<img src="images/07-kalman-filter-c-non-linear-estimation.png" alt="Non-Linear Measurements" width="100%">

### Uncertainty Propagation

<img src="images/07-kalman-filter-d-uncertainty-propagation.png" alt="Information Perspective -- Uncertainty Propagation" width="100%">

---

## Assumed Density Tracking

`Assumed Density Tracking` is the foundation of practical MOT system:

* it helps to transform `the Chapman-Kolmogorov prediction` and `the Bayesian update` into closed-form parameter update.

* it ensures the predictable complexity of the actual MOT system.

<img src="images/08-assumed-density-tracking--a-gaussian-density.png" alt="Assumed Density Tracking: Gaussian" width="100%">

<img src="images/08-assumed-density-tracking--b-update-process.png" alt="Assumed Density Tracking: Sequential Param Update" width="100%">

<img src="images/08-assumed-density-tracking--c-advantages.png" alt="Assumed Density Tracking: Advantages" width="100%">

<img src="images/08-assumed-density-tracking--d-approximation.png" alt="Assumed Density Tracking: Approximation" width="100%">


