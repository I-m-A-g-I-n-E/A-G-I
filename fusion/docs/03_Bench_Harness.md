# Module 03: Benchmarking Harness

## 1. Rationale & Purpose

A novel controller is only as credible as the rigor of its evaluation. The purpose of this Benchmarking Harness is to provide a standardized, automated, and unbiased framework for comparing the performance of the 48CB-Core controller against relevant baselines.

This is the scientific backbone of the MVP. It ensures that every claim we make is backed by reproducible, statistically significant evidence. It is designed to be transparent and auditable, allowing a third party (like a potential investor or a facility's science team) to scrutinize our methodology and replicate our results. The harness automates the process of running experiments, collecting data, computing metrics, and generating reports, which is essential for achieving the scale and rigor required.

## 2. Architecture & Components

The harness is a collection of Python scripts and notebooks orchestrated by a workflow manager (like `prefect` or `luigi`).

### 2.1. Experiment Definition
- **Purpose:** To define the exact configuration for a given benchmark run in a clear, text-based format (e.g., YAML or JSON).
- **Contents:**
    - The dataset to be used (historical or synthetic).
    - The controller to be tested (e.g., `48CB_v1.0`, `PID_baseline`, `MPC_baseline`).
    - The specific version of the simulator (if applicable).
    - Ablation settings (e.g., `canonicalization_disabled`).
    - The number of trials or shots to run.
    - The random seed for reproducibility.

### 2.2. Baseline Controllers
- **Purpose:** To provide credible points of comparison for the 48CB-Core.
- **Implementations:**
    - **`PID_baseline`:** A set of well-tuned Proportional-Integral-Derivative controllers, representing the current industry standard for many real-time control tasks in fusion.
    - **`MPC_baseline`:** A Model Predictive Control baseline that uses a simplified model of the plasma to plan control actions. This represents a more advanced "conventional" approach.
    - **`NoControl_baseline`:** A passive run to establish the baseline instability rate.
    - *(Optional)* **`ML_baseline`:** A state-of-the-art machine learning predictor (e.g., a recurrent neural network) for disruption prediction, coupled with a simple action policy.

### 2.3. Key Performance Indicator (KPI) Library
- **Purpose:** To compute the specific, pre-defined metrics that quantify controller performance.
- **Magnetic Confinement KPIs:**
    - **Disruption Rate:** Percentage of shots that end in disruption.
    - **Time-to-Warning:** Time between the first alert from a disruption predictor and the actual event.
    - **AUC (Area Under Curve):** For ROC curves, measuring the trade-off between true positives and false positives for predictors.
    - **Controller Thrash:** The variance of the actuator signals, measuring how "nervous" the controller is.
    - **Confinement Proxy (H-factor):** A measure of how well the controller maintains plasma energy.
    - **ELM Peak Proxy:** The amplitude of the largest ELM-like events in the simulation.
- **Inertial Confinement KPIs:**
    - **Asymmetry Norms:** Root-mean-square (RMS) values for low-mode asymmetries (`|P2|`, `|P4|`).
    - **Correction Cost:** The number and magnitude of corrective actions taken.

### 2.4. Reporting Engine
- **Purpose:** To automatically generate the plots, tables, and dashboards that summarize the results of a benchmark run.
- **Outputs:**
    - **Comparison Plots:** Box plots and violin plots comparing the KPI distributions for 48CB vs. baselines.
    - **Time-Series Overlays:** Plots of specific shots showing the controller's actions and the plasma's response over time.
    - **Statistical Tables:** Tables of mean KPI values, standard deviations, and confidence intervals.
    - **Interactive Dashboards:** Web-based dashboards (e.g., using Plotly/Dash or Streamlit) for interactively exploring the results.

## 3. Success Criteria
- The harness can run a 1,000-shot benchmark campaign and generate a full report with a single command.
- All results, including plots and tables, must report confidence intervals to demonstrate statistical significance.
- The entire process, from experiment definition to final report, must be fully reproducible from a given random seed.
- The baseline controllers must be tuned to be competitive, ensuring that any performance gains from the 48CB are meaningful.
