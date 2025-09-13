# Module 02: Reduced-Order Simulators

## 1. Rationale & Purpose

To prove the efficacy of the 48CB-Core controller, we need a controlled environment in which to test it at scale. Real fusion experiments are too expensive and slow for initial controller development and validation. These reduced-order simulators provide a fast, cheap, and reproducible "digital twin" of the essential control problems in both magnetic and inertial confinement fusion.

Their purpose is **not** to be perfectly accurate physical models. Instead, their purpose is to be **plausibly realistic control challenges**. They must capture the key cause-and-effect relationships between actuators and instabilities with enough fidelity to demonstrate that the 48CB-Core's control strategies are valid and effective. They are the sandbox in which we will run thousands of experiments to generate the statistical evidence needed to justify a real-world pilot.

## 2. Simulator Descriptions

### 2.1. `TokamakSim-lite`
- **Domain:** Magnetic Confinement Fusion (Tokamaks).
- **Purpose:** To simulate the core challenge of maintaining a stable, high-performance plasma in the face of multiple interacting instabilities, primarily focused on **disruption avoidance**.
- **Key Features:**
    - **State Vector:** A simplified representation of the plasma state, including core parameters (`Ip`, density, temperature), profile information (`q`, `li`), and instability markers (NTM amplitude, locked mode proxy).
    - **Actuator Models:** Simplified models for the effects of RMP coils, ECCD/ECRH heating, neutral beams, and fueling on the plasma state.
    - **Instability Dynamics:** A set of coupled ordinary differential equations (ODEs) that model the growth and saturation of key instabilities. The model is tuned so that certain regions of the state space lead to a "disruption" (a terminal event).
    - **Stochasticity:** Includes options for adding noise to diagnostics and actuator responses to test controller robustness.
- **Implementation:** A configurable, JAX-based ODE solver for speed. The entire simulation is differentiable, allowing for future optimization work.

### 2.2. `ICFSim-lite`
- **Domain:** Inertial Confinement Fusion.
- **Purpose:** To simulate the core challenge of achieving a symmetric implosion, primarily focused on **low-mode asymmetry control**.
- **Key Features:**
    - **State Vector:** Represents the imploding shell's shape using a basis of spherical harmonics (P2, P4, etc.).
    - **Actuator Models:** Models the effect of adjusting laser beam power within cohorts (e.g., the 48-phase cohorts) on the growth of these asymmetries. Includes a model for cross-beam energy transfer (CBET) as a key coupling mechanism.
    - **Instability Dynamics:** A simplified model of Rayleigh-Taylor instability growth, where initial asymmetries are amplified during the implosion.
    - **Diagnostic Model:** Simulates the process of taking a low-resolution X-ray image of the implosion and fitting it to derive the asymmetry coefficients, mimicking the measurement process.
- **Implementation:** A fast, vectorized NumPy/JAX simulation that can run thousands of shots in minutes.

## 3. Role in the MVP

- **Training & Tuning:** While the 48CB-Core is not "trained" in the ML sense, the simulators are used to tune its internal models (e.g., the predictors in the `GameEncoder`).
- **Large-Scale Benchmarking:** We will run thousands of simulated shots with both the 48CB controller and baseline controllers to generate robust statistics on performance KPIs (e.g., disruption rate, asymmetry reduction).
- **Generating Synthetic Datasets:** These simulators will be used to create the synthetic portion of our benchmark datasets, allowing for controlled and repeatable experiments.
- **Demonstration:** They provide a visually compelling way to demonstrate the controller's effectiveness in real-time, showing a plasma that would have disrupted being saved by the 48CB's actions.

## 4. Success Criteria
- `TokamakSim-lite` can simulate a 2-second plasma discharge at 1kHz resolution in under 10 seconds of wall-clock time.
- `ICFSim-lite` can simulate 1,000 implosions in under 60 seconds.
- The simulators must be able to reproduce the qualitative failure modes (disruptions, asymmetric implosions) that are observed in real experiments.
- The simulated actuator responses must be sensitive enough to show a clear difference in outcomes between the 48CB controller and the baseline controllers.
