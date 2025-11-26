# **Target Network Power Indexes**

The **target-network-power-index** repository implements two target-based
extensions of the classical **Network Power Index (NPI)** and **Network Power Flow (NPF)**:

- **Target Network Power Index (T-NPI)**
- **Target Network Power Flow (T-NPF)**

These extensions measure how effective control concentrates on a **specific target firm**
within a complex ownership network, capturing both **direct and indirect** propagation
of corporate influence.

The framework also provides an **interactive stress-testing environment** designed
to evaluate how resilient control over a target is when shareholder stakes change
incrementally in the network.

---

## **Motivation**

Modern ownership systems have become multi-layered and highly interconnected
due to cross-border acquisitions, institutional investors, index funds, and
pyramidal corporate structures.  

Classical NPI and NPF are among the best tools for measuring system-wide
control concentration.  
However, because they operate **globally**, they cannot show how influence
accumulates around a **specific strategic firm**.

This repository introduces:

- **T-NPI**: a target-focused extension tracking how indirect ownership chains amplify control.
- **T-NPF**: a flow-based propagation measure centered on a target.

Together, these metrics allow analysts to:

- identify ultimate owners,
- detect hidden influence routes,
- quantify control resilience,
- evaluate how concentrated or dispersed acquisitions affect governance.

---

## **Features**

✔ Calculation of NPI, NPF, T-NPI, T-NPF  
✔ Stress-testing module for incremental acquisition scenarios  
✔ Automatic network reduction around the target  
✔ Visualization utilities  
✔ Adaptable to any ownership graph (NetworkX compatible)  
✔ Four alternative scenarios for allocating missing ownership shares  
✔ Example scripts ready to run  

---

# **Directory Structure**

```
target-network-power-index/
│
├─ network_power_indexes/
│ ├─ calculate_index.py # Core NPI / NPF / T-NPI / T-NPF methods
│ ├─ utils.py # Helper functions
│ ├─ plots.py # Plotting utilities
│ ├─ simulations.py # Stress testing modules
│
├─ example/
│ ├─ calculate_network_index.py # Computes indexes for any graph (4 scenarios)
│ ├─ first_simulation.py # Stress test: incremental increases by top shareholders
│ ├─ second_simulation.py # Stress test: coordinated increases by a conglomerate
│
└─ README.md
```

---

# **Example Scripts**

The directory **example/** contains three fully-functional scripts demonstrating how
to apply the framework.

---

## **1. `calculate_indexes.py` — Compute all indexes for any network**

This script loads an ownership graph (any NetworkX graph) and computes:

- NPI  
- NPF  
- Target-NPI  
- Target-NPF  

using **four scenarios** for distributing **unobserved / missing shares**.

### **The Four Scenarios**

Following standard literature, the script implements:

### **Scenario 1 — Equal Redistribution (Berle & Means; Mizuno)**  
Missing shares are redistributed **equally across all known shareholders**.  
Interprets unknown investors as part of the dispersed “floating” capital.

### **Scenario 2 — Leech Relaxed Scenario**  
Small unobserved shareholders act **independently** and do not affect control.  
They are **excluded** from the effective control configuration.

### **Scenario 3 — Coordinated Proportional Redistribution**  
Private shareholders coordinate.  
Residual ownership is redistributed **proportionally** to their observed stakes.

### **Scenario 4 — Coordinated Equal Redistribution**  
Shares are allocated **equally** across private shareholders acting as a block.

These assumptions provide robustness checks and allow analysts to evaluate how
results change depending on how unobserved shareholders behave.

---

## **2. `first_simulation.py` — Stress test (increment top shareholders)**

This script:

1. Identifies the direct top shareholders of the target  
2. Increases each shareholder’s stake **incrementally and independently** (e.g. +2% per step)  
3. Recomputes the NPI / T-NPI / control structure at each iteration  

This is useful to evaluate:

- takeover risk from concentrated acquisitions  
- sensitivity of the target’s control structure  
- thresholds where ultimate control shifts  

---

## **3. `second_simulation.py` — Stress test (coordinated conglomerate increase)**

This simulation captures **coordinated acquisition strategies**:

1. Select a firm inside a corporate group  
2. Identify all companies belonging to the same group  
3. Increase all their stakes **simultaneously**  
4. Compute NPI, T-NPI and control shifts  

This models scenarios such as:

- corporate groups acting collectively  
- asset managers increasing positions across multiple ETFs/funds  
- dispersed but coordinated accumulation strategies  

Practical results show that:

- **Concentrated acquisitions** may fail to change ultimate control  
- **Dispersed coordinated acquisitions** can erode state control dramatically  

---

# **Research Foundations**

This framework is rooted in the following methodological concepts:

- Ownership networks represented as a weighted directed graph  
- Multi-layer propagation of control through indirect shareholding paths  
- Identification of ultimate owners  
- Stress testing the resilience of control in the presence of structural changes  

The representation follows:

- \( V \): nodes as firms, funds, individuals  
- \( E \subseteq V \times V \): ownership edges  
- \( w(e) \in [0,100] \): percentage owned  

This allows modeling:

- direct control  
- blockholder control  
- pyramidal control  
- multi-tier propagation of influence  

---

# **Case Study**

The provided methodology was applied to a Autogenerated firm with:

- 20% held by the state (Company 1) 
- 70.2% free float

Stress tests revealed:

- state control is **robust** to concentrated acquisitions  
- but **vulnerable** to dispersed coordinated strategies  
- intermediate entities significantly **amplify** the influence of private groups  

---

# **How to Use**

```python
python example/calculate_network_index.py
python example/first_simulation.py
python example/second_simulation.py
```
