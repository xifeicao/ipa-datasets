# Paper Framework: ML Prediction of Particle Fracture in Battery Electrode Calendering

**Target journal:** Journal of Power Sources (primary) | Electrochemistry Communications (letter option)
**Date drafted:** 2026-06-01
**Author:** Xifei Cao, University of Michigan

---

## Validated Data Findings (EDA Results)

Before the outline, the key empirical facts established from the IPA granulation dataset (720 runs):

| Finding | Value | Implication for paper |
|---|---|---|
| RF overall test R² (physics-aug) | **0.920** | Model is viable |
| Dominant feature | **peak_pressure_MPa (70.3%)** | Physics-consistent result |
| 2nd feature | material_enc (11.5%) | Material class matters |
| LOMO R² – MCC_101 left out | 0.679 | Moderate transfer to plastic |
| LOMO R² – Mannitol_SD left out | **0.402** | Poor transfer to brittle/NMC → motivates DEM |
| Brittle material avg fines | 28.5% vs. 20.4% (plastic) | Brittle = more fracture |
| Pressure quintle trend | Monotonic: Q1=37.5%, Q5=17.8% fines | Higher compact quality reduces post-mill fines |

**Core paper argument:**
IPA pharmaceutical data provides a strong within-dataset ML model (R²=0.92), but it generalizes poorly to brittle materials (R²=0.40 for Mannitol_SD = NMC analog). This gap motivates adding battery-specific ARTISTIC DEM data as a supplement.

---

## Proposed Title

**Full paper:**
> *Data-Driven Prediction of Particle Fracture During Electrode Calendering: Bridging Pharmaceutical Roll Compaction Data and Battery-Specific DEM Simulation with Machine Learning*

**Short letter version:**
> *Why Pharmaceutical Compaction Data Fails to Transfer to Battery Electrode Calendering — and How DEM Fills the Gap*

---

## Section 1: Introduction (~600 words)

### 1.1 Hook and motivation
Battery electrode calendering is a critical manufacturing step that determines electrode porosity, tortuosity, and ultimately cell energy density and cycle life. However, excessive calendering pressure induces particle fracture in NMC cathode active material — a brittle ceramic with fracture toughness ~0.5–1.5 MPa·m^0.5 — generating microcracks that accelerate capacity fade, increase interfacial resistance, and compromise mechanical integrity of the electrode.

*Key citations needed:*
- NMC cracking during calendering: Müller et al. (2021), Xu et al. (2022)
- Calendering effect on porosity/tortuosity: Haselrieder et al. (2013)
- DEM simulation of electrode calendering: Bormashenko / ARTISTIC group papers

### 1.2 Current state and gap
Existing approaches to calendering optimization are:
- **Empirical trial-and-error**: time-consuming, material-wasteful
- **DEM/FEM simulation** (ARTISTIC, Abaqus GTN): accurate but expensive (hours per run)
- **No ML-predictive framework** exists that directly maps process parameters → particle fracture risk for battery materials

The pharmaceutical industry has developed systematic roll compaction datasets (IPA datasets, 720–3000 runs) covering the relationship between specific compaction force (SCF), roll geometry, and material deformation class → ribbon density and granule fines fraction. These datasets share physical mechanisms with battery electrode calendering (Hertz contact mechanics, pressure-driven densification). However, **it is unknown whether models trained on pharmaceutical data transfer to battery-specific materials**.

### 1.3 Contributions (bullet-point for reviewers)
1. **Transfer learning audit**: Quantify how well pharmaceutical roll compaction ML models generalize to brittle-class materials (NMC analog: Mannitol_SD) using leave-one-material-out cross-validation. Finding: R² = 0.40 — poor, motivating battery-specific data.
2. **ARTISTIC DEM fracture extension**: Generate ~100 ARTISTIC 2D DEM calendering runs spanning pressure 20–200 MPa, varying cohesion parameters. Measure bond-breakage count as a battery-native fracture metric.
3. **Multi-source ML framework**: Combine IPA pharmaceutical data + ARTISTIC DEM data in a physics-augmented Random Forest / XGBoost. Demonstrate improved R² for brittle-material fracture prediction.
4. **Design space map**: Generate 2D contour of (SCF × gap_width) → fracture risk for NMC-like materials. Identify fracture-safe operating windows.
5. **SHAP mechanistic insight**: Confirm peak nip pressure as the dominant fracture driver (70% importance); quantify material class effect.

---

## Section 2: Background (~400 words)

### 2.1 Roll compaction physics
The Johanson (1965) rolling theory governs the relationship between roll geometry, feed properties, and peak nip pressure:

```
Peak pressure P_peak ∝ SCF / (gap × roll_diameter^0.5) × f(nip_angle, friction)
Densification factor DF = drawn-in width / gap
```

In the IPA datasets, these are pre-computed: `peak_pressure_MPa`, `densification_factor`, `nip_angle_deg`.

For battery electrode calendering, the same Johanson framework applies, with:
- Roll gap: 20–100 µm (vs. 1–6 mm for pharma ribbons)
- Pressure range: 50–200 MPa (overlapping pharma dataset range: 6–175 MPa)
- Material: NMC (~150–200 GPa elastic modulus, brittle) vs. MCC (plastic) or Mannitol (brittle)

### 2.2 Particle fracture mechanisms
- **Hertz contact fracture**: When inter-particle contact stress exceeds fracture toughness K_IC, cracks nucleate at contact points. Critical for brittle particles.
- **In pharmaceutical granulation**: post-compaction milling breaks ribbons into granules; fines fraction < 100 µm is the fracture proxy (higher fines = weaker, more fragmented ribbon).
- **In battery calendering**: NMC particles crack internally during compression. No milling step — fracture occurs in situ. Requires DEM contact-force tracking or in-situ CT to measure.

### 2.3 ARTISTIC DEM framework
The ARTISTIC project (Brussels/Imperial/Warwick) simulates battery electrode manufacturing via LIGGGHTS/LAMMPS DEM. The 2D simplified model (`index.html`) tracks AM-CBD bond cohesion and reports broken bond counts during calendering — a direct proxy for interfacial and particle-level fracture events. Key parameters:
- `pressure_MPa` (platen servo setpoint)
- `AM_cohesion_scale` (AM-AM cohesion strength)
- `CBD-AM_cohesion` (interfacial strength)
- Output: broken bond count per run

---

## Section 3: Data Sources and Feature Engineering (~800 words)

### 3.1 IPA pharmaceutical datasets

**Primary: Granulation dataset** (720 runs, 24 columns)

| Input category | Features |
|---|---|
| Process parameters | `scf_kN_per_cm`, `gap_width_mm`, `roll_speed_rpm`, `feed_screw_rpm` |
| Equipment design | `roll_diameter_mm`, `roll_width_mm`, `roll_surface`, `sealing_system` |
| Material class | `material`, `deformation_type`, `true_density_gcc` |
| Intermediate physics | `peak_pressure_MPa`, `densification_factor`, `nip_angle_deg` |
| Fracture-proxy targets | `fines_fraction_pct`, `granule_d50_um`, `ribbon_rel_density`, `ribbon_porosity` |

**Secondary: RC dataset** (600 runs, 11 columns) — DOE focused on force/speed/gap/density relationship via Johanson + Heckel equations. Used as independent validation.

**Key EDA findings (Figure 2):**
- Brittle materials (Mannitol_SD) produce 40% more fines than plastic (MCC_101): 28.5% vs. 20.4%
- `peak_pressure_MPa` is the strongest predictor (Pearson r = -0.755 with fines_fraction, meaning higher pressure → better compact → fewer fines in pharma context)
- Fines decreases monotonically from Q1 pressure (37.5%) to Q5 (17.8%) across all material classes

*Note: In pharma, higher SCF → stronger compact → fewer milling fines. In battery context, higher pressure → particle internal fracture increases. This inversion is the central domain-transfer challenge.*

### 3.2 ARTISTIC 2D DEM data generation

Generate N=100 runs via scripted parameter sweeps in the 2D DEM (`index.html` automation or Python-controlled DOM):

| Parameter | Range | Steps |
|---|---|---|
| `target_pressure_MPa` | 20, 50, 80, 120, 160, 200 | 6 levels |
| `CBD-CBD_cohesion` | 0.04, 0.08, 0.12, 0.16 | 4 levels |
| `CBD-AM_cohesion` | 0.08, 0.12, 0.16, 0.20 | 4 levels |
| `AM_cohesion_scale` | 0.00, 0.05, 0.10 | 3 levels |

Output per run:
- `broken_bond_count`: total AM-CBD/AM-AM broken bonds at run end
- `final_porosity`: porosity after target pressure reached
- `thickness_reduction_pct`: compaction strain

These 100 DEM runs represent battery-specific brittle-material fracture data. The `broken_bond_count` is a direct particle-level fracture metric absent from the IPA pharmaceutical datasets.

### 3.3 Feature engineering and domain mapping

**Categorical encoding:**
- `roll_surface`: smooth=0, knurled=1
- `sealing_system`: side=0, rim_rolls=1
- `material` → `deformation_class`: plastic=0, mixed=1, brittle=2

**Domain mapping (pharma → battery):**
- NMC active material ↔ brittle deformation class (Mannitol_SD analog)
  - Justification: Both are ceramic/crystalline with similar fracture-dominated failure
  - Elastic modulus: Mannitol ~20 GPa, NMC ~150 GPa (different scale, but same deformation class)
- CBD binder domain ↔ plastic deformation class
- Calendering pressure range: pharma 6–175 MPa ↔ battery 50–200 MPa (overlapping)

**Physics-augmented feature set (11 features for the main model):**
```
[scf_kN_per_cm, gap_width_mm, roll_speed_rpm, roll_diameter_mm, roll_width_mm,
 material_enc, roll_surface_enc, sealing_enc,
 peak_pressure_MPa, densification_factor, nip_angle_deg]
```

---

## Section 4: Machine Learning Framework (~700 words)

### 4.1 Model architecture

**Two-stage physics-augmented approach:**

```
Stage 1 — Intermediate physics predictor
  Input:  [scf, gap, roll_speed, roll_diameter, roll_surface, material]
  Output: [peak_pressure_MPa, densification_factor, nip_angle_deg]
  Model:  XGBoost regressor (3 independent models)
  Role:   Compute physics intermediates for DEM runs where geometry → pressure 
          is not pre-calculated

Stage 2 — Fracture predictor
  Input:  all 11 physics-augmented features
  Output: [fines_fraction_pct, broken_bond_count (DEM), final_porosity]
  Model:  XGBoost regressor
```

The two-stage architecture mirrors the physical causal chain:
process parameters → nip pressure → particle fracture

### 4.2 Baselines and comparison

| Model | Features | Expected test R² |
|---|---|---|
| Linear Regression | process only | 0.906 (confirmed) |
| Random Forest | process only | 0.910 (confirmed) |
| Random Forest | physics-augmented | 0.920 (confirmed) |
| XGBoost | physics-augmented | ~0.93 (to be confirmed) |
| XGBoost + DEM data | physics-augmented + battery DEM | goal: R² > 0.80 for brittle |

### 4.3 Training protocol

- Combined dataset: 720 (granulation) + 600 (RC) + 100 (ARTISTIC DEM) = 1420 rows
- 80/20 train-test split, stratified by material class
- Leave-one-material-out (LOMO) CV: key result is Mannitol_SD R² rising from 0.40 (no DEM) to target >0.70 (with DEM)
- Hyperparameter tuning: Optuna Bayesian optimization (n_trials=100)
- Software: Python 3.11, scikit-learn 1.4, XGBoost 2.0, SHAP 0.45, matplotlib 3.8

### 4.4 SHAP interpretability

SHAP (SHapley Additive exPlanations) values for each feature:

- **Beeswarm plot**: shows direction and magnitude of each feature's contribution across all samples
- **Dependence plot**: `peak_pressure_MPa` vs. SHAP value → identify fracture onset threshold
- Expected result: peak_pressure threshold at ~80–120 MPa for brittle materials (Hertz contact fracture onset)

---

## Section 5: Results (~1200 words)

### 5.1 Model performance (Table 1)

Tabulate R², RMSE for all models × feature sets × target variables.

**Key numbers to report:**
- Physics-aug RF on granulation: R²=0.920, RMSE=2.56% fines
- LOMO Mannitol_SD (no DEM): R²=0.402, RMSE=6.14% → **poor brittle transfer**
- LOMO Mannitol_SD (with DEM): R²=TBD → demonstrate improvement
- Predicted vs. actual scatter plot (Figure 4): tight cluster around diagonal for overall dataset

### 5.2 Feature importance (Figure 5: SHAP beeswarm)

Confirmed ranking from RF:
1. `peak_pressure_MPa`: 70.3% importance
2. `material_enc` (deformation class): 11.5%
3. `sealing_enc`: 8.0%
4. `roll_speed_rpm`: 4.4%
5. `scf_kN_per_cm`: 1.8% (note: subsumed by peak_pressure)

**Narrative:** Peak nip pressure explains 70% of fracture variance. This is physically consistent — Hertz contact stress drives particle deformation and fracture. SCF alone (1.8%) is not the direct driver; rather, it acts through peak_pressure (modulated by gap width and roll diameter). Material deformation class is the second driver — brittle materials fracture at lower Hertz stress thresholds.

### 5.3 Brittle material domain gap (Figure 6)

LOMO CV results — compare before and after adding DEM data:
- Mannitol_SD alone predicts → R² = 0.40 (pharma data) → R² = XX (pharma + DEM)
- Visual: scatter plot for Mannitol_SD predictions, colored by source (pharma vs. DEM)
- Key interpretation: pharmaceutical-to-battery transfer requires brittle-material-specific data

### 5.4 Calendering design space map (Figure 7)

2D contour: `scf_kN_per_cm` (x-axis) × `gap_width_mm` (y-axis) → `fines_fraction_pct`
- For brittle material class (= NMC analog)
- Fixed: roll_diameter=250mm, roll_speed=15rpm, knurled surface
- Overlay: target porosity iso-contour (e.g., 30% porosity line)
- Overlay: fracture-safe threshold line (e.g., fines < 25%)
- **Safe operating window**: low-right region (moderate SCF, larger gap → moderate pressure, good porosity, low fracture)

### 5.5 ARTISTIC DEM validation (Figure 8)

Scatter: XGBoost predicted fines_fraction_pct vs. ARTISTIC DEM broken_bond_count (normalized)
- Spearman rank correlation r_s > 0.80 expected (DEM fracture count should rank-order consistently with pharmaceutical fines proxy)
- This validates that the pharmaceutical fines_fraction is a meaningful fracture proxy for the battery context

---

## Section 6: Discussion (~700 words)

### 6.1 Physical interpretation of peak_pressure dominance
The Johanson model shows that nip pressure is the product of SCF, gap inverse, densification factor, and friction-dependent nip angle. SCF alone is insufficient — a 10 kN/cm SCF at 2mm gap creates very different pressure than at 5mm gap. The ML model learns this interaction correctly by weighting peak_pressure (computed from all inputs together) most heavily. This supports using physics-augmented features over raw process parameters.

For battery calendering: the implication is that specifying SCF targets alone is insufficient. Engineers should track the actual nip pressure, which depends on electrode porosity, particle size distribution, and roll gap simultaneously.

### 6.2 Why brittle material transfer fails (LOMO R² = 0.40)
The pharmaceutical dataset has only 240 Mannitol_SD (brittle) runs out of 720. More importantly, the brittle deformation regime involves fracture-dominated failure that produces qualitatively different granule size distributions than plastic deformation. The ML model trained on plastic-dominated data underestimates the sensitivity of brittle materials to pressure variations near the fracture threshold. This mirrors the challenge in battery electrode calendering: MCC-calibrated models will underpredict NMC fracture.

### 6.3 DEM data as domain bridge
The ARTISTIC DEM bond-breakage count captures particle-level cohesive fracture directly, providing battery-material-specific fracture signatures that complement the pharmaceutical fines_fraction proxy. Key differences:
- Pharma fines: post-compaction milling fracture (aggregate scale)
- DEM bond breakage: in-situ cohesion failure (particle contact scale)
- Both reflect the same underlying Hertz-contact-driven fracture physics → allows cross-calibration

### 6.4 Limitations
1. IPA datasets are fully synthetic (generated from physical models, not physical experiments) — absolute fines_fraction values may not match real pharmaceutical compactors
2. ARTISTIC 2D DEM is simplified (2D geometry, browser-based) vs. full 3D LIGGGHTS simulation
3. NMC ↔ Mannitol_SD analogy is approximate; elastic moduli differ by ~10×
4. CBD binder crushing (Abaqus crushable foam model) not included — future work

### 6.5 Future work
1. Incorporate Abaqus GTN model outputs (damage parameter D) as an explicit particle fracture target for NMC → supervised learning on FEM data
2. Include electrode microstructure inputs from CT imaging (particle size distribution, coordination number) → graph neural network approach
3. Validate against experimental calendering data (calendar pressure sensor + post-calendering CT)
4. Extend to graphite anode calendering (plastic deformation, different fracture mechanisms)

---

## Section 7: Conclusion (~200 words)

1. Established that IPA pharmaceutical roll compaction datasets provide a viable ML framework for predicting compaction-induced particle fracture (R²=0.92 overall), but **generalize poorly to brittle NMC-analog materials (R²=0.40 in leave-one-material-out CV)**.

2. Demonstrated that ARTISTIC DEM bond-breakage data supplements the pharmaceutical datasets by providing battery-specific brittle fracture signatures, improving LOMO R² to [result].

3. Identified **peak nip pressure** as the dominant predictor (70.3% SHAP importance), confirming Johanson rolling theory predictions: SCF alone is insufficient — gap width and roll diameter must be jointly controlled.

4. Produced a 2D design-space map enabling fracture-safe calendering window selection for NMC-like brittle active materials.

5. The multi-source ML framework is generalizable: combining open pharmaceutical compaction datasets with targeted battery DEM simulations provides a cost-effective path to predictive calendering process optimization without extensive experimental campaigns.

---

## Figures (7 total)

| Figure | Content | Data source |
|---|---|---|
| **Fig 1** | Conceptual schematic: calendering → particle fracture → IPA datasets + DEM → ML framework | (illustration) |
| **Fig 2** | Dataset overview: (a) fines_fraction by material type, (b) peak_pressure histogram, (c) fines vs. pressure scatter colored by material | IPA granulation CSV |
| **Fig 3** | Two-stage ML architecture diagram | (diagram) |
| **Fig 4** | Model performance: 4-panel predicted vs. actual (LR, RF proc, RF phys, XGBoost+DEM) | model outputs |
| **Fig 5** | SHAP beeswarm plot: global feature importance, all 11 features | SHAP library |
| **Fig 6** | LOMO CV: Mannitol_SD predictions before vs. after DEM supplement (scatter plot) | LOMO results |
| **Fig 7** | 2D design space map: SCF × gap → fines_fraction contour, safe-window overlay | XGBoost predictions |

---

## Tables (2 total)

| Table | Content |
|---|---|
| **Table 1** | Model comparison: R², RMSE, training time for all 4 models × 2 feature sets |
| **Table 2** | LOMO CV R² by material class: before vs. after DEM data addition |

---

## Suggested Journal Submission Path

### Option A: Full paper — Journal of Power Sources (JPS)
- Impact factor: ~9.2
- Scope: battery manufacturing, electrode processing, simulation
- Length: 6000–8000 words + 7 figures
- Advantage: high impact, large battery-focused readership
- Timeline: ~3 months review

### Option B: Letter — Electrochemistry Communications
- Impact factor: ~5.5
- Scope: short communications, letters
- Length: 3000 words + 4 figures
- Cut to: Sections 1 (400 words) + 3 (data overview only) + 4.2 (RF results) + 5.1 + 5.2 + 6 (short) + 7
- Advantage: faster (~6 weeks review), good for establishing priority

### Option C: Methods paper — npj Computational Materials
- Impact factor: ~12.4
- Scope: computational materials science, ML for materials
- Emphasis: reframe as "multi-source data fusion for process-structure ML" with battery calendering as application
- Advantage: highest impact, but requires more rigorous ML methodology

**Recommendation: Start with JPS full paper. If R² for DEM-supplemented model is < 0.75, pivot to ECom letter focusing on the LOMO domain-gap finding alone.**

---

## Implementation Checklist

- [ ] Confirm granulation CSV loads, R² = 0.92 (done ✓)
- [ ] Confirm LOMO Mannitol_SD = 0.40 (done ✓)
- [ ] Generate 100 ARTISTIC 2D DEM runs (scripted pressure sweep)
- [ ] Train XGBoost on combined dataset; tune with Optuna
- [ ] Run SHAP analysis; generate beeswarm plot
- [ ] Generate 2D design space contour map (SCF × gap, brittle class)
- [ ] Check ARTISTIC DEM bond-breakage vs. fines_fraction Spearman correlation
- [ ] Write manuscript (Section 1 first; confirm message with advisor before full draft)
- [ ] Prepare figures in matplotlib (300 dpi, journal color palette)
- [ ] Submit to JPS
