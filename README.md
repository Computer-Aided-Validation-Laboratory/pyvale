<p align="center">
  <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/pyvale_logo.png" alt="PyVale" width="620">
</p>

<p align="center"><strong>Your virtual engineering laboratory: design experiments, analyse measurements, and iterate with confidence.</strong></p>

<p align="center">
  <a href="https://pypi.org/project/pyvale/"><img src="https://img.shields.io/pypi/v/pyvale?label=PyPI" alt="PyPI version"></a>
  <a href="https://pypi.org/project/pyvale/"><img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Supported Python versions"></a>
  <a href="https://github.com/Computer-Aided-Validation-Laboratory/pyvale/actions/workflows/tests.yml"><img src="https://img.shields.io/github/actions/workflow/status/Computer-Aided-Validation-Laboratory/pyvale/tests.yml?branch=main&label=tests" alt="Tests"></a>
  <a href="https://github.com/Computer-Aided-Validation-Laboratory/pyvale/actions/workflows/wheels.yml"><img src="https://img.shields.io/github/actions/workflow/status/Computer-Aided-Validation-Laboratory/pyvale/wheels.yml?branch=main&label=wheels" alt="Wheels"></a>
  <a href="https://computer-aided-validation-laboratory.github.io/pyvale/"><img src="https://img.shields.io/github/actions/workflow/status/Computer-Aided-Validation-Laboratory/pyvale/docs.yml?branch=main&label=docs" alt="Documentation"></a>
  <a href="https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Computer-Aided-Validation-Laboratory/pyvale" alt="MIT license"></a>
</p>

PyVale is a general purpose toolbox for simulation driven experimental design
and experimental mechanics. Build virtual sensor arrays, generate realistic
camera images, analyse DIC measurements, and feed what you learn into the next
experiment. Compare simulations with measurements under uncertainty and
identify material properties from full-field strain and force data. Tools
for sensor placement optimisation and experimental design are under active
development.

## PyVale Design Framework

PyVale connects experimental design, measurement simulation, data analysis simulation validation and uncertainty quantification in an iterative workflow. Its core modules can be used independently or combined to close the loop between simulation and experiment.

| Capability | What it gives you | Documentation |
|:---|:---|:---:|
| **SensorSim** | Virtual sensor arrays, uncertainty models, and repeated simulated experiments | [Examples](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_basics_sensorsim.html) · [Guide](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_sensorsim.html) |
| **DIC** | Two dimensional and stereo correlation, shape reconstruction, displacement, and strain | [Examples](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_dic.html) · [Guide](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_dic.html) |
| **Render** | Synthetic camera images, deforming meshes, and optical realism | [Examples](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_render3d.html) · [UV Examples](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_renderuvs.html) |
| **Validation** | Simulation-to-experiment comparisons and validation metrics under uncertainty | [Examples](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_valid.html) · [Guide](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_valid.html) |
| **VFM** | Identification of homogeneous and spatially varying material properties from strain and force histories | [Examples](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/tree/main/src/pyvale/examples/vfm) |

## Core Capabilities

### SensorSim · simulate measurements and uncertainty

Create virtual thermocouples, strain gauges, and other sensor arrays on
multiphysics simulations. Model systematic and random uncertainty, repeat
virtual experiments, and inspect the resulting measurement distributions.

[**SensorSim examples**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_basics_sensorsim.html) · [**User guide**](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_sensorsim.html)

| Sensor locations | Simulated sensor traces |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/basics_ex0_locs.png" alt="Virtual sensor locations" width="520"> | <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/basics_ex0_traces.png" alt="Simulated sensor traces" width="520"> |

### DIC · analyse deformation from images

Run two dimensional and stereo digital image correlation on synthetic or
experimental images. Define regions of interest, correlate large image sets,
reconstruct surfaces, and calculate displacement and strain.

[**DIC examples**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_dic.html) · [**User guide**](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_dic.html)

| Stereo region of interest | Reconstructed shape |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/dic_ex10_roi.png" alt="Stereo DIC region of interest" width="520"> | <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/dic_ex10_3d.png" alt="Stereo DIC reconstructed shape" width="520"> |

### Render · build virtual camera experiments

Render deforming finite element meshes through the verified Riley rasteriser or
the optional Blender backend. Configure camera geometry, distortion, point
spread functions, textures, stereo pairs, and physically meaningful speckle
scales.

[**Render examples →**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_render3d.html) · [**UV mapping examples →**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_renderuvs.html)

<p align="center">
  <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/render3d_ex1d_riley_dicuq.png" alt="Riley render of a speckled plate with a hole" width="900">
</p>

### Validation · compare simulation with experiment

Compare simulation predictions with experimental measurements under
uncertainty. Use validation metrics to quantify discrepancies in scalar
observables and time histories, then inspect where the predictions and
measurements differ.

[**Validation examples**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples_valid.html) · [**User guide**](https://computer-aided-validation-laboratory.github.io/pyvale/guide_user/guide_valid.html)

| Thermocouple validation envelope | Weighted temperature distributions |
|:---:|:---:|
| <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/val_ex10_fig03_mavm_support_envelope_TC3.png" alt="Validation metric support envelope for thermocouple TC3" width="520"> | <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/val_ex10_fig05_2_dextremes_wcdfs_TC3.png" alt="Weighted cumulative temperature distributions for thermocouple TC3" width="520"> |

### VFM · identify material properties from full-field measurements

Use the Virtual Fields Method to identify elastoplastic properties from
full-field strain and measured force histories. Fit uniform properties or
estimate how yield strength varies across a specimen using slice-wise
identification.

[**VFM examples**](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/tree/main/src/pyvale/examples/vfm)

<p align="center">
  <img src="https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/pyvale/main/images/vfm_yield_strength.png" alt="Identified yield strength in four slices of a synthetic tensile specimen" width="900">
</p>

## Install

PyVale supports Python 3.11 and newer. Most users will use Riley as their rendering engine for image simulation however, if you need path tracing then Blender integration requires Python 3.13 and the optional Blender dependencies.

| Platform | Install commands |
|:---|:---|
| pip | `pip install pyvale` |
| uv | `uv add pyvale` |
| Blender tools | `pip install "pyvale[blender]"` |

[**Installation guide**](https://computer-aided-validation-laboratory.github.io/pyvale/install/install.html) · [**Browse all examples**](https://computer-aided-validation-laboratory.github.io/pyvale/examples/examples.html) · [**Open the documentation**](https://computer-aided-validation-laboratory.github.io/pyvale/)

## Acknowledgements

PyVale is developed by the Computer Aided Validation Team and collaborators.
Its motivation comes from the demanding simulation validation experiments
needed in fusion engineering, while its tools are intended for experimental
mechanics generally.

[Contributors](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/CONTRIBUTORS.md) · [Contributing](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/CONTRIBUTING.md) · [Citation](https://computer-aided-validation-laboratory.github.io/pyvale/cite.html) · [MIT license](https://github.com/Computer-Aided-Validation-Laboratory/pyvale/blob/main/LICENSE)
