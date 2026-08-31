# Pyvale Common Abbreviations

A glossary of abbreviations used across the `pyvale` source code, grouped by
usage. Keep new abbreviations consistent with these forms where possible.

## Loop Iterators

Double-letter iterators are the numpy-style convention sanctioned in
`dev/README.md`. The letters hint at what is being iterated over:

- `ii`, `jj`, `kk`, `ll`: generic first, second, third, fourth loop indices;
  `kk` is also the key half of key/value pairs (`for kk, vv in ...`)
- `vv`: value half of key/value pairs (`for kk, vv in sens_vars`)
- `mm`: mapper/key half of function-table pairs (`for kk, mm in sens_funcs`)
- `ff`: frame index (`for ff in range(num_frames)`)
- `ee`: element index (`for ee in range(elem_coords.shape[0])`)
- `ss`: sensor or set entry (`for ss in sens_data_dict`); note this collides
  with exodus *side sets* terminology, so prefer it only for sensors
- `bb`: element-variable block tuples (`(field, block)` pairs)
- `pp`: process/frame record dictionaries in parallel loading loops
- `dd`, `tt`, `nn`: occasional data/time/node indices in local scopes

## Sensors & Simulation

- `sens`: sensor(s); the most common abbreviation in the codebase
  (`sens_data`, `SensDesc`, `sens_pos`)
- `sim`: simulation (`SimData`, `sim_case_*`, `SimLoaderByTime`)
- `exp`: experiment or experimental data (`ExpData`, `ExpLoadOpts`,
  `expsim`)
- `disp`: displacement, never display (`disp_x`, `field_disp_keys`)
- `coords`: nodal coordinates, shape `(nodes, spatial_dims)`
- `dims`: dimensionality or the simulation-dimension dictionary
  (`get_sim_dims`); also `num_spat_dims`
- `temp`: temperature in thermomechanical simulations, but temporary in
  `temp_mask`/`temp_dir`; ambiguous, prefer spelling out one of the two in
  new code
- `calc`: calculate (`calc_first_surface_metric`)
- `gen`: generator (`AnalyticSimDataGen`, `gen_gold_measurements`)
- `ref`: reference, usually the undeformed reference image/frame/config
- `seed`, `rng`: random-number seed and `np.random.default_rng` generator
- `glob`: exodus *global* variables (`glob_vars`), alongside `node_vars`,
  `elem_vars`, and `side_sets`

## Cameras & Rendering

- `cam`: camera; `cam0`/`cam1` are stereo camera pairs
- `roi`: region of interest (`roi_cent_world`)
- `px`: pixel (`mm_per_px`); spelled `pixels_` in field names
  (`pixels_num`, `pixels_size`, `pixels_count`)
- `leng`: length (`leng_per_px`, field-of-view extents)
- `pos`: position (`pos_world`)
- `rot`: rotation, normally a scipy `Rotation` (`rot_world`)
- `fov`: field of view (`field_of_view`)
- `psf`: point-spread function (`EPSFType`, `GaussianPSF`)
- `trans`: transformation (`trans_mat`, `tensor_trans`); the `mat` fragment
  appears only inside `trans_mat` - materials are always spelled out
- `sub_sample` vs `subsample`: historical inconsistency - the 3D `Camera`
  uses `sub_sample`, while `Camera2D` and `PxInt2DOpts` use `subsample`

## DIC & Images

- `dic`: digital image correlation (`pyvale.dic`, `DIC2D`)
- `img`/`image`: images; field names spell out `image`
- `mask`: validity/specimen mask applied to image pixels

## Data Structures & Options

- `opts`: options dataclass instances (`ImageDefOpts`, `VisOpts*`,
  `SimLoadOpts`)
- `num`: number-of counter prefix (`num_frames`, `num_spat_dims`)
- `arr`: array
- `err`: sensor error models (`ErrChain`, error calculators); raised
  exceptions use `error`/`exception` instead
- `chain`: error-model chain applied to synthetic sensor data (`err_chain`,
  `set_error_chain`)
- `vars`: exodus variable groups (`node_vars`, `elem_vars`, `glob_vars`)
- `config`: renderer/backend configuration objects, spelled out unlike
  `opts`

## Files & Libraries

- `np`: numpy; `plt`: matplotlib.pyplot; `pv`: pyvista; `nc`: netCDF4
- Format tokens in names: `csv`, `yaml`, `tiff`, `bmp`, `npy`, and exodus
  `.e` files

## Naming Conventions

- `E` prefix: enumerations (`EMeshType`, `ELightType`, `EElementType`)
- `I` prefix: abstract-base-class interfaces (`ISensor`, `IRenderer3D`,
  `IImageWarp2D`)
- `_` leading underscore: private/internal members and modules
