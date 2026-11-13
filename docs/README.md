# building the documentation locally

install the requirements for the docs:
```bash
pip install -r requirements.txt
sudo apt install doxygen
```
Regenerate the API documentation using `apidoc.sh` (from this directory or the
repository root). Use a Python installation containing Sphinx; set `PYTHON` if
it is not the shell's `python3`:

```bash
PYTHON=.venv/bin/python bash docs/apidoc.sh  # From the repository root
```

The script clears only recognised, Git-ignored generated Python API pages before
regenerating them. Tracked pages (including `pyvale.render.uvtools.rst`), manual
content, galleries and guides are preserved. Generation is staged and the previous
API pages are restored if regeneration fails. Newly generated pages carry an
ownership comment; track a page or remove that comment before maintaining it by
hand. C++ documentation generation runs afterward as before.

With the docs Python installation already on PATH, you can also run from here:
```bash
bash apidoc.sh
```
build the documentation:
```bash
make html
```

## Landing-page images

Generate the physics-simulation field and mesh images from PyVale's case18
Exodus data with the project environment:

```bash
.venv/bin/python scripts/docs_image_gen.py
```

Outputs and a dataset/time-selection manifest are written to
`pyvale-output/docs-images/`. The data API downloads and caches the dataset if
needed. The field image uses viridis, native coordinates and field units, with
displacement warping disabled and no sensor overlays. Copy the selected field
image to `images/physics_simulation_field.png` to update the landing page;
`docs/source/_static/physics_simulation_field.png` links to it.

The other capability images come from the existing SensorSim, DIC and Render
examples. The backend logos are copied unchanged from their upstream README
assets:

- [Riley logo](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/riley-raster/main/images/riley_logo.png)
- [Rabbit FEM logo](https://raw.githubusercontent.com/Computer-Aided-Validation-Laboratory/rabbit-fem/main/images/logo_rabbitfem.png)

They are stored in `images/` and linked from `docs/source/_static/`.
Citation formats for the PyVale DIC and Riley papers live on the separate
`docs/source/cite.rst` page, available through the documentation sidebar.
