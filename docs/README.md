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
