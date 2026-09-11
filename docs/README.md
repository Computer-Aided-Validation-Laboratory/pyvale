# building the documentation locally

install the requirements for the docs:
```bash
pip install -r requirements.txt
sudo apt install doxygen
```
init the API documentation using the `apidoc.sh` script:
```bash
bash apidoc.sh 
```
build the documentation:
```bash
make html
```
