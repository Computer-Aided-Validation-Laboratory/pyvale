
#generate the python api documentation
sphinx-apidoc -f -o ./source ../src/pyvale/ ../src/pyvale/data/* ../src/pyvale/examples/ --tocfile api_py

sed -i 's/^pyvale$/Detailed Python API/' source/api_py.rst
sed -i 's/=/====/g' source/api_py.rst
sed -i '/^pyvale package$/,+1d; /^Submodules$/,+1d' source/pyvale.rst
sed -i 's/^pyvale\.\(.*\) module$/\1/' source/pyvale.rst
sed -i '/^Module contents$/,/^ *:undoc-members:/d' source/pyvale.rst
