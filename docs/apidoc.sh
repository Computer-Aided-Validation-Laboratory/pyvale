sphinx-apidoc -f -o ./source ../src/pyvale/ ../src/pyvale/data/* ../src/pyvale/examples/ --tocfile api

sed -i 's/^pyvale$/Detailed API/' source/api.rst
sed -i '/^pyvale package$/,+1d; /^Submodules$/,+1d' source/pyvale.rst
sed -i 's/^pyvale\.\(.*\) module$/\1/' source/pyvale.rst
