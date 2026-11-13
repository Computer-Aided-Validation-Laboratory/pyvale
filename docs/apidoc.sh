#!/usr/bin/env bash
set -e
set -o pipefail

echo "=========================================="
echo " GENERATING PYTHON API DOCUMENTATION..."
echo "=========================================="

# Function to handle errors
error_exit() {
    echo "ERROR: $1"
    exit 1
}

# Generate from any working directory. PYTHON can select the docs virtualenv.
docs_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${PYTHON:-python3}" "$docs_dir/generate_apidoc.py" || error_exit "Python API generation failed"

echo "PYTHON API DOCUMENTATION GENERATED SUCCESSFULLY!"

echo "=========================================="
echo " GENERATING C++ API DOCUMENTATION..."
echo "=========================================="

if ! command -v doxygen >/dev/null 2>&1; then
    error_exit "doxygen is required to generate the native API reference"
fi
( cd "$docs_dir/source" && doxygen Doxyfile ) || error_exit "doxygen generation failed"
"${PYTHON:-python3}" "$docs_dir/generate_native_apidoc.py" || error_exit "Native API generation failed"

echo "C++ API DOCUMENTATION GENERATED SUCCESSFULLY!"
