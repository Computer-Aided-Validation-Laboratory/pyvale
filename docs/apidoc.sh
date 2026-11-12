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
cd "$docs_dir"

# Current module roots used by the C++ documentation steps below.
modules=("dic" "mooseherder" "sensorsim" "verif" "data" "strain" "render")

echo "PYTHON API DOCUMENTATION GENERATED SUCCESSFULLY!"

# ------------------------------------------------------------------
# Generate C++ API documentation
# ------------------------------------------------------------------
echo "=========================================="
echo " GENERATING C++ API DOCUMENTATION..."
echo "=========================================="

# Run Doxygen (optional, if you still want it)
if command -v doxygen >/dev/null 2>&1; then
    ( cd ./source && doxygen Doxyfile ) || error_exit "doxygen generation failed"
else
    echo "doxygen not found: skipping C++ XML generation"
fi

# Create RST files for each C++ header
for mod in "${modules[@]}"; do
    src_dir="../src/pyvale/${mod}/cpp"
    out_dir="source/cpp/${mod}"

    mkdir -p "$out_dir"

    for file in "$src_dir"/*.hpp; do
        [ -e "$file" ] || continue

        filename=$(basename "$file")
        base="${filename%.hpp}"
        rst_file="${out_dir}/${base}.rst"

        echo "Generating $rst_file"

        {
            echo "${filename}"
            # Title underline must be at least as long as the title.
            printf '%.0s=' $(seq 1 "${#filename}")
            echo ""
            echo ""
            echo ".. doxygenfile:: ${filename}"
            echo "   :project: pyvale"
            echo "   :path: ${src_dir}"
        } > "$rst_file"
    done
done

# ------------------------------------------------------------------
# Generate module-level index RST files for C++ headers
# ------------------------------------------------------------------
for mod in "${modules[@]}"; do
    out_dir="source/cpp/${mod}"
    index_file="source/cpp/cpp_${mod}.rst"

    headers=""
    for f in "$out_dir"/*.rst; do
        [ -e "$f" ] || continue  # skip if no files
        headers="$headers $(basename "$f")"
    done

    # Skip if no headers found
    [ -z "$headers" ] && continue

    # Sort headers alphabetically
    headers=$(echo $headers | tr ' ' '\n' | sort)

    echo "Generating module index $index_file"

    {
        echo "${mod}"
        echo "=================="
        echo ""
        echo ".. toctree::"
        echo "   :caption: C++ Source Files"
        echo "   :maxdepth: 1"
        echo ""
        for h in $headers; do
            echo "   ${mod}/${h}"
        done
    } > "$index_file"
done



echo "C++ RST FILES GENERATED SUCCESSFULLY!"

