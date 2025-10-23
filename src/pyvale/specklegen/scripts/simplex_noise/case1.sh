SIMULATION="simplex_noise"
CASE="case1"

if [ ! -d logs/$SIMULATION ]; then
  mkdir -p logs/$SIMULATION
fi

> logs/$SIMULATION/$CASE.log

for SPECKLE_SIZE_WIDTH in 10 15 20 30 40
do
for SPECKLE_SIZE_HEIGHT in 10 15 20 30 40
do
  python -u runfiles/main_specklegen_simplex_noise.py \
    --output_path ./output/$SIMULATION/$CASE/ \
    --screen_size_width 1000 \
    --screen_size_height 800 \
    --speckle_size_width $SPECKLE_SIZE_WIDTH \
    --speckle_size_height $SPECKLE_SIZE_HEIGHT \
    --bit_depth 8 \
    --theme "white_on_black" \
    --seed 1234 \
    >> logs/$SIMULATION/$CASE.log 2>&1  # Append to the log file
done
done