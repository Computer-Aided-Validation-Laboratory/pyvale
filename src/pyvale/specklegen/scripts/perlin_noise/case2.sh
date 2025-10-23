SIMULATION="perlin_noise"
CASE="case2"

if [ ! -d logs/$SIMULATION ]; then
  mkdir -p logs/$SIMULATION
fi

> logs/$SIMULATION/$CASE.log

for RES_WIDTH in 10 25 100    
do
for RES_HEIGHT in 10 25 100
do
for OCTAVES in 1 2 3
do
  python -u runfiles/main_specklegen_perlin_noise.py \
    --output_path ./output/$SIMULATION/$CASE/ \
    --screen_size_width 1000 \
    --screen_size_height 800 \
    --res_width $RES_WIDTH \
    --res_height $RES_HEIGHT \
    --bit_depth 8 \
    --theme "white_on_black" \
    --type_gen "fractal" \
    --octaves $OCTAVES \
    >> logs/$SIMULATION/$CASE.log 2>&1  # Append to the log file
done
done
done