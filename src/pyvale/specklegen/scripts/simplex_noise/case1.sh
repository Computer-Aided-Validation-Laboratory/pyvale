SIMULATION="simplex_noise"
CASE="case1"

> logs/$SIMULATION/$CASE.log

for SPECKLE_SIZE in 10.0 15.0 25.0 30.0 40.0   
do
  python -u runfiles/main_specklegen_simplex_noise.py \
    --output_path ./output/$SIMULATION/$CASE/ \
    --screen_size_width 1000 \
    --screen_size_height 800 \
    --speckle_size $SPECKLE_SIZE \
    --bit_depth 8 \
    --theme "white_on_black" \
    --seed 1234 \
    >> logs/$SIMULATION/$CASE.log 2>&1  # Append to the log file
done