SIMULATION="base"
CASE="case2"

> logs/$SIMULATION/$CASE.log

for SPECKLE_SIZE in $(seq 10 5 25)             
do
for SIGMA_COEF in 1 3 5
do
  python -u runfiles/main_specklegen.py \
    --output_path ./output/$SIMULATION/$CASE/ \
    --speckle_size $SPECKLE_SIZE \
    --screen_size_width 1000 \
    --screen_size_height 800 \
    --bit_depth 8 \
    --theme "black_on_white" \
    --random_seed 10 \
    --sigma $(($SPECKLE_SIZE/$SIGMA_COEF)) \
    --reduce_overlap "False" \
    >> logs/$SIMULATION/$CASE.log 2>&1  # Append to the log file

done
done