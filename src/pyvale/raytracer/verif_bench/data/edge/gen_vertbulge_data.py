import os
import numpy as np

def generate_vertbulge_data():
    path = "data/edge/tri6_vertbulge"
    os.makedirs(path, exist_ok=True)
    
    # Corners
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([10.0, 0.0, 0.0])
    v2 = np.array([5.0, 10.0, 0.0])
    
    # Midside nodes placed to trigger the bug
    # Edge 0-1 (base): Straight
    m01 = np.array([5.0, 0.0, 0.0])
    # Edge 1-2: Midside directly above V1
    m12 = np.array([10.0, 5.0, 0.0])
    # Edge 2-0: Midside directly above V0
    m20 = np.array([0.0, 5.0, 0.0])
    
    nodes = [v0, v1, v2, m01, m12, m20]
    
    # coords.csv
    with open(f"{path}/coords.csv", "w") as f:
        for n in nodes:
            f.write(f"{n[0]:.18e},{n[1]:.18e},{n[2]:.18e}\n")
            
    # connectivity.csv
    with open(f"{path}/connectivity.csv", "w") as f:
        f.write("0,1,2,3,4,5\n")
        
    # uvs.csv (center them)
    with open(f"{path}/uvs.csv", "w") as f:
        for n in nodes:
            u = 0.3 + (n[0] / 10.0) * 0.4
            v = 0.3 + (n[1] / 10.0) * 0.4
            f.write(f"{u:.18e},{v:.18e}\n")
            
    # field_disp_x.csv, etc.
    for axis, i in [("x", 0), ("y", 1), ("z", 2)]:
        with open(f"{path}/field_disp_{axis}.csv", "w") as f:
            for n in nodes:
                val = n[i] * 0.01
                f.write(f"{val:.18e},{val:.18e}\n")

if __name__ == "__main__":
    generate_vertbulge_data()
