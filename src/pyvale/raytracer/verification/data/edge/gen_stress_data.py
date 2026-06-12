import os
import numpy as np

def generate_stress_data():
    path = "data/edge/tri6_stress"
    os.makedirs(path, exist_ok=True)
    
    # Corners
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([10.0, 0.0, 0.0])
    v2 = np.array([5.0, 8.660254037844386, 0.0])
    v3 = np.array([5.0, -8.660254037844386, 0.0])
    
    # Original Midsides
    m01 = (v0 + v1) / 2.0
    m12 = (v1 + v2) / 2.0
    m20 = (v2 + v0) / 2.0
    m13 = (v1 + v3) / 2.0
    m30 = (v3 + v0) / 2.0
    
    # Centroids
    c1 = (v0 + v1 + v2) / 3.0
    c2 = (v0 + v1 + v3) / 3.0
    
    def move_towards(p, target, dist):
        vec = target - p
        vec = vec / np.linalg.norm(vec)
        return p + vec * dist

    def move_away(p, target, dist):
        vec = p - target
        vec = vec / np.linalg.norm(vec)
        return p + vec * dist

    # Triangle 1 (Convex - pull towards C1)
    # Triangle 2 (Concave - pull away from C2)
    # Note: Shared edge M01 is pulled towards C1, which is away from C2. Perfect.
    
    m01_new = move_towards(m01, c1, 1.0)
    m12_new = move_towards(m12, c1, 1.0)
    m20_new = move_towards(m20, c1, 1.0)
    m13_new = move_away(m13, c2, 1.0)
    m30_new = move_away(m30, c2, 1.0)
    
    nodes = [v0, v1, v2, v3, m01_new, m12_new, m20_new, m13_new, m30_new]
    
    # coords.csv
    with open(f"{path}/coords.csv", "w") as f:
        for n in nodes:
            f.write(f"{n[0]:.18e},{n[1]:.18e},{n[2]:.18e}\n")
            
    # connectivity.csv
    # Tri6: 0,1,2 corners, 3,4,5 midsides (0-1, 1-2, 2-0)
    with open(f"{path}/connectivity.csv", "w") as f:
        f.write("0,1,2,4,5,6\n") # Tri 1
        f.write("1,0,3,4,8,7\n") # Tri 2
        
    # uvs.csv (rescale to be between 0.3 and 0.7)
    with open(f"{path}/uvs.csv", "w") as f:
        for n in nodes:
            u = 0.3 + (n[0] / 10.0) * 0.4
            v = 0.3 + ((n[1] + 8.660254037844386) / (2.0 * 8.660254037844386)) * 0.4
            f.write(f"{u:.18e},{v:.18e}\n")
            
    # field_disp_x.csv, field_disp_y.csv, field_disp_z.csv
    # Simple linear displacement for two timesteps
    for axis, i in [("x", 0), ("y", 1), ("z", 2)]:
        with open(f"{path}/field_disp_{axis}.csv", "w") as f:
            for n in nodes:
                val1 = n[i] * 0.01
                val2 = n[i] * 0.1
                f.write(f"{val1:.18e},{val2:.18e}\n")

if __name__ == "__main__":
    generate_stress_data()
