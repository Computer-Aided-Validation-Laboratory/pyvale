import time
import numpy as np
import camerac
import cythtest
import cython

def main() -> None:
    n_divs: int = 20000
    n_runs: int = 10

    x = np.linspace(0,20,n_divs)
    y = np.linspace(0,10,n_divs)


    print("Running numpy meshgrid")
    run_times = np.zeros((n_runs,))
    for nn in range(n_runs):
        start_time = time.perf_counter()
        (x_grid,y_grid) = np.meshgrid(x,y)
        run_times[nn] = time.perf_counter() - start_time

    print(f"Numpy meshgrid over {n_runs} runs = {np.mean(run_times)}")

    print("Running cython meshgrid")
    run_times = np.zeros((n_runs,))
    for nn in range(n_runs):
        start_time = time.perf_counter()
        (x_grid,y_grid) = camerac.meshgrid2d(x,y)
        run_times[nn] = time.perf_counter() - start_time

    print(f"Compiled meshgrid over {n_runs} runs = {np.mean(run_times)}")

    # to_norm = np.arange(0,1e9)
    # print("Normalising")
    # normd = cythtest.normalize(to_norm)
    # print("Finished")





if __name__ == "__main__":
    main()