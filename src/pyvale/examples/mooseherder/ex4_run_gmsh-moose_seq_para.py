'''
==============================================================================
EXAMPLE: Run parallel gmsh+MOOSE simulation editing the gmsh parameters only

Author: Lloyd Fletcher, Rory Spencer
==============================================================================
'''
from pathlib import Path
from mooseherder import (MooseHerd,
                         MooseRunner,
                         MooseConfig,
                         GmshRunner,
                         InputModifier,
                         DirectoryManager)

USER_DIR = Path.home()

def main():
    """main _summary_
    """
    print("-"*80)
    print('EXAMPLE: Herd Setup')
    print("-"*80)

    # Setup MOOSE runner and input modifier
    moose_input = Path('scripts/moose/moose-mech-gmsh.i')
    moose_modifier = InputModifier(moose_input,'#','')

    moose_config = MooseConfig().read_config(Path.cwd() / 'moose-config.json')
    moose_runner = MooseRunner(moose_config)
    moose_runner.set_run_opts(n_tasks = 1,
                              n_threads = 2,
                              redirect_out = True)

    # Setup Gmsh
    gmsh_input = Path('scripts/gmsh/gmsh_tens_spline_2d.geo')
    gmsh_modifier = InputModifier(gmsh_input,'//',';')

    gmsh_path = USER_DIR / 'gmsh/bin/gmsh'
    gmsh_runner = GmshRunner(gmsh_path)
    gmsh_runner.set_input_file(gmsh_input)

    # Setup herd composition
    sim_runners = [gmsh_runner,moose_runner]
    input_modifiers = [gmsh_modifier,moose_modifier]
    dir_manager = DirectoryManager(n_dirs=4)

    # Start the herd and create working directories
    herd = MooseHerd(sim_runners,input_modifiers,dir_manager)
    herd.set_num_para_sims(n_para=4)
    # Don't have to clear directories on creation of the herd but we do so here
    # so that directory creation doesn't raise errors
    dir_manager.set_base_dir(Path('examples/'))
    dir_manager.clear_dirs()
    dir_manager.create_dirs()

    # Create variables to sweep in a list of dictionaries for mesh parameters
    # 2^3=8 combinations possible
    p0 = [1E-3,2E-3]
    p1 = [1.5E-3,2E-3]
    p2 = [1E-3,3E-3]
    var_sweep = list([])
    for nn in p0:
        for ee in p1:
            for pp in p2:
                var_sweep.append([{'p0':nn,'p1':ee,'p2':pp},None])

    print('Herd sweep variables:')
    for vv in var_sweep:
        print(vv)

    print()
    print("-"*80)
    print('EXAMPLE: Run Gmsh+MOOSE once, modify gmsh only')
    print("-"*80)

    # Single run saved in sim-workdir-1
    herd.run_once(0,var_sweep[0])

    print(f'Run time (once) = {herd.get_iter_time():.3f}) seconds')
    print("-"*80)
    print()

    print("-"*80)
    print('EXAMPLE 4b: Run MOOSE sequentially, modify gmsh only')
    print("-"*80)

    # Run all variable combinations (8) sequentially in sim-workdir-1
    herd.run_sequential(var_sweep)

    print(f'Run time (sequential) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()
    print("-"*80)
    print('EXAMPLE: Run MOOSE in parallel, modify gmsh only')
    print("-"*80)

    # Run all variable combinations across 4 MOOSE instances with two runs saved in
    # each sim-workdir
    herd.run_para(var_sweep)

    print(f'Run time (parallel) = {herd.get_sweep_time():.3f} seconds')
    print("-"*80)
    print()

if __name__ == '__main__':
    main()

