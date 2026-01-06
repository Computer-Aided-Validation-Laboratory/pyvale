.. _guide_sensorsim:

Sensor Simulation Guide
=======================

The sensor simulation module is pyvale is used for modelling sensors including sources of uncertainty with a particular focus on systematic an Type B measurement errors. Systematic and Type B measurement errors are the most difficult to characterise in practice and often contribute the most to the overall uncertainty of a given measurement. 

How does sensor simulation work?
--------------------------------
Here we will outline some key concepts needed to understand sensor simulation in pyvale. The sensor simulation model assumes the user has already performed a physics simulation that contains the field (scalar, vector or tensor) they want their virtual sensors to 'measure'. The user then provides information about the type of sensor they want to model, the locations of the sensors, the sampling times of the sensors, orientation of the sensors and any measurement errors they wish to model (i.e. systematic and random errors.)  

Measurement Simulation Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Pyvale sensor simulation is built on the concept of a 'sensor array' which is a group of sensors of the same type. An example would be an array of thermocouples applied to a component to measure temperature. Grouping similar sensors in this way allows them to be batch processed using vectorised numpy operations. Sensor arrays can be built to sample a physical field that is a scalar (e.g. temperature), vector (e.g. displacement of force) or tensor (e.g. strain) quantity. A sensor array in pyvale implements the following measurement simulation model:

measurement = truth + systematic errors + random errors

The 'truth' is taken from the input physics simulation provided by the user. If the user provides a mesh-based simulation the element shape functions are used for spatial interpolation to the sensor locations to calculate the truth. However, if the input physics simulation is mesh-free then Delaunay triangulation is performed with linear interpolation. By default the sensors are assumed to sample at the input simulation time steps. However, if the user provides specific time steps for sampling then linear interpolation between time steps is performed afer the spatial interpolation. For vector and tensor field sensors it is also possible to specify the orientation of the sensor with respect to the underlying simulation coordinates. 

All terms in the measurement simulation model are implemented as arrays with 3 dimensions and the following shape: (# sensors, # field components, # measurement times). The field components will be 1 for scalar fields, 2 for vector fields in 2D, 3 for vector fields in 3D and so on. In the next section we discuss how the error array is calculated using an error chain.

Error Chains & Simulated Errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulated measurement errors are added to to a sensor array using an 'error chain'. An error chain is an ordered list of simulated measurement errors that will be evaluated when the sensor array is used to generate simulated measurements. Once all errors in the chain are evaluated they are added to the truth array and returned to the user 

Simulated measurement errors in pyvale are classed as systematic or random and have a dependence as either independent or dependent. Systematic errors typically provide an offset, shift or bias in our measurement that is constant over time steps. However, it is also possible for systematic errors to be a function of the underlying field or to drift. Random errors are sampled from user specified probability distributions and are resampled at every time step. 

The sensor simulation module provides an extensive library of systematic and random errors including:

**Systematic Errors**: sampled from a probability distribution (in simulation untis or as a percentage), constant offsets (in simulation units or as a percentage), round-off error, digitisation error, saturation error, calibration errors and field errors (uncertainty in sensor position, sampling time and orientation).  
**Random Errors**: sampled from a probability distribution (in simulation units or as a percentage). 

Field errors are a special type of systematic error that requires additional interpolation of the underlying physical field. Examples of this are errors which perturb the sensor parameters such as uncertainty in the position, orientation or sampling time of a sensor array.  

Simulated measurement errors also have a dependence. The dependence determines what basis is used to calculate the error. For independent errors the truth is used as a basis if required. An example of this would be a random error that generates a percentage noise. If this error is independent then the percentage is calculated as a percentage of the truth. A dependent error is calculated based on the accumulated measurement at the given point in the error chain. For this case our same random error generating a percentage noise would calculate the percentage based on the truth plus the contribution of any errors before it in the chain. It is also possible to implement dependent field errors in which case the perturbations to the sensor array parameters (positions, times and orientation) are carried through the error chain instead of being based on the initial condition. Now that we understand how measurement errors are simulated in pyvale we can discuss how to user our sensor arrays to run many simulated experiments with a Monte-Carlo method.

Experiment Simulation
^^^^^^^^^^^^^^^^^^^^^

The experiment simulator is used to run Monte-Carlo simulations for a given set of input physics simulations and sensor arrays. The ability to apply the same sensor arrays to a set of input physics simulations provides the ability to analyse what the sensors would measure if the underlying inputs to the simulation were changed. For example: What would we measure with our virtual strain gauges if the material properties where +/-10% different to the nominal case? or; What would we measure if the boundary conditions where different to what we assumed in the nominal case? The experiment simulator also allows us to apply multiple sensor arrays to the same multi-physics simulation which measure the same or different fields. For example we could apply a sensor array of thermocouples (measuring temperature), pyrometers (measuring temperature) and strain gauges (measuring strain) to a thermo-mechanical input simulation.

When we run a simulated experiment the virtual sensor arrays are invoked 'N' times for each input physics simulation to generate 'N' virtual experiments for every combination of input physics simulation and sensor array. This means the total number of virtual experiments will be: # input physics simulations x # sensor arrays x N. The experiment simulator returns us a dictionary that contains simulated experiment data arrays for each combination of input simulation and sensor array. Each experiment data array has a similar shape to our sensor measurement array with an additional first dimension for the virtual experiment number: (# virtual experiments, # sensors in the array, # field components, # time steps). The experiment simulator will return us an experiment data array for the simulated measurements, the total systematic errors and the total random errors all having the same shape. The experiment data arrays obey the measurement simulation model, recall: measurement = truth + systematic errors + random errors. This means we can reconstruct the truth array using the arrays we have in our simulated experimental data dictionary, noting that the truth array will be the same over all N experiments. 

The experiment simulator supports parallelisation of simulations over a user specified number of workers. Simulations can be parallised such that all 'N' experiments for a given combination of input simulation and sensor array are run per worker; or such that an individual simulation is run per worker. For lightweight sensor simulation including just point sensors parallelising over the input simulations and sensor arrays running 'N' experiments per worker will be most efficient. For heavy sensor simulation (such as thos involving cameras) it will typically be best to parallelise such that a single virtual experiment is run per worker.   

Given the first dimension of the experimental data array is the number of experiments we can easily calculate statistics over this dimension of the array. The sensor simulation module provides us functionality to extract common statistics from an experimental data dicitionary such as: minimum, maximum, mean, standard deviation, median, quartiles and median absolute deviation. The sensor simulation module also provides IO functionality for saving and loading simulated experiments using the numpy array stack format '.npz'. Once we have a set of simulated experiments we can then use this for additional analysis with the other toolboxes in pyvale such as calculating validation metrics (currently under construction). 

User Workflow
^^^^^^^^^^^^^
The user workflow for pyvale sensor simulation consists of four main steps:
 
#. Load multi-physics simulation data.
#. Build virtual sensor arrays and attach an error simulation chain
#. Run simulated experiments.
#. Analyse & visualise the results. 

For a code example showing these examples in action using the pyvale sensor simulation module head to the :ref:`basics examples <examples_sensorsim_basics>`.

Creating your own sensor models
--------------------------------

While we have built and tested pyvale primarily with thermal and solid mechanics problems the interfaces for building sensor simulation models in pyvale are physics agnostic. A sensor is just something that samples a scalar, vector or tensor field at the given locations and given times. This means it is possible to expand on the existing sensor models in pyvale to apply to your own use case such as: velocity or pressure field measurements in fluid mechanics or neutron flux spectrums for neutronics experiments with radiation foils. 

Bring your own simulation data
------------------------------

To use pyvale with your own input physics simulation you will first need to parse the simulation data into a ``SimData`` object. To help users input arbitrary simulation data we have developed tools that allow users to directly load the data as either plain text delimited files (e.g. *.csv) or numpy arrays (i.e. *.npy). An example of loading arbitrary simulation data can be found in the :ref:`extended examples <examples_sensorsim_extended>`. 

For mesh-based simulations it is essential that the element connectivity tables conform to the node winding (i.e. node ordering) specified for exodus output files which can be found `here <https://sandialabs.github.io/seacas-docs/html/element_types.html>`_.    

We also support directly reading the exodus output format (.e) as this is what is generated by the open-source finite element framework we use called `MOOSE <https://mooseframework.inl.gov/>`_ (Multi-physics Object Oriented Simulation Enviroment). In the future we are looking to support directly loading output from other open-source finite element tools such as NGSolve and FEniCS. 
