Here I'll outline one method of accessing C++ variables from python using Cython as an interface. 

#### C++ vectors
In this example I've created a `std::vector` in an external C++ file (dicengine.hpp) within a namespace (dic2d). 
In my Cython `.pyx` file I add a `cdef extern from` section that allows python to access these variables.

```python
from libcpp.vector cimport vector

cdef extern from "../cpp/dicengine.hpp" namespace "dic2d":

    # Declare the result arrays 
    extern vector[int] ss_coord_list

    # you might want to use a 

    # C++ function that runs the DIC engine
    void dic_engine(...);

def call_dic_engine(...):

  dic_engine(..)

  # Create a Cython memoryview of a 1D integer array
  cdef int[::1] ss_list_view = <int [:ss_coord_list.size()]>ss_coord_list.data()

   # converts the memoryview into a NumPy array without copying.
  subsets_1d = np.frombuffer(ss_list_view, dtype=np.int32)

  # Reshape Numpy array to (x,y) pairs of coordinates.
  subsets = subsets_1d.reshape(ss_coord_list.size()//2, 2)



```
The `cdef extern from` block tells Cython that these functions and variables exist in an external C++ file (`dicengine.hpp`) within the namespace `dic2d`.
`extern vector[int] ss_coord_list` declares an external `std::vector<int>` named `ss_coord_list` which is defined and populated within the C++ code.
`def call_dic_engine(...):` is used to run C++ from within python and create a memoryview to our `std::vector`.
In this example I was interested in storying the subset coordinates as a 2d array of (x,y) values for each subset in the region of interest. To do this I:
* Use a  memoryview (int[::1]) to access the C++ vector without copying.
* Converts the memoryview into a NumPy array.
* Reshapes it into a 2D NumPy array storing (x, y) pairs.

#### C Arrays
I won't cover go through line by line because the principles will remain the exact same, you just need to be careful with memory allocation and deallocation with standard C arrays:

```python
cdef extern from "../cpp/dicengine.hpp":
    extern int* ss_coord_array  # Pointer to dynamically allocated C array
    extern int ss_coord_size    # Number of elements

    # Make sure to allocate in C code. We can assume the allocation is done in dic_engine(...)
    void dic_engine(...);

    # free the memory of the array. No garbage collector.
    void free_array()

def call_dic_engine(...):

    dic_engine(..)
   # Convert to memoryview
    cdef int[::1] ss_list_view = <int[:ss_coord_size]> ss_coord_array

    # Convert to NumPy array
    subsets_1d = np.frombuffer(ss_list_view, dtype=np.int32)

    # Reshape for (x, y) pairs
    subsets = subsets_1d.reshape(ss_coord_size // 2, 2)

def cleanup():
    free_array()  # Free memory allocated in C

```
