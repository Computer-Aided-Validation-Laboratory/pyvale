'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import os
import numpy as np
import pyvale.imagesim.textreader as tr

# CLASS DEFINITIONS
class Options:
    def __init__(self,in_path,in_sim_num,dims):
        self.path = in_path
        self.sim_num = in_sim_num
        self.dims = dims

        # Default file extension and prefix
        self.ext = '.txt'
        self.pref_node_loc = 'Node_Locs_'
        self.pref_elem_tab = 'Elem_Table_'
        self.pref_sim_param = 'SimParams_'

        self.disp_loc = 'Node'
        self.stress_strain_loc = 'Node'

        self.disp_str = '_Disp_'
        self.force_str = '_Force_'
        self.stress_str = '_Stress_'
        self.strain_str = '_Strain_'

        # Flags
        self.flag_cmd_line =  False
        self.flag_load_params = True
        self.flag_load_disp_fields = True
        self.flag_load_strain_fields = False
        self.flag_load_node_force = True

    #-------------------------------------------------------------------------
    def set_path(self,in_path):
        self.path = in_path

    def set_sim_num(self,in_sim_num):
        self.sim_num = in_sim_num

    #-------------------------------------------------------------------------
    def get_sim_param_file(self):
        return self._file_str(self.pref_sim_param)

    def get_node_loc_file(self):
        return self._file_str(self.pref_node_loc)

    def get_elem_tab_file(self):
        return self._file_str(self.pref_elem_tab)

    def get_disp_frame_file(self,frame):
        return self._frame_file_str(self.disp_loc,
                                    self.disp_str,
                                    frame)

    def get_vector_frame_file(self,vector_str,frame):
        return self._frame_file_str(self.disp_loc,
                                    ('_'+vector_str+'_'),
                                    frame)

    def get_stress_frame_file(self,frame):
        return self._frame_file_str(self.stress_strain_loc,
                                    self.stress_str,
                                    frame)

    def get_strain_frame_file(self,frame):
        return self._frame_file_str(self.stress_strain_loc,
                                    self.strain_str,
                                    frame)

    def get_tensor2_frame_file(self,tensor_str,frame):
        return self._frame_file_str(self.stress_strain_loc,
                                    ('_'+tensor_str+'_'),
                                    frame)

    #-------------------------------------------------------------------------
    # Internal Methods
    def _file_str(self,in_str):
        return (self.path
                + in_str
                + str(int(self.sim_num))
                + self.ext)


    def _frame_file_str(self,loc_str,data_str,frame):
        return (self.path
                + loc_str
                + data_str
                + str(int(self.sim_num))
                + '_'
                + str(int(frame))
                + self.ext)

# End Class: Options
#-----------------------------------------------------------------------------

# Data structure for holding simulation parameters which are read from a file
class SimParams:
    pass

# Data structure for holding the initial node positions in the mesh
class Nodes:
    def __init__(self):
        self.nums = list([])
        self.loc_x = list([])
        self.loc_y = list([])
        self.loc_z = list([])

# TODO: element functions are not fully implemented
# Data structure for holding the element table
class Elems:
    pass

#-----------------------------------------------------------------------------
# Generic Vector Field Class and Inheriting Vector Classes
class Vector:
    def __init__(self):
        self.node_nums = list([])
        self.x = list([])
        self.y = list([])
        self.z = list([])

class Disp(Vector):
    pass

class Vel(Vector):
    pass

class Accel(Vector):
    pass

class Force(Vector):
    pass

#-----------------------------------------------------------------------------
# Generic Tensorial Field Class and Inheriting Tensor Classes
class Tensor2:
    def __init__(self):
        self.node_nums = list([])
        self.xx = list([])
        self.yy = list([])
        self.zz = list([])
        self.xy = list([])
        self.yz = list([])
        self.xz = list([])

class Stress(Tensor2):
    pass

class Strain(Tensor2):
    pass

#-----------------------------------------------------------------------------
# Data structure for holding the output fields and nodal locations
class FEData:
    def __init__(self,in_path,in_sim_num,dims):
        self.opts = Options(in_path,in_sim_num,dims)
        self.params = SimParams()
        self.nodes = Nodes()
        self.disp = Disp()
        self.stress = Stress()
        self.strain = Strain()
        self.force = Force()

# FUNCTIONS
#-----------------------------------------------------------------------------
# Reads the simulation parameters file into a data structure, one field is
# created for each parameter. This is not a core ANSYS file but is generated
# by the user to describe the simulation.
def read_params(file_path):
    # Create empty params structure to return
    params = SimParams()

    # File IO block
    with open(file_path,'r') as sim_file:
        # Read in all lines from the file
        all_lines = sim_file.readlines()

        # Loop over all lines in the file and split them if necessary
        delim_list = [',','|']
        delim_counts = []

        # Both flags need to be true so that we can use the headers as attributes
        # and then assign them the values stored in the data list
        header_flag = False
        data_flag = False
        headers = []
        data = []

        # Loop over all lines we read and find a header row followed by a data
        # row, once we have this we match the two and push them into the params
        # data structure
        for ss in all_lines:
            if ss:
                # See if we can find a delimiter in the string by counting them
                for dd in delim_list:
                    delim_counts.append(ss.count(dd))

                # If we find a delimiter we should split the string, process the
                # data and see if we have headers or data
                if sum(delim_counts) > 0:
                    max_ind = delim_counts.index(max(delim_counts))

                    # Split the string based on the found delimiter
                    split_line = ss.split(delim_list[max_ind])

                    # Based on the split line work out what type of data we are
                    # dealing with
                    for sl in split_line:
                        if sl.strip():
                            if tr.is_float(sl.strip()):
                                data_flag = True
                                data.append(float(sl.strip()))
                            else:
                                header_flag = True
                                headers.append(sl.strip())

                    # Check the state of both flags, if both are true then we set
                    # attributes in the SimParams struct and clear everything.
                    if header_flag and data_flag:
                        if len(headers) == len(data):
                            for pp in range(len(headers)):
                                setattr(params,headers[pp],data[pp])

                        headers = []
                        data = []
                        header_flag = False
                        data_flag = False
                    elif data_flag and not header_flag:
                        headers = []
                        data = []
                        header_flag = False
                        data_flag = False

                # Reset the delimiter count for reading the next line
                delim_counts = []

    # Return the params structure based on what we pulled from the file
    return params

#-----------------------------------------------------------------------------
# Reads in the nodal locations from ANSYS *.txt file
def read_nodes(fe_opts):
    # Create node empty node data structure to return
    fe_nodes = Nodes()

    file_path = fe_opts.get_node_loc_file()

    # Data read in specifiers
    val_spec = [1]*4 # [NodeNum,X,Y,Z]
    max_len = 20 # ANSYS max characters per number in node file e.g.: -0.450000000000E-001

    # Read the numeric data from the file into an array
    data_array = np.array(tr.read_data_by_spec(file_path,val_spec,max_len))

    # Push the data columns into the struct fields
    fe_nodes.nums = data_array[:,0]
    fe_nodes.loc_x = data_array[:,1]
    fe_nodes.loc_y = data_array[:,2]
    fe_nodes.loc_z = data_array[:,3]

    return fe_nodes
#-----------------------------------------------------------------------------
# Reads in the ANSYS element table from a *.txt file. Currently only works for
# 2D data because the node numbers spill onto the next row for 3D data.
# TODO: implement function to read the element table
'''
def read_elems(fe_opts):
    # Create empty structure to return
    fe_elems = Elems()

    if fe_opts.dims != 2:
        print('WARNING: Cannot read element table for non 2D data')
        return fe_elems

    # Compressed header to find in file
    header = 'ELEMMATTYPRELESYSECNODES'
    file_path = fe_opts.get_elem_tab_file()
    data_list = tr.read_data_after_header(file_path, header)
    data_array = np.array(data_list)

    fe_elems.nums = data_array[:,0]
    fe_elems.nodes = data_array[:,6:]

    return fe_elems
'''
#------------------------------------------------------------------------------
# READ NODAL FORCE VECTOR FILE
def read_node_force(fe_opts):
    # Create empty structure to return
    vector_str = 'Force'
    fe_vector = Force()

    # Data read in specifiers
    if fe_opts.dims == 3:
        val_spec = [1]*4 # [NodeNum,Fx,Fy,Fz]
    else:
        val_spec = [1]*3 # [NodeNum,Fx,Fy]

    max_len = 11 # ANSYS max characters per number in node force file e.g.: -0.2271E-10

    # If the next displacement file exists we should read it in
    ff = 1
    file_path = fe_opts.get_vector_frame_file(vector_str,ff)
    while os.path.exists(file_path):
        data_array = np.array(tr.read_data_by_spec(file_path,val_spec,max_len))

        # Node numbers don't change so grab them once
        if ff == 1:
            fe_vector.node_nums = data_array[:,0].transpose()

        # Append all displacement components as flat arrays
        fe_vector.x.append(data_array[:,1])
        fe_vector.y.append(data_array[:,2])
        if fe_opts.dims == 3:
            fe_vector.z.append(data_array[:,3])

        # Update file name for next iteration
        ff += 1
        file_path = fe_opts.get_vector_frame_file(vector_str,ff)

    fe_vector.x = np.array(fe_vector.x).transpose()
    fe_vector.y = np.array(fe_vector.y).transpose()
    if fe_opts.dims == 3:
        fe_vector.z = np.array(fe_vector.z).transpose()

    fe_vector.sum_x = np.sum(fe_vector.x,axis=0)
    fe_vector.sum_y = np.sum(fe_vector.y,axis=0)
    if fe_opts.dims == 3:
        fe_vector.sum_z = np.sum(fe_vector.z,axis=0)

    return fe_vector

#------------------------------------------------------------------------------
# READ VECTOR FIELD VARIABLES
def read_vector(fe_opts,in_vector,vector_str):
    # Create empty structure to return
    fe_vector = in_vector

    # Data read in specifiers
    val_spec = [1]*5 # [NodeNum,Ux,Uy,Uz,Usum]
    max_len = 13 # ANSYS max characters per number in disp file e.g.: -0.47336E-006

    # If the next displacement file exists we should read it in
    ff = 1
    file_path = fe_opts.get_vector_frame_file(vector_str,ff)
    while os.path.exists(file_path):
        data_array = np.array(tr.read_data_by_spec(file_path,val_spec,max_len))

        # Node numbers don't change so grab them once
        if ff == 1:
            fe_vector.node_nums = data_array[:,0].transpose()

        # Append all displacement components as flat arrays
        fe_vector.x.append(data_array[:,1])
        fe_vector.y.append(data_array[:,2])
        fe_vector.z.append(data_array[:,3])

        # Update file name for next iteration
        ff += 1
        file_path = fe_opts.get_vector_frame_file(vector_str,ff)

    fe_vector.x = np.array(fe_vector.x).transpose()
    fe_vector.y = np.array(fe_vector.y).transpose()
    fe_vector.z = np.array(fe_vector.z).transpose()

    return fe_vector

def read_disp(fe_opts):
    return read_vector(fe_opts,Disp(),'Disp')

def read_vel(fe_opts):
    return read_vector(fe_opts,Vel(),'Vel')

def read_accel(fe_opts):
    return read_vector(fe_opts,Accel(),'Accel')

#------------------------------------------------------------------------------
# READ TENSOR FIELD VARIABLES
def read_tensor2(fe_opts,in_tensor,tensor_str):
    # Create empty structure to return
    fe_tensor = in_tensor

    # Data read in specifiers
    val_spec = [1]*7 # [NodeNum,xx,yy,zz,xy,yz,xz]
    max_len = 13 # ANSYS max characters per number in tensor file e.g.: -0.32339E-006

    # If the next file exists we should read it in
    ff = 1
    file_path = fe_opts.get_tensor2_frame_file(tensor_str,ff)
    while os.path.exists(file_path):
        data_array = np.array(tr.read_data_by_spec(file_path,val_spec,max_len))

        # Node numbers don't change so grab them once
        if ff == 1:
            fe_tensor.node_nums = data_array[:,0].transpose()

        # Append all displacement components as flat arrays
        fe_tensor.xx.append(data_array[:,1])
        fe_tensor.yy.append(data_array[:,2])
        fe_tensor.zz.append(data_array[:,3])
        fe_tensor.xy.append(data_array[:,4])
        fe_tensor.yz.append(data_array[:,5])
        fe_tensor.xz.append(data_array[:,6])

        # Update file name for next iteration
        ff += 1
        file_path = fe_opts.get_tensor2_frame_file(tensor_str,ff)

    fe_tensor.xx = np.array(fe_tensor.xx).transpose()
    fe_tensor.yy = np.array(fe_tensor.yy).transpose()
    fe_tensor.zz = np.array(fe_tensor.zz).transpose()
    fe_tensor.xy = np.array(fe_tensor.xy).transpose()
    fe_tensor.yz = np.array(fe_tensor.yz).transpose()
    fe_tensor.xz = np.array(fe_tensor.xz).transpose()

    return fe_tensor

def read_strain(fe_opts):
    return read_tensor2(fe_opts,Strain(),'Strain')

def read_stress(fe_opts):
    return read_tensor2(fe_opts,Stress(),'Stress')

#-----------------------------------------------------------------------------
# trity Functions

def get_tensor_node_locs(all_nodes,tensor_node_nums):
    tensor_nodes = Nodes()
    for nn in range(all_nodes.nums.shape[0]):
        for ss in range(tensor_node_nums.shape[0]):
            if all_nodes.nums[nn] == tensor_node_nums[ss]:
                tensor_nodes.nums.append(all_nodes.nums[nn])
                tensor_nodes.loc_x.append(all_nodes.loc_x[nn])
                tensor_nodes.loc_y.append(all_nodes.loc_y[nn])
                tensor_nodes.loc_z.append(all_nodes.loc_z[nn])


    tensor_nodes.loc_x = np.array(tensor_nodes.loc_x)
    tensor_nodes.loc_y = np.array(tensor_nodes.loc_y)
    tensor_nodes.loc_z = np.array(tensor_nodes.loc_z)

    return tensor_nodes
