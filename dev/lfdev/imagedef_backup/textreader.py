'''
================================================================================
pyvale: the python computer aided validation engine

License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def check_list_float(in_list):
    check_float = [];
    for ii in range(len(in_list)):
        check_float.append(is_float(in_list[ii]))
    return check_float

def check_list_len(in_list,max_len):
    check_len = [];
    for ii in range(len(in_list)):
        check_len.append(len(in_list[ii]) > max_len)

    return check_len

def read_data_after_header(file_path,header):
    with open(file_path,'r') as data_file:
        # Read in all lines from the file
        all_lines = data_file.readlines()

        found_header = False
        found_data = False
        num_vals_per_line = -1
        line_num = 0

        data_array = []
        for ss in all_lines:
            ss = ss.strip()
            # If the string isn't empty it could be the header or the data we
            # want to read in
            if ss:
                # Strip out remaining whitespace and compare to the character
                # string for the header
                if ss.replace(' ','') == header:
                    found_header = True

                # If we have found the header we can now look for data. We
                # should skip any intial lines until we find numeric values.
                # Once we find numeric data we keep reading until we don't get
                # numeric data anymore
                elif found_header:
                    split_line = ss.split()

                    if num_vals_per_line > 0:
                        if num_vals_per_line != len(split_line):
                            raise ValueError('Number of data values in file changed at line {}\n'.format(line_num))
                    elif num_vals_per_line < 0:
                        num_vals_per_line = len(split_line)

                    # If the first element of the split line is numeric then
                    # we have data and we should read it in
                    if is_float(split_line[0]):
                        found_data = True
                        data_row = []
                        for sl in split_line:
                            if is_float(sl):
                                data_row.append(float(sl))
                        data_array.append(data_row)

                    # If the first element is not numeric but we have already
                    # found data then we should break because we are at the end
                    elif found_data:
                        break

            else:
                # If the string is empty and we have read in some data then we
                # are finished and can stop reading in more data. If we haven't
                # found data then this is a blank line after the header and
                # we keep going until we find some data.
                if found_data:
                    break

            line_num += 1

        return data_array


def read_data_by_spec(file_path,val_spec,max_len):
    with open(file_path,'r') as data_file:
        # Read in all lines from the file
        all_lines = data_file.readlines()

        data_array = []
        line_num = 0
        for ss in all_lines:
            ss = ss.strip()
            # If the string isn't empty it could be the data we want to read in
            if ss:
                split_line = ss.split()

                # Count the number of strings in the split that can be converted
                # to floats and the lengths of the strings
                check_float = check_list_float(split_line)
                check_len = check_list_len(split_line,max_len)

                missing_vals = sum(val_spec) - sum(check_float)

                # If the line matches the number and position of floats we are
                # expecting per line then we can read in the data.
                if check_float == val_spec:
                    data_row = []
                    # Go through the split line and extract the values based on the
                    # val_spec flags
                    for ii in range(len(split_line)):
                        if val_spec[ii]:
                            data_row.append(float(split_line[ii]))

                    # Append the row of data to the overall data array
                    data_array.append(data_row)

                # If there is a numeric value and one of the strings is longer than
                # expected there might be numeric data to partition
                elif missing_vals > 0 and sum(check_len) > 0:
                    data_row = []

                    # Based on how many long strings we have try to split all of
                    # them and put the whole thing into a list
                    temp_line = []
                    for sl in split_line:
                        # If the string is too long there is a problem to fix
                        if len(sl) > max_len:
                            # Split the string into equal parts
                            if (len(sl)/max_len).is_integer():
                                str_parts = [sl[pp:pp+max_len] for pp in range(0, len(sl), max_len)]
                                for sp in str_parts:
                                    temp_line.append(sp)
                            else:
                                temp_str = sl
                                str_parts = []
                                prev_split = 0
                                for cc in range(len(temp_str)-1):
                                    # if the next character is a minus then this
                                    # might have replaced whitespace between numbers
                                    # but we need to ignore '-' with 'e'
                                    if temp_str[cc+1] == '-' and is_float(temp_str[cc]):
                                        str_parts.append(temp_str[prev_split:cc+1])
                                        prev_split = cc+1

                                # At the end of the loop we grab the last string
                                if prev_split != len(temp_str):
                                    str_parts.append(temp_str[prev_split:len(temp_str)])

                                # Push all split strings onto the data line
                                for sp in str_parts:
                                    temp_line.append(sp)

                        # If there isn't a problem put the string onto the line
                        else:
                            temp_line.append(sl)

                    # Go through separated strings and check if we can convert
                    check_float_again =  check_list_float(temp_line)
                    if check_float_again == val_spec:
                        for tl in temp_line:
                            if is_float(tl):
                                data_row.append(float(tl))

                    # Check that the data row has the expected number of values,
                    # if so append it to the data array
                    if len(data_row) == sum(val_spec):
                        data_array.append(data_row)

            # Increment the line number for error checking
            line_num += 1

        # Send back the data list we have read in
        return data_array