import pytest
import pytest_mock
import pyvale.sensorsim as sens
import matplotlib.pyplot as plt

def drafttest_set_subplotsize(mocker):
    field_key: str = "temperature"
    mock_sensor_data = mocker.Mock()
    mock_sensor_data.tc_array = "whatever this is"
    sens.plot_time_traces(mock_sensor_data.tc_array, field_key)
