try: 
    import cPickle as pickle
except ImportError:
    import pickle
import sys


''' If True, don't do anything that might be annoying. '''
_DISABLED = False

# Data set names
DATASET_RMS_ICP = 'RMS ICP'
DATASET_RMS_MERGE = 'RMS merge'

''' Data storage for plots.  

Each entry has a data set name (the key) and a list of values.  Structured
values (such as list and tuples) are allowed, but there is no check for
homogeneity of the data set.
'''
_DATA = {
    DATASET_RMS_ICP: [],
    DATASET_RMS_MERGE: [],
}

_OUTPUT_FOLDER = None


def disable_plotter():
    global _DISABLED
    _DISABLED = True


def enable_plotter():
    global _DISABLED
    _DISABLED = False


def output_folder(folder_name):
    ''' Set the storage folder for plots, if written to file. '''
    global _OUTPUT_FOLDER
    _OUTPUT_FOLDER = folder_name
    return _OUTPUT_FOLDER


def collect_data(data_set, data_point, debug):
    ''' Collect data and store it in a specific data set. '''
    if _DISABLED:
        return

    try:
        _DATA[data_set].append(data_point)
    except KeyError:
        if debug:
            print "Data set \"{}\" unknown.".format(data_set)
    except AttributeError:
        if debug > 1:
            print "Data set representation not as expected."


def store_data(pickle_file):
    '''
    Write recorded data set to file.

    Note that the argument is a file reference, not a file name!
    '''
    pickle.dump(_DATA, pickle_file)


def retrieve_data(pickle_file):
    '''
    Read a data set from a file.

    Note that the argument is a file reference, not a file name!
    '''
    _DATA = pickle.load(pickle_file)


def create_plots(plot_folder=None):
    '''
    Create all plots we want to use for the report. 
    
    When no file name is given, display the plot as a separate window.
    '''
    if _DISABLED:
        return
    create_rms_plot(plot_folder)


def create_rms_plot(plot_folder):
    if _DISABLED: 
        return
    pass
