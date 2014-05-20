try: 
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import matplotlib.pyplot as plt


class Plotter(object):
    ''' Creates plots from collected data. '''

    # Data set names
    # Each data set identifier is defined by a tuple:
    #     (plot title, x_label, y_label)
    DATASET_RMS_ICP = ('RMS ICP', 'iteration', 'RMS')
    DATASET_RMS_MERGE = ('RMS merge', 'iteration', 'RMS')

    ''' If True, don't do anything that might be time consuming/graphical. '''
    _disabled = False

    ''' Where plots will be stored. If None, only show them. '''
    _output_folder = None

    ''' Data storage for plots.  

    Each entry has a data set name (the key) and a list of values.
    Structured values (such as list and tuples) are allowed, but there is
    no check for homogeneity of the data set.
    '''
    _data = {
            DATASET_RMS_ICP: [],
            DATASET_RMS_MERGE: [],
    }


    def __del__(self):
        sys.stdout.write("Killing {}.{}\n".format(__name__,
                                                  self.__class__.__name__))
        with open('pickle', 'w') as f:
            self.store_data(f)
        sys.stdout.flush()


    @classmethod
    def disable(cls):
        cls._disabled = True


    @classmethod
    def enable(cls):
        cls._disable = False


    @classmethod
    def is_disabled(cls):
        return cls._disabled;


    @classmethod
    def output_folder(cls, folder_name):
        ''' Set the storage folder for plots, if written to file. '''
        cls._output_folder = folder_name
        return cls._output_folder


    @classmethod
    def collect_data(cls, data_set, data_point, debug):
        ''' Collect data and store it in a specific data set. '''
        if cls._disabled:
            return

        try:
            cls._data[data_set].append(data_point)
        except KeyError:
            if debug:
                print "Data set \"{}\" unknown.".format(data_set)
        except AttributeError:
            if debug > 1:
                print "Data set representation not as expected."


    @classmethod
    def store_data(cls, pickle_file):
        '''
        Write recorded data set to file.

        Note that the argument is a file reference, not a file name!
        '''
        pickle.dump(cls._data, pickle_file)


    @classmethod
    def retrieve_data(cls, pickle_file):
        '''
        Read a data set from a file.

        Note that the argument is a file reference, not a file name!
        '''
        cls._data = pickle.load(pickle_file)


    def create_plots(self):
        '''
        Create all plots we want to use for the report. 
        
        When no file name is given, display the plot as a separate window.
        '''
        if self.is_disabled():
            return
        for params, data in self._data:
            self._plot(data, *params)


    def _plot(data, name, xlabel, ylabel, zipped_data=False, drawing='r'):
        if zipped_data:
            data, data_y = zip(*data)
        else:
            data_y = range(len(data))

        plt.plot(data, data_y, drawing)
        plt.plot.xlabel(xlabel)
        plt.plot.ylabel(ylabel)
        plt.title(name)

        plt.show()

plotter = Plotter()
