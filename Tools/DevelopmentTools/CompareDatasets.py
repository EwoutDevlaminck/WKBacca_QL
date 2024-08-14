"""
A class providing functions useful to compare different quantities from
different runs of the WKBeam code. The use of a class instead of just a
collection of functions allows to load multiple comparisons in the same session
and to access the datasets that ramain stored as attribute of the object.

The intended use of this module is for tracking differences while
developing the code.

All functions should support also old version of datasets, so that
comparisons can be made with old data as well.
"""

# Standard import statements
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import collections


# File objtects
WKBeamRayTracing = collections.namedtuple('WKBeamRayTracing', 
                                          ['h5file', 'version'])


# Main class for comparison of ray tracing files
class WKBeamRayDiff(object):

    """
    Object useful to compare two ray tracing runs.
    
    Usage:
    
      >>> compare = WKBeamRayDiff('filename0', 'filename1')
      >>> data = compare.get('name_of_dataset') # load data into array
      >>> compare.get_keys(i) # get the keys() of the dataset i=0,1
      >>> compare.datadiff(name) # diff for the dataset name
      >>> compare.view_traces(name, rayindex, componentindex=None) # plot traces
      >>> compare.view:rays(rayindex) # plot rays
 
    """

    # Fixed attributes (valid names of datasets)
    parameters = ['FreqGHz',
                  'Mode',
                  'antennapolangle',
                  'antennatorangle',
                  'beamwidth1',
                  'beamwidth2',
                  'centraleta1',
                  'centraleta2',
                  'curvatureradius1',
                  'curvatureradius2',
                  'fileindex',
                  'nmbrRays',
                  'rayStartX',
                  'rayStartY',
                  'rayStartZ']
    
    traces = ['TracesCorrectionFactor',
              'TracesMode',
              'TracesTime',
              'TracesPsi',
              'TracesNxNyNz',
              'TracesGroupVelocity']
    
    special_traces = ['TracesWfct', 'TracesXYZ',
                      'TracesNparallel', 'TracesphiN', 'TracesNumberScattEvents']

    valid_data_names = [parameters, traces, special_traces]
    
    # Initialization
    def __init__(self, filename0, filename1):
        
        """
        Initialization of the object.
        """

        self.files = self.__open_hdf5__(filename0, filename1)

        return None

    # Open the two hdf5 datasets
    def __open_hdf5__(self, filename0, filename1):
        
        """
        Loads the two hdf5 files to be compared.
        Returns (file1, file2) where file1 and file2 are h5py File objects.
        """
        
        # Load the datasets
        file0 = h5py.File(filename0, 'r')
        file1 = h5py.File(filename1, 'r')
        
        # Check the version (old files have key 'Traces', newest files
        # distinguish between 'TracesXYZ' for the trajectory and
        # 'TracesWfct' for the Wigner function.
        version0 = self.__check_version__(file0)
        version1 = self.__check_version__(file1)

        # Create the file object
        file0 = WKBeamRayTracing(h5file=file0, version=version0)
        file1 = WKBeamRayTracing(h5file=file1, version=version1)

        return file0, file1

    # Open the two hdf5 datasets
    def __check_version__(self, h5file):
        
        """
        Check the data format.
        """
        
        # Safety check
        assert isinstance(h5file, h5py._hl.files.File) 

        # Use the key 'Traces' and 'TracesXYZ' to distinguish
        if 'Traces' in h5file.keys():
            version = 'old'
        elif 'TracesXYZ' in h5file.keys():
            version = 'new'
        else:
            msg = "File {}: format not recognized.".format(h5file.filename)
            raise RuntimeError(msg)
            
        return version

    # Check dataset names
    def __check_name__(self, name):
        
        """
        Names of quantities should conform to the 'new' standard.
        """

        if name in self.parameters:
            data_type = 'parameter'
        elif name in self.traces:
            data_type = 'trace'
        elif name in self.special_traces:
            data_type = name 
        else:
            msg = """Name {} does not seem to be valid.
            Names of quantity should be consistent with the new format.
            Check the attribute valid_data_names for a list.
            """.format(name)
            raise RuntimeError(msg)
        
        return data_type

    # Retrive a parameter (same for both new and old formats)
    def __get_parameter__(self, parameter, index):
        
        """
        Retrieve the value of the parameter from the file.
        """
        
        return self.files[index].h5file.get(parameter)[()]
        
    # Retrieve a trace (this depends on the version of the file)
    def __get_trace__(self, trace, index):
        
        """
        Load trace data depending on the format.
        """
        
        f = self.files[index].h5file
        v = self.files[index].version

        try:
            rowdata = f.get(trace)[()]
        except AttributeError:
            msg = """ Trace {} not available. 
            Check with the method get_keys().""".format(trace)
            raise RuntimeError(msg)
        
        if v == 'new':
            data = rowdata
        elif v == 'old':
            # cut the first point
            data = rowdata[...,1:]
            # correct an old issue
            if trace == 'TracesCorrectionFactor':
                data = data[:,0,:]
            
        return data

    # Retrieve rays and Wigner function
    def __get_ray_and_Wigner_data__(self, index):

        """
        Load rays and Wigner function.
        """

        f = self.files[index].h5file
        v = self.files[index].version

        # Depending on the version remove the first point in time
        if v == 'old':
            rowdata = f.get('Traces')[()]
            rays = rowdata[:,0:3,1:] # cut the first point
            Wfct = rowdata[:,3,1:] # cut the first point
        elif v == 'new':
            rays = f.get('TracesXYZ')[()]
            Wfct = f.get('TracesWfct')[()]
        else:
            raise RuntimeError('File version not recognized')

        return rays, Wfct

    # Retrieve Nparallel or phiN trace (this depends on the version of the file)
    def __get_Nparallel_or_phiN__(self, trace, index):
        
        """
        Load trace data depending on the format.
        """
        
        f = self.files[index].h5file
        v = self.files[index].version

        msg = """ Trace {} not available. 
        Check with the method get_keys().""".format(trace)

        if v == 'new':
            try:
                data = f.get(trace)[()]
            except AttributeError:
                raise RuntimeError(msg)
                
        elif v == 'old':
            try:
                rowdata = f.get('TracesNparallelphiN')[()]
            except AttributeError:
                raise RuntimeError(msg)

            if trace == 'TracesNparallel':
                data = rowdata[:,0,1:] # cut the first point
            elif trace == 'TracesphiN':
                data = rowdata[:,1,1:] # cut the first point
            
        return data

    # Retrieve the number of scattering events for each ray
    def __get_number_of_scattering_events__(self, index):
        
        """
        Load the number of scattering events from each ray.
        """

        f = self.files[index].h5file
        v = self.files[index].version

        if v == 'new':
            data = f.get('TracesNumberScattEvents')[()]
                
        elif v == 'old':
            rowdata = f.get('Traces')[()]
            data = rowdata[:,0,0]   
            
        return data

    # Get-function wrapper
    def get(self, dataset, index):

        """
        Load the dataset into numpy array for file with index index.
        """

        data_type = self.__check_name__(dataset)
        
        if data_type == 'parameter':
            data = self.__get_parameter__(dataset, index)
        elif data_type == 'trace':
            data = self.__get_trace__(dataset, index)
        elif data_type == 'TracesXYZ':
            rays, Wigner = self.__get_ray_and_Wigner_data__(index)
            data = rays
        elif data_type == 'TracesWfct':
            rays, Wigner = self.__get_ray_and_Wigner_data__(index)
            data = Wigner
        elif data_type == 'TracesNparallel' or data_type == 'TracesphiN':
            data = self.__get_Nparallel_or_phiN__(dataset, index)
        elif data_type == 'TracesNumberScattEvents':
            data = self.__get_number_of_scattering_events__(index)

        return data

    # Get keys() of each dataset
    def get_keys(self, index):
        
        """
        list of available keys in the file of the given index=0,1.
        """
        
        return self.files[index].h5file.keys()
        
    # Compute the maximum difference of the value of two datasets
    def datadiff(self, dataset):
        
        """
        Maximum of the absolute value of the difference of a trace.
        """
    
        data0 = self.get(dataset, 0)
        data1 = self.get(dataset, 1)
        
        try:
            assert data0.shape == data1.shape
        except AssertionError:
            raise AssertionError('Datasets have different shape')
        
        return np.max(np.abs(data0 - data1))

    # Visualize a trace via plot
    def view_traces(self, name, rayindex, componentindex=None):
        
        """
        Plot a trace from the two dataset to visualize the differences.
        If componentindex is not passed it is assumed that the trace is taken
        from a scalar quantity and it is not needed to specify the component.
        """

        assert name not in self.parameters
        
        data0 = self.get(name, 0)
        data1 = self.get(name, 1)

        if componentindex == None:
            trace0 = data0[rayindex]
            trace1 = data1[rayindex]
        else:
            trace0 = data0[rayindex, componentindex]
            trace1 = data1[rayindex, componentindex]
            
        fig = plt.figure(figsize=(8,15))

        ax1 = fig.add_subplot(211)
        ax1.plot(trace0, label=name+' file 0')
        ax1.plot(trace1, label=name+' file 1')
        plt.legend()

        ax2 = fig.add_subplot(212)
        ax2.plot(trace0 - trace1)
        ax2.set_xlabel('integration points')
        ax2.set_title('difference')
        
        plt.show()

        return None
        
    # Visualize rays
    def view_rays(self, rayindex):
        
        """
        Plot two rays corresponding to the same ray index in the two
        data files.
        """
        
        rays0, Ws0 = self.__get_ray_and_Wigner_data__(0)
        rays1, Ws1 = self.__get_ray_and_Wigner_data__(1)

        ray0x = np.trim_zeros(rays0[rayindex,0], 'b')
        ray0y = np.trim_zeros(rays0[rayindex,1], 'b')
        ray0z = np.trim_zeros(rays0[rayindex,2], 'b')

        ray1x = np.trim_zeros(rays1[rayindex,0], 'b')
        ray1y = np.trim_zeros(rays1[rayindex,1], 'b')
        ray1z = np.trim_zeros(rays1[rayindex,2], 'b')
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(ray0x, ray0y, ray0z, label='ray0')
        ax.plot(ray1x, ray1y, ray1z, label='ray1')
        ax.set_xlabel('X', fontsize=20)
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20)
        ax.legend()

        plt.show()
        
        return None


# Main class for comparison of ray tracing files
class WKBeamBinDiff(object):

    """
    Object useful to compare two binned datasets.
    
    Usage:
    
      >>> compare = WKBeamBinDiff('filename0', 'filename1')
      >>> data = compare.get('name_of_dataset') # load data into array
      >>> compare.get_keys(i) # get the keys() of the dataset i=0,1
      >>> compare.datadiff(name) # diff for the dataset name
 
    """
    
    # Initialization
    def __init__(self, filename0, filename1):
        
        """
        Initialization of the object.
        """

        self.files = self.__open_hdf5__(filename0, filename1)

        return None

    # Open the two hdf5 datasets
    def __open_hdf5__(self, filename0, filename1):
        
        """
        Loads the two hdf5 files to be compared.
        Returns (file1, file2) where file1 and file2 are h5py File objects.
        """
        
        # Load the datasets
        file0 = h5py.File(filename0, 'r')
        file1 = h5py.File(filename1, 'r')
        
        return file0, file1
    # Get-function wrapper
    def get(self, dataset, index):

        """
        Load the dataset into numpy array for file with index index.
        """

        try:
            assert dataset in self.files[index].keys()
        except AssertionError:
            msg = """Dataset {} not present in file number {}
            In order to check the available keys one can use the method
            get.keys(index), where index is the file index = 0,1.
            """.format(dataset, index)
            raise RuntimeError(msg)

        return self.files[index].get(dataset)[()]        

    # Get keys() of each dataset
    def get_keys(self, index):
        
        """
        list of available keys in the file of the given index=0,1.
        """
        
        return self.files[index].keys()
        
    # Compute the maximum difference of the value of two datasets
    def datadiff(self, dataset):
        
        """
        Maximum of the absolute value of the difference of a trace.
        """
    
        data0 = self.get(dataset, 0)
        data1 = self.get(dataset, 1)

        try:
            assert data0.shape == data1.shape
        except AssertionError:
            raise AssertionError('Datasets have different shape')
        
        return np.max(np.abs(data0 - data1))

    # Compute the maximum difference of the value of two datasets
    def datadiff(self, dataset):
        
        """
        Maximum of the absolute value of the difference of a trace.
        """
    
        data0 = self.get(dataset, 0)
        data1 = self.get(dataset, 1)

        if type(data0) == str and type(data1) == str:
            if data0 == data1:
                print('\nDatasets {} are two identical strigns.'.format(dataset))
                return None
            else:
                print('\nDatasets {} are different.'.format(dataset))
                print('File 0 has ', data0)
                print('File 0 has ', data1)
                return None

        try:
            assert data0.shape == data1.shape
        except AssertionError:
            raise AssertionError('Datasets have different shape')
        
        return np.max(np.abs(data0 - data1))

    # Plot of the difference of binned traces
    def diffbinned2d(self):
        
        """
        Plot the difference of binned traced in two-dimensions
        by means of a pcolormesh plot.
        """
        
        data0 = self.get('BinnedTraces', 0)
        data1 = self.get('BinnedTraces', 1)
        
        try:
            assert len(data0.shape) == 3
            assert len(data1.shape) == 3
        except AssertionError:
            msg = "Either or both bined traces are not two-dimensional."
            raise RuntimeError(msg)
            
        fig = plt.figure(figsize=(8,15))
        ax1 = fig.add_subplot(211)
        p1 = ax1.pcolormesh(data0[:,:,0] - data1[:,:,0])
        plt.colorbar(p1, ax=ax1)
        ax2 = fig.add_subplot(212)
        p2 = ax2.pcolormesh(data0[:,:,1] - data1[:,:,1])
        plt.colorbar(p2, ax=ax2)
        plt.show()

        return None
