import numpy as np


class InnerProdNumpyArray(np.ndarray):
    def __new__(cls, input_array, transpose_scaling_factor=1.):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.transpose_scaling_factor = transpose_scaling_factor
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.transpose_scaling_factor = getattr(obj, 'transpose_scaling_factor', None)

    @property
    def T(self):
        return self.transpose_scaling_factor * self.transpose()
