"""
Module for sparse arrays using dictionaries. Inspired in part 
by ndsparse (https://launchpad.net/ndsparse) by Pim Schellart

Jan Erik Solem, Feb 9 2010.
solem@maths.lth.se (bug reports and feedback welcome)
"""

import numpy


class sparray(object):
    """ Class for n-dimensional sparse array objects using
        Python's dictionary structure.
    """
    def __init__(self, shape, default=0, dtype=float):
        
        self.__default = default #default value of non-assigned elements
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.__data = {}


    def __setitem__(self, index, value):
        """ set value to position given in index, where index is a tuple. """
        self.__data[index] = value

    def __getitem__(self, index):
        """ get value at position given in index, where index is a tuple. """
        return self.__data.get(index,self.__default)

    def __delitem__(self, index):
        """ index is tuples of element to be deleted. """
        if self.__data.has_key(index):
            del(self.__data[index])
            

    def __add__(self, other):
        """ Add two arrays. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] + other.__default
            out.__default = self.__default + other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val + other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __sub__(self, other):
        """ Subtract two arrays. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] - other.__default
            out.__default = self.__default - other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val - other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mul__(self, other):
        """ Multiply two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] * other.__default
            out.__default = self.__default * other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val * other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __div__(self, other):
        """ Divide two arrays (element wise). 
            Type of division is determined by dtype. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] / other.__default
            out.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val / other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __truediv__(self, other):
        """ Divide two arrays (element wise). 
            Type of division is determined by dtype. """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] / other.__default
            out.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val / other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __floordiv__(self, other):
        """ Floor divide ( // ) two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] // other.__default
            out.__default = self.__default // other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val // other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __mod__(self, other):
        """ mod of two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] % other.__default
            out.__default = self.__default % other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val % other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __pow__(self, other):
        """ power (**) of two arrays (element wise). """
        
        if self.shape == other.shape:
            out = self.__class__(self.shape, self.dtype)
            out.__data = self.__data.copy()
            for k in set.difference(set(out.__data.keys()),set(other.__data.keys())):
                out.__data[k] = out.__data[k] ** other.__default
            out.__default = self.__default ** other.__default
            for k in other.__data.keys():
                old_val = out.__data.setdefault(k,self.__default)
                out.__data[k] = old_val ** other.__data[k]        
            return out
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __iadd__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] + other.__default
            self.__default = self.__default + other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val + other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __isub__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] - other.__default
            self.__default = self.__default - other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val - other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imul__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] * other.__default
            self.__default = self.__default * other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val * other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __idiv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] / other.__default
            self.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val / other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __itruediv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] / other.__default
            self.__default = self.__default / other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val / other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ifloordiv__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] // other.__default
            self.__default = self.__default // other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val // other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __imod__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] % other.__default
            self.__default = self.__default % other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val % other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __ipow__(self, other):
        
        if self.shape == other.shape:
            for k in set.difference(set(self.__data.keys()),set(other.__data.keys())):
                self.__data[k] = self.__data[k] ** other.__default
            self.__default = self.__default ** other.__default
            for k in other.__data.keys():
                old_val = self.__data.setdefault(k,self.__default)
                self.__data[k] = old_val ** other.__data[k]        
            return self
        else:
            raise ValueError('Array sizes do not match. '+str(self.shape)+' versus '+str(other.shape))

    def __str__(self):
        return str(self.dense())

    def dense(self):
        """ Convert to dense NumPy array. """
        out = self.__default * numpy.ones(self.shape)
        for ind in self.__data:
            out[ind] = self.__data[ind]
        return out

    def sum(self):
        """ Sum of elements."""
        s = self.__default * numpy.array(self.shape).prod()
        for ind in self.__data:
            s += (self.__data[ind] - self.__default)
        return s


if __name__ == "__main__":
    
    #test cases
    
    #create a sparse array
    a = sparray((3,3),default=0,dtype=float)
    a[0,0] = 1
    a[1,1] = 2
    a[2,2] = 3
    print(a)
    print(a.dense())
    print(a.sum())

    #add two sparse arrays
    b = sparray((3,3),default=0,dtype=float)
    b[0,0] = 1
    b[1,1] = 2
    b[2,2] = 3
    c = a + b
    print(c)
    print(c.dense())
    print(c.sum())    