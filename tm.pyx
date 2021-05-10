# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair

def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

cpdef target_mean_v3(data, x_name, y_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[long] y = np.asfortranarray(data[y_name], dtype=np.long)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.long)

    target_mean_v3_impl(result, x, y, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, long[:] x, long[:] y, const long nrow):
    #cdef dict value_dict = dict()
    #cdef dict count_dict = dict()

    #cdef tuple value_dict = tuple()
    #cdef tuple count_dict = tuple()
    
    cdef unordered_map[float,float] value_dict
    cdef unordered_map[float,float] count_dict
    
    cdef int i
    for i in range(nrow):
        #if x[i] not in value_dict.keys():
        if value_dict.count(x[i]) == 0:
            value_dict[x[i]] = y[i]
            count_dict[x[i]] = 1
        else:
            value_dict[x[i]] += y[i]
            count_dict[x[i]] += 1

    i=0
    for i in range(nrow):
        result[i] = (value_dict[x[i]] - y[i])/(count_dict[x[i]]-1)
