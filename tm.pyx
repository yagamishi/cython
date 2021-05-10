# distutils: language=c++
import numpy as np
cimport numpy as np
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair

cpdef target_mean_v3(data, x_name, y_name):
    cdef long nrow = data.shape[0]
    cdef np.ndarray[double] result = np.asfortranarray(np.zeros(nrow), dtype=np.float64)
    cdef np.ndarray[long] y = np.asfortranarray(data[y_name], dtype=np.long)
    cdef np.ndarray[long] x = np.asfortranarray(data[x_name], dtype=np.long)

    target_mean_v3_impl(result, x, y, nrow)
    return result

cdef void target_mean_v3_impl(double[:] result, long[:] x, long[:] y, const long nrow):
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
