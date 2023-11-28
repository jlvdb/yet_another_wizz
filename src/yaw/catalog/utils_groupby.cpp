#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <thread>
#include <future>


npy_intp get_size_checked(const PyArrayObject *arr_obj, int argidx, int dtype , const std::string &typestr) {
    /**
     * Get the size of a numpy array and check memory layout.
     *
     * Checks that the array is one dimensional, contains at least a single
     * element, elements have a specific datatype and the data is contiguous in
     * memory.
     *
     * @param arr_obj The numpy array object.
     * @param argidx The index in the functions argument list, used to format
     *               error messages.
     * @param dtype Data type the array elements should have, expressed as numpy
     *              datatype numerical code (e.g. NPY_DOUBLE).
     * @param typestr Descriptive name of the data type (e.g. 'float64').
     * @return Number of array elements (==length) or -1 if any check fails.
     */
    // check that the array is 1-dim
    if (PyArray_NDIM(arr_obj) != 1) {
        PyErr_SetString(PyExc_IndexError, "input arrays must be 1-dimensional");
        return -1;
    }
    // check the array contains elements
    npy_intp size = PyArray_SIZE(arr_obj);
    if (size <= 0) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have size greater than 0");
        return -1;
    }
    // check if the array is of the requested type
    if (PyArray_TYPE(arr_obj) != dtype) {
        const std::string message = "input array " + std::to_string(argidx) + " must be of type " + typestr;
        PyErr_SetString(PyExc_TypeError, message.c_str());
        return -1;
    }    // check if the array is contiguous
    if (!PyArray_ISCONTIGUOUS(arr_obj)) {
        PyErr_SetString(PyExc_IndexError, "input arrays must be contiguous");
        return -1;
    }
    return size;
}


std::unordered_map<int64_t, std::vector<double>> groupby(const std::vector<int64_t> &indices, const std::vector<double> &data) {
    /**
     * Split a data vector into subvectors based on grouping indices.
     *
     * Implements a simple group-by using a hashmap with a new vector for each
     * unique grouping index value. Data elements with the same grouping index
     * are collected in the same vector in the hashmap.
     *
     * @param indices Grouping index that determines how the data is split.
     * @param data Data vector to be split in groups.
     * @return Mapping from group index to data of group, stored as vector.
     */
    int64_t size = data.size();
    std::unordered_map<int64_t, std::vector<double>> groupedData;
    groupedData.reserve(size);  // Reserve space for better performance

    for (int64_t i = 0; i < size; ++i) {
        int64_t index = indices[i];
        groupedData[index].push_back(data[i]);
    }
    return groupedData;
}


template <typename T>
std::vector<T> numpy_array_to_vector(const PyArrayObject *array) {
    /**
     * Convert a 1-dim, contiguous numpy array with non-zero size to a vector
     * without copying.
     *
     * @param array The numpy array.
     * @return A vector pointing to the data of the numpy array. The vector is
     *         empty if the construction failed.
     */
    npy_intp size = PyArray_SIZE(array);
    T *buffer = static_cast<T*>(PyArray_DATA(array));
    if (size <= 0 || buffer == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "failed to retrieve array size or data pointer");
        return std::vector<T>();
    }
    return std::vector<T>(buffer, buffer + size);
}


template <typename T>
PyObject* vector_to_numpy_array(const std::vector<T>& vec) {
    /**
     * Copy a vector to a newly created numpy array.
     *
     * @param vec The vector to copy.
     * @return Numpy arrow object holing a copy of the data or a null pointer if
     *         an error occured.
     */
    npy_intp size = vec.size();
    T *vec_copy = new T[size];
    std::copy(vec.begin(), vec.end(), vec_copy);

    int dtype;
    if (std::is_same<T, double>::value) {
        dtype = NPY_DOUBLE;
    } else if (std::is_same<T, int64_t>::value) {
        dtype = NPY_INT64;
    } else {
        PyErr_SetString(PyExc_TypeError, "encountered unsupported output data type");
        delete[] vec_copy;
        return nullptr;
    }

    PyObject *array = PyArray_SimpleNewFromData(1, &size, dtype, vec_copy);
    if (array == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate memory for numpy array");
        delete[] vec_copy;  // Release memory in case of failure
        return nullptr;
    }
    // Set the flag to manage the memory
    // NOTE: this step may not be necessary anymore since the data is now copied
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(array), NPY_ARRAY_OWNDATA);
    return array;
}


PyObject* map_to_dict(const std::unordered_map<int64_t, std::vector<double>>& data) {
    /**
     * Convert an unordered map holding vectors to a python dict holding numpy
     * arrays.
     *
     * NOTE: This function seems to contain a memory leak, potentially related
     *       to reference couting of python dictionary keys. With each run, the
     *       memory profile of the python interpreter is slightly growing. When
     *       running `sys.getrefcount(result) - 1`, the reference count for the
     *       arrays is 1 as expected but for the keys as high as a few thousand.
     *
     * @param data The unordered map to repack.
     * @return A python dictionary containing copies of the array data contained
     *         in the input map.
     */
    PyObject *pyDict = PyDict_New();
    if (!pyDict) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create output dictionary");
        return nullptr;
    }
    for (const auto& entry : data) {
        PyObject *key = PyLong_FromLong(entry.first);
        PyObject *value = vector_to_numpy_array<double>(entry.second);
        if (!key || !value) {
            PyErr_SetString(PyExc_RuntimeError, "failed to create patch key/value pair");
            Py_XDECREF(key);
            Py_XDECREF(value);
            Py_DECREF(pyDict);
            return nullptr;
        }
        PyDict_SetItem(pyDict, key, value);
        Py_DECREF(key);
        Py_DECREF(value);
    }
    return pyDict;
}


extern "C" PyObject *groupby_arrays(PyObject *self, PyObject *args) {
    /**
     * Python-callable function that implements the groupby algorithm for an
     * index array and four data arrays, right ascension, declincation, weights
     * and redshift.
     *
     * The four data arrays are grouped by the index array values. The function
     * returns at tuple of python dictionaries, one for each data array. Each
     * dictionary maps from unique index values to a numpy array of the data
     * values in that group.
     *
     * NOTE: It was easier to implement a function with fixed parameters here
     *       and wrap them in an external python function. The external function
     *       also ensures that the arrays are contiguous and cast to the correct
     *       datatype.
     *
     * @return 4-tuple of dictionaries with mapping from int to numpy array.
     */
    // pyargs: patch (array i64), ra (array f64), dec (array f64), weight (array f64), redshift (array f64)
    PyArrayObject *patch_arrobj, *ra_arrobj, *dec_arrobj, *weight_arrobj, *redshift_arrobj;
    if (!PyArg_ParseTuple(
            args, "O!O!O!O!O!",
            &PyArray_Type, &patch_arrobj,
            &PyArray_Type, &ra_arrobj,
            &PyArray_Type, &dec_arrobj,
            &PyArray_Type, &weight_arrobj,
            &PyArray_Type, &redshift_arrobj
    )) {
        PyErr_SetString(PyExc_TypeError, "invalid argument, expected a numpy array");
        return nullptr;
    }

    // check the input data
    npy_intp size_patch = get_size_checked(patch_arrobj, 1, NPY_INT64, "int64");
    if (size_patch == -1) return nullptr;
    npy_intp size_ra = get_size_checked(ra_arrobj, 2, NPY_FLOAT64, "float64");
    if (size_ra == -1) return nullptr;
    npy_intp size_dec = get_size_checked(dec_arrobj, 3, NPY_FLOAT64, "float64");
    if (size_dec == -1) return nullptr;
    npy_intp size_weight = get_size_checked(weight_arrobj, 4, NPY_FLOAT64, "float64");
    if (size_weight == -1) return nullptr;
    npy_intp size_redshift = get_size_checked(redshift_arrobj, 5, NPY_FLOAT64, "float64");
    if (size_redshift == -1) return nullptr;
    // ensure that the arrays have matching size
    if (size_patch != size_ra || size_patch != size_dec || size_patch != size_weight || size_patch != size_redshift) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // get data as vectors
    std::vector<int64_t> patch_vec = numpy_array_to_vector<int64_t>(patch_arrobj);
    std::vector<double> ra_vec = numpy_array_to_vector<double>(ra_arrobj);
    std::vector<double> dec_vec = numpy_array_to_vector<double>(dec_arrobj);
    std::vector<double> weight_vec = numpy_array_to_vector<double>(weight_arrobj);
    std::vector<double> redshift_vec = numpy_array_to_vector<double>(redshift_arrobj);
    if (patch_vec.size() == 0 || ra_vec.size() == 0 || dec_vec.size() == 0 || weight_vec.size() == 0 || redshift_vec.size() == 0) {
        PyErr_SetString(PyExc_ValueError, "failed to access numpy array data");
        return nullptr;
    }

    // Use std::async to process each data array on a separate thread
    auto future_ra = std::async(
        std::launch::async, groupby, std::ref(patch_vec), std::ref(ra_vec)
    );
    auto future_dec = std::async(
        std::launch::async, groupby, std::ref(patch_vec), std::ref(dec_vec)
    );
    auto future_weight = std::async(
        std::launch::async, groupby, std::ref(patch_vec), std::ref(weight_vec)
    );
    auto future_redshift = std::async(
        std::launch::async, groupby, std::ref(patch_vec), std::ref(redshift_vec)
    );
    auto result_ra = future_ra.get();
    auto result_dec = future_dec.get();
    auto result_weight = future_weight.get();
    auto result_redshift = future_redshift.get();

    // convert to python tuple of dicts[int, array]
    PyObject *dict_ra = map_to_dict(result_ra);
    PyObject *dict_dec = map_to_dict(result_dec);
    PyObject *dict_weight = map_to_dict(result_weight);
    PyObject *dict_redshift = map_to_dict(result_redshift);
    if (!dict_ra || !dict_dec || !dict_weight || !dict_redshift) {
        Py_XDECREF(dict_ra);
        Py_XDECREF(dict_dec);
        Py_XDECREF(dict_weight);
        Py_XDECREF(dict_redshift);
        return nullptr;
    }
    PyObject *pyresult = Py_BuildValue("OOOO", dict_ra, dict_dec, dict_weight, dict_redshift);
    Py_DECREF(dict_ra);
    Py_DECREF(dict_dec);
    Py_DECREF(dict_weight);
    Py_DECREF(dict_redshift);
    return pyresult;
}


// Module method table
static PyMethodDef module_methods[] = {
    {
        "_groupby_arrays",
        groupby_arrays,
        METH_VARARGS,
        "Run a groupby operation based on a patch index array applied to a ra, dec, weight and redshift array"
    },
    {nullptr, nullptr, 0, nullptr} // Sentinel
};


// Module definition
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_groupby",         // Module name
    nullptr,            // Module documentation
    -1,                 // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    module_methods
};


// Module initialization function
PyMODINIT_FUNC PyInit__groupby(void) {
    import_array(); // Initialize NumPy

    // Return the created module
    return PyModule_Create(&module_definition);
}
