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
    PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(array), NPY_ARRAY_OWNDATA);
    return array;
}


PyObject* map_to_dict(const std::unordered_map<int64_t, std::vector<double>>& data) {
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
