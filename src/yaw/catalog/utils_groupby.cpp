#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <thread>
#include <future>


std::unordered_map<int64_t, std::vector<double>> groupby_array(const std::vector<double> &data, const std::vector<int64_t> &indices) {
    int64_t size = data.size();
    std::unordered_map<int64_t, std::vector<double>> groupedData;
    groupedData.reserve(size); // Reserve space for better performance

    for (int64_t i = 0; i < size; ++i) {
        int64_t index = indices[i];
        groupedData[index].push_back(data[i]);
    }

    return groupedData;
}


// Helper struct to map C++ types to NumPy data types
template <typename T>
struct npy_traits {};

// Specializations for supported types
template <>
struct npy_traits<double> {
    static constexpr int dtype = NPY_FLOAT64;
};
template <>
struct npy_traits<float> {
    static constexpr int dtype = NPY_FLOAT32;
};
template <>
struct npy_traits<int64_t> {
    static constexpr int dtype = NPY_INT64;
};
template <>
struct npy_traits<int32_t> {
    static constexpr int dtype = NPY_INT32;
};


template <typename T>
int numpy_array_check_type(PyArrayObject *array) {
    if (PyArray_TYPE(array) != npy_traits<T>::dtype || !PyArray_ISCONTIGUOUS(array)) {
        return 1;
    }
    return 0;
}


template <typename T>
std::vector<T> numpy_array_to_vector(PyArrayObject *array) {
    npy_intp size = PyArray_SIZE(array);
    T* buffer = static_cast<T*>(PyArray_DATA(array));
    if (size <= 0 || buffer == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "failed to retrieve array size or data pointer");
        return std::vector<T>();
    }
    return std::vector<T>(buffer, buffer + size);
}


template <typename T>
PyObject* vector_to_numpy_array(const std::vector<T>& vec) {
    npy_intp size = vec.size();
    T* vec_copy = new T[size];
    std::copy(vec.begin(), vec.end(), vec_copy);

    int dtype;
    if (std::is_same<T, double>::value) {
        dtype = NPY_DOUBLE;
    } else if (std::is_same<T, float>::value) {
        dtype = NPY_FLOAT;
    } else if (std::is_same<T, int64_t>::value) {
        dtype = NPY_INT64;
    } else if (std::is_same<T, int32_t>::value) {
        dtype = NPY_INT32;
    } else {
        PyErr_SetString(PyExc_TypeError, "encountered unsupported output data type");
        delete[] vec_copy;
        return nullptr;
    }

    PyObject* array = PyArray_SimpleNewFromData(1, &size, dtype, vec_copy);
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
    PyObject* pyDict = PyDict_New();
    if (!pyDict) {
        PyErr_SetString(PyExc_RuntimeError, "failed to create output dictionary");
        return nullptr;
    }
    for (const auto& entry : data) {
        PyObject* key = PyLong_FromLong(entry.first);
        PyObject* value = vector_to_numpy_array(entry.second);
        if (!key || !value) {
            PyErr_SetString(PyExc_RuntimeError, "create patch key/value pair");
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
    // Parse the input arguments
    PyArrayObject *patchObj, *raObj, *decObj;
    PyArrayObject *weightObj = nullptr;
    PyArrayObject *redshiftObj = nullptr;
    if (!PyArg_ParseTuple(
            args, "O!O!O!|OO",
            &PyArray_Type, &patchObj,
            &PyArray_Type, &raObj,
            &PyArray_Type, &decObj,
            &weightObj,
            &redshiftObj)) {
        PyErr_SetString(PyExc_TypeError, "invalid argument, expected a numpy array");
        return nullptr;
    }

    // convert arrays
    if (numpy_array_check_type<int64_t>(patchObj)) {
        PyErr_SetString(PyExc_TypeError, "'patch' must be of type int64");
        return nullptr;
    }
    std::vector<int64_t> patch = numpy_array_to_vector<int64_t>(patchObj);
    if (numpy_array_check_type<double>(raObj)) {
        PyErr_SetString(PyExc_TypeError, "'ra' must be of type float64");
        return nullptr;
    }
    std::vector<double> ra = numpy_array_to_vector<double>(raObj);
    if (numpy_array_check_type<double>(decObj)) {
        PyErr_SetString(PyExc_TypeError, "'dec' must be of type float64");
        return nullptr;
    }
    std::vector<double> dec = numpy_array_to_vector<double>(decObj);
    if (numpy_array_check_type<double>(weightObj)) {
        PyErr_SetString(PyExc_TypeError, "'weight' must be of type float64");
        return nullptr;
    }
    std::vector<double> weight = numpy_array_to_vector<double>(weightObj);
    if (numpy_array_check_type<double>(redshiftObj)) {
        PyErr_SetString(PyExc_TypeError, "'redshift' must be of type float64");
        return nullptr;
    }
    std::vector<double> redshift = numpy_array_to_vector<double>(redshiftObj);

    // check the array sizes
    if (patch.size() == 0) {
        PyErr_SetString(PyExc_ValueError, "'patch' must have non-zero length");
        return nullptr;
    } else if (patch.size() != ra.size()) {
        PyErr_SetString(PyExc_IndexError, "length of 'ra' does not match 'patch'");
        return nullptr;
    } else if (patch.size() != dec.size()) {
        PyErr_SetString(PyExc_IndexError, "length of 'dec' does not match 'patch'");
        return nullptr;
    } else if (patch.size() != weight.size()) {
        PyErr_SetString(PyExc_IndexError, "length of 'weight' does not match 'patch'");
        return nullptr;
    } else if (patch.size() != redshift.size()) {
        PyErr_SetString(PyExc_IndexError, "length of 'redshift' does not match 'patch'");
        return nullptr;
    }

    // Use std::async to process each data array on a separate thread
    auto future1 = std::async(std::launch::async, groupby_array, std::ref(ra), std::ref(patch));
    auto future2 = std::async(std::launch::async, groupby_array, std::ref(dec), std::ref(patch));
    auto future3 = std::async(std::launch::async, groupby_array, std::ref(weight), std::ref(patch));
    auto future4 = std::async(std::launch::async, groupby_array, std::ref(redshift), std::ref(patch));
    auto result1 = future1.get();
    auto result2 = future2.get();
    auto result3 = future3.get();
    auto result4 = future4.get();

    // convert to python tuple of dicts
    PyObject* pydict1 = map_to_dict(result1);
    PyObject* pydict2 = map_to_dict(result2);
    PyObject* pydict3 = map_to_dict(result3);
    PyObject* pydict4 = map_to_dict(result4);
    if (!pydict1 || !pydict2 || !pydict3 || !pydict4) {
        return nullptr;
    }
    PyObject *pyresult = Py_BuildValue("OOOO", pydict1, pydict2, pydict3, pydict4);
    Py_XDECREF(pydict1);
    Py_XDECREF(pydict2);
    Py_XDECREF(pydict3);
    Py_XDECREF(pydict4);
    return pyresult;
}


// Module method table
static PyMethodDef module_methods[] = {
    {"_groupby_arrays", groupby_arrays, METH_VARARGS, "TODO"},
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
