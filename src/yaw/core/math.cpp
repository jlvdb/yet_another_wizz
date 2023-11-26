#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


npy_intp get_size_checked(PyArrayObject *arr_obj) {
    // check that the array is 1-dim
    if (PyArray_NDIM(arr_obj) != 1) {
        PyErr_SetString(PyExc_ValueError, "input arrays must be 1-dimensional");
        return -1;
    }
    // check the array contains elements
    npy_intp size = PyArray_SIZE(arr_obj);
    if (size <= 0) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have size greater than 0");
        return -1;
    }
    // check if the arrays are of type float64
    if (PyArray_TYPE(arr_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "input arrays must be of type float64");
        return -1;
    }
    // check if the arrays are contiguous
    if (!PyArray_ISCONTIGUOUS(arr_obj)) {
        PyErr_SetString(PyExc_ValueError, "input arrays must be contiguous");
        return -1;
    }
    return size;
}


static PyObject* _rebin(PyObject* self, PyObject* args) {
    // pyargs: bins_new (array f64), bins_old (array f64), counts_old (array f64)
    PyArrayObject *bins_new_arrobj, *bins_old_arrobj, *counts_old_arrobj;

    // Parse the argument
    if (!PyArg_ParseTuple(
            args, "O!O!O!",
            &PyArray_Type, &bins_new_arrobj,
            &PyArray_Type, &bins_old_arrobj,
            &PyArray_Type, &counts_old_arrobj)) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected three numpy arrays");
        return nullptr;
    }
    // check the input data
    npy_intp size_new = get_size_checked(bins_new_arrobj);
    if (size_new == -1) return nullptr;
    npy_intp size_old = get_size_checked(bins_old_arrobj);
    if (size_old == -1) return nullptr;
    npy_intp n_bins_new = size_new - 1;
    npy_intp n_bins_old = size_old - 1;
    npy_intp size_counts = get_size_checked(counts_old_arrobj);
    if (size_counts == -1) return nullptr;
    if (size_counts != n_bins_old) {
        PyErr_SetString(PyExc_ValueError, "Number of bins and counts does not match");
        return nullptr;
    }

    // create the output numpy arrays with the same size and datatype
    PyObject *counts_new_obj = PyArray_EMPTY(1, &n_bins_new, NPY_FLOAT64, 0);
    if (!counts_new_obj) {
        PyErr_SetString(PyExc_TypeError, "Failed to allocate output arrays");
        Py_XDECREF(counts_new_obj);
        return nullptr;
    }

    // get pointers to the arrays
    double *bins_new_array = static_cast<double *>(PyArray_DATA(bins_new_arrobj));
    double *bins_old_array = static_cast<double *>(PyArray_DATA(bins_old_arrobj));
    double *counts_old_array = static_cast<double *>(PyArray_DATA(counts_old_arrobj));
    double *counts_new_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(counts_new_obj)));

    // iterate the new bins and check which of the old bins overlap with it
    npy_intp i_new, i_old;
    double zmin_n, zmax_n;
    double zmin_o, zmax_o;
    double zmin_overlap, zmax_overlap;
    double count, fraction;
    int contains, overlaps_min, overlaps_max;
    for (i_new = 0; i_new < n_bins_new; i_new++) {
        zmin_n = bins_new_array[i_new];
        zmax_n = bins_new_array[i_new+1];

        for (i_old = 0; i_old < n_bins_old; i_old++) {
            zmin_o = bins_old_array[i_old];
            zmax_o = bins_old_array[i_old+1];
            count = counts_old_array[i_old];

            // check for full or partial overlap
            contains = (zmin_n >= zmin_o) && (zmax_n < zmax_o);
            overlaps_min = (zmin_n <= zmin_o) && (zmax_n > zmin_o);
            overlaps_max = (zmin_n <= zmax_o) && (zmax_n > zmax_o);

            if (contains || overlaps_min || overlaps_max) {
                // compute fractional bin overlap
                zmin_overlap = MAX(zmin_o, zmin_n);
                zmax_overlap = MIN(zmax_o, zmax_n);
                fraction = (zmax_overlap - zmin_overlap) / (zmax_o - zmin_o);

                // assume uniform distribution of data in bin and increment
                // counts by the bin count weighted by the overlap fraction
                counts_new_array[i_new] += count * fraction;
            }
        }
    }
    // construct output array
    return counts_new_obj;
}


static PyMethodDef _math_methods[] = {
    {"_rebin", _rebin, METH_VARARGS, "Step-wise interpolate histogram counts to a new binning"},
    {nullptr, nullptr, 0, nullptr}  // Sentinel
};


static struct PyModuleDef _math_module = {
    PyModuleDef_HEAD_INIT,
    "_math",       // Module name
    nullptr,          // Module documentation
    -1,            // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _math_methods  // Method table
};


PyMODINIT_FUNC PyInit__math(void) {
    import_array();  // Initialize NumPy

    return PyModule_Create(&_math_module);
}
