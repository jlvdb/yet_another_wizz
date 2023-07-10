#include <Python.h>
#include <numpy/arrayobject.h>


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


static PyObject* _rebin(PyObject* self, PyObject* args) {
    // pyargs: bins_new (array f64), bins_old (array f64), counts_old (array f64)
    PyArrayObject *np_bins_new, *np_bins_old, *np_counts_old;

    // Parse the argument
    if (!PyArg_ParseTuple(args, "OOO", &np_bins_new, &np_bins_old, &np_counts_old)) {
        return NULL;
    }

    // Check if inputs are NumPy arrays
    if (!PyArray_Check(np_bins_new)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array for 'bins_new'");
        return NULL;
    }
    if (!PyArray_Check(np_bins_old)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array for 'bins_old'");
        return NULL;
    }
    if (!PyArray_Check(np_counts_old)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array for 'counts_old'");
        return NULL;
    }

    // Get the array data
    double* bins_new = (double*)PyArray_DATA(np_bins_new);
    double* bins_old = (double*)PyArray_DATA(np_bins_old);
    double* counts_old = (double*)PyArray_DATA(np_counts_old);

    // Get the array dimensions
    npy_intp n_bins_new = PyArray_SHAPE(np_bins_new)[0] - 1;
    npy_intp n_bins_old = PyArray_SHAPE(np_bins_old)[0] - 1;
    npy_intp n_counts_old = PyArray_SHAPE(np_counts_old)[0];
    if (n_counts_old != n_bins_old) {
        PyErr_SetString(PyExc_ValueError, "Number of bins and counts does not match");
        return NULL;
    }
    // create the output array
    PyArray_Descr* descr = PyArray_DescrFromType(NPY_DOUBLE);
    npy_intp dims[1] = { n_bins_new };
    PyObject* np_counts_new = PyArray_Zeros(1, dims, descr, 0);
    double* counts_new = PyArray_DATA(np_counts_new);

    // iterate the new bins and check which of the old bins overlap with it
    npy_intp i_new, i_old;
    double zmin_n, zmax_n;
    double zmin_o, zmax_o;
    double zmin_overlap, zmax_overlap;
    double count, fraction;
    int contains, overlaps_min, overlaps_max;
    for (i_new = 0; i_new < n_bins_new; i_new++) {
        zmin_n = bins_new[i_new];
        zmax_n = bins_new[i_new+1];

        for (i_old = 0; i_old < n_bins_old; i_old++) {
            zmin_o = bins_old[i_old];
            zmax_o = bins_old[i_old+1];
            count = counts_old[i_old];

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
                counts_new[i_new] += count * fraction;
            }
        }
    }

    return np_counts_new;
}


static PyMethodDef _math_methods[] = {
    {"_rebin", _rebin, METH_VARARGS, "Step-wise interpolate histogram counts to a new binning"},
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef _math_module = {
    PyModuleDef_HEAD_INIT,
    "_math",       // Module name
    NULL,          // Module documentation
    -1,            // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _math_methods  // Method table
};


PyMODINIT_FUNC PyInit__math(void) {
    import_array();  // Initialize NumPy

    return PyModule_Create(&_math_module);
}
