#include <Python.h>
#include <numpy/arrayobject.h>

static PyArrayObject *ensure_double_array(PyObject *);
int check_array_dim(PyArrayObject *, const int);
void parse_totals(PyObject *, PyObject *, PyArrayObject *, PyArrayObject *);
void parse_counts(PyObject *, PyArrayObject *);

// helper functions ////////////////////////////////////////////////////////////

static PyArrayObject *ensure_double_array(PyObject *array_like) {
    // TODO
    // TODO
    PyArrayObject *contiguous;
    
    if (PyArray_IS_C_CONTIGUOUS(array_like)) {
        contiguous = (PyArrayObject *)array_like;
        if (PyArray_TYPE(contiguous) == NPY_DOUBLE) {
            Py_INCREF(array_like);
            return contiguous;
        }
    }

    PyArray_Descr *descr = PyArray_DescrFromType(NPY_DOUBLE);
    if (descr == NULL) {
        PyErr_SetString(PyExc_TypeError, "failed to create dtype");
        return NULL;
    }

    contiguous = (PyArrayObject *)PyArray_FromAny(
        array_like,
        descr,  // refcount decremented
        0,
        0,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ENSURECOPY,
        NULL
    );
    if (contiguous == NULL) {
        PyErr_SetString(PyExc_TypeError, "failed to convert input to numpy array");
        return NULL;
    }
    return contiguous;
}

int check_array_dim(PyArrayObject *array, const int ndim) {
    // TODO
    // TODO
    if (PyArray_NDIM(array) != ndim) {
        PyErr_SetString(PyExc_ValueError, "input array does not have expected dimensions");
        return 0;
    }
    return 1;
}

void parse_totals(PyObject *obj1, PyObject *obj2, PyArrayObject *array1, PyArrayObject *array2) {
    // TODO
    // TODO
    array1 = ensure_double_array(obj1);
    array2 = ensure_double_array(obj2);
    if (array1 == NULL || array2 == NULL) {
        goto error;
    }

    const int ndim = 2;
    if (!check_array_dim(array1, ndim) || !check_array_dim(array2, ndim)) {
        goto error;
    }

    npy_intp *shape1 = PyArray_DIMS(array1);
    npy_intp *shape2 = PyArray_DIMS(array2);
    if (shape1[0] != shape2[0] || shape1[1] != shape2[1]) {
        PyErr_SetString(PyExc_ValueError, "shape of input arrays is not equal");
        goto error;
    }

    return;

error:
    Py_XDECREF(array1);
    Py_XDECREF(array2);
    array1 == NULL;
    array2 == NULL;
}

void parse_counts(PyObject *obj, PyArrayObject *array) {
    // TODO
    // TODO
    array = ensure_double_array(obj);
    if (array == NULL) {
        goto error;
    }

    if (!check_array_dim(array, 3)) {
        goto error;
    }

    npy_intp *shape = PyArray_DIMS(array);
    if (shape[1] != shape[2]) {
        PyErr_SetString(PyExc_ValueError, "input array must have shape (num_bins, num_patches, num_patches)");
        goto error;
    }

    return;

error:
    Py_XDECREF(array);
    array == NULL;
}

// public functions ////////////////////////////////////////////////////////////

static PyObject *totals_sum_patches_cross(PyObject* self, PyObject* args) {
    // TODO
    // TODO
    PyObject *totals1_obj, *totals2_obj;
    if (!PyArg_ParseTuple(args, "OO", &totals1_obj, &totals2_obj)) {
        return NULL;
    }

    PyArrayObject *totals1_arr, *totals2_arr;
    parse_totals(totals1_obj, totals2_obj, totals1_arr, totals2_arr);
    if (totals1_arr == NULL || totals2_arr == NULL) {
        Py_XDECREF(totals1_arr);
        Py_XDECREF(totals2_arr);
        return NULL;
    }

    npy_intp *shape = PyArray_DIMS(totals1_arr);
    npy_intp num_bins = shape[0];
    npy_intp num_patches = shape[1];

    npy_intp result_shape[1] = {num_bins};
    PyObject *result_obj = PyArray_SimpleNew(1, result_shape, NPY_DOUBLE);
    if (result_obj == NULL) {
        Py_DECREF(totals1_arr);
        Py_DECREF(totals2_arr);
        return NULL;
    }

    double *totals1 = (double *)PyArray_DATA(totals1_arr);
    double *totals2 = (double *)PyArray_DATA(totals2_arr);
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_obj);

    for (int b = 0; b < num_bins; ++b) {
        double *row1 = totals1 + b * num_patches;
        double *row2 = totals2 + b * num_patches;

        double sum = 0.0;
        for (int p1 = 0; p1 < num_patches; ++p1) {
            double t1 = row1[p1];
            for (int p2 = 0; p2 < num_patches; ++p2) {
                sum += t1 * row2[p2];
            }
        }
        result[b] = sum;
    }

    Py_DECREF(totals1_arr);
    Py_DECREF(totals2_arr);
    return result;
}

static PyObject *totals_sum_patches_auto(PyObject* self, PyObject* args) {
    // TODO
    // TODO
    PyObject *totals1_obj, *totals2_obj;
    if (!PyArg_ParseTuple(args, "OO", &totals1_obj, &totals2_obj)) {
        return NULL;
    }

    PyArrayObject *totals1_arr, *totals2_arr;
    parse_totals(totals1_obj, totals2_obj, totals1_arr, totals2_arr);
    if (totals1_arr == NULL || totals2_arr == NULL) {
        Py_XDECREF(totals1_arr);
        Py_XDECREF(totals2_arr);
        return NULL;
    }

    npy_intp *shape = PyArray_DIMS(totals1_arr);
    npy_intp num_bins = shape[0];
    npy_intp num_patches = shape[1];

    npy_intp result_shape[1] = {num_bins};
    PyObject *result_obj = PyArray_SimpleNew(1, result_shape, NPY_DOUBLE);
    if (result_obj == NULL) {
        Py_DECREF(totals1_arr);
        Py_DECREF(totals2_arr);
        return NULL;
    }

    double *totals1 = (double *)PyArray_DATA(totals1_arr);
    double *totals2 = (double *)PyArray_DATA(totals2_arr);
    double *result = (double *)PyArray_DATA((PyArrayObject *)result_obj);

    for (int b = 0; b < num_bins; ++b) {
        double *row1 = totals1 + b * num_patches;
        double *row2 = totals2 + b * num_patches;

        double sum = 0.0;
        for (int p1 = 0; p1 < num_patches; ++p1) {
            double t1 = row1[p1];
            sum += 0.5 * t1 * row2[p1];  // diagonal
            for (int p2 = p1 + 1; p2 < num_patches; ++p2) {
                sum += t1 * row2[p2];
            }
        }
        result[b] = sum;
    }

    Py_DECREF(totals1_arr);
    Py_DECREF(totals2_arr);
    return result;
}

static PyMethodDef methods[] = {
    {"totals_sum_patches_cross", totals_sum_patches_cross, METH_VARARGS, "TODO"},
    {"totals_sum_patches_auto", totals_sum_patches_auto, METH_VARARGS, "TODO"},
    {"totals_sum_jackknife_cross", totals_sum_jackknife_cross, METH_VARARGS, "TODO"},
    {"totals_sum_jackknife_auto", totals_sum_jackknife_auto, METH_VARARGS, "TODO"},

    {"counts_sum_patches_cross", counts_sum_patches_cross, METH_VARARGS, "TODO"},
    {"counts_sum_patches_auto", counts_sum_patches_auto, METH_VARARGS, "TODO"},
    {"counts_sum_jackknife_cross", counts_sum_jackknife_cross, METH_VARARGS, "TODO"},
    {"counts_sum_jackknife_auto", counts_sum_jackknife_auto, METH_VARARGS, "TODO"},

    {NULL, NULL, 0, NULL}  // Sentinel
};


// module configuration ////////////////////////////////////////////////////////

static struct PyModuleDef jackknife_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "yaw.utils.jackknife",
    .m_doc = "TODO",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit_balltree(void) {
    import_array();
    if (PyErr_Occurred()) {
        PyErr_SetString(PyExc_ImportError, "failed to import NumPy array module");
        return NULL;
    }

    PyObject *module = PyModule_Create(&jackknife_module);
    if (module == NULL) {
        return NULL;
    }

    return module;
}
