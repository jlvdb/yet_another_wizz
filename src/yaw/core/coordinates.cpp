#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SGN(x) ((x) < 0 ? -1 : 1)


double euclidean_distance(double x, double y, double z, double x_c, double y_c, double z_c) {
    /**
     * TODO
     */
    double dx = x - x_c;
    double dy = y - y_c;
    double dz = z - z_c;
    return sqrt(dx*dx + dy*dy + dz*dz);
}


npy_intp get_size_checked(const PyArrayObject *arr_obj) {
    /**
     * TODO
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
    // check if the array is of type float64
    if (PyArray_TYPE(arr_obj) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_TypeError, "input arrays must be of type float64");
        return -1;
    }
    // check if the array is contiguous
    if (!PyArray_ISCONTIGUOUS(arr_obj)) {
        PyErr_SetString(PyExc_IndexError, "input arrays must be contiguous");
        return -1;
    }
    return size;
}


extern "C" PyObject* coord_sky_to_sphere(PyObject *self, PyObject *args) {
    /**
     * TODO
     */
    // pyargs: ra (array f64), dec (array f64)
    PyArrayObject *ra_arrobj, *dec_arrobj;
    if (!PyArg_ParseTuple(
        args, "O!O!",
        &PyArray_Type, &ra_arrobj,
        &PyArray_Type, &dec_arrobj
    )) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected two numpy arrays");
        return nullptr;
    }

    // check the input data
    npy_intp size_ra = get_size_checked(ra_arrobj);
    if (size_ra == -1) return nullptr;
    npy_intp size_dec = get_size_checked(dec_arrobj);
    if (size_ra == -1) return nullptr;
    // ensure that the arrays have matching size
    if (size_ra != size_dec) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // create the output numpy arrays with the same size and datatype
    PyObject *x_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    PyObject *y_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    PyObject *z_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    if (!x_obj || !y_obj || !z_obj) {
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate output arrays");
        Py_XDECREF(x_obj);
        Py_XDECREF(y_obj);
        Py_XDECREF(z_obj);
        return nullptr;
    }

    // get pointers to the arrays
    double *ra_array = static_cast<double *>(PyArray_DATA(ra_arrobj));
    double *dec_array = static_cast<double *>(PyArray_DATA(dec_arrobj));
    double *x_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(x_obj)));
    double *y_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(y_obj)));
    double *z_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(z_obj)));
    if (!ra_array || !dec_array || !x_array || !y_array || !z_array) {
        PyErr_SetString(PyExc_ValueError, "failed to access numpy array data");
        return nullptr;
    }


    // compute the new coordinates
    for (npy_intp i = 0; i < size_ra; ++i) {
        double sin_ra = sin(ra_array[i]);
        double sin_dec = sin(dec_array[i]);

        double cos_ra = sqrt(1 - sin_ra * sin_ra);
        double cos_dec = sqrt(1 - sin_dec * sin_dec);

        x_array[i] = cos_ra * cos_dec;
        y_array[i] = sin_ra * cos_dec;
        z_array[i] = sin_dec;
    }

    // return the arrays as tuple
    PyObject *pyresult = Py_BuildValue("OOO", x_obj, y_obj, z_obj);
    Py_XDECREF(x_obj);
    Py_XDECREF(y_obj);
    Py_XDECREF(z_obj);
    return pyresult;
}


extern "C" PyObject *coord_sphere_to_sky(PyObject *self, PyObject *args) {
    /**
     * TODO
     */
    // pyargs: x (array f64), y (array f64), z (array f64)
    PyArrayObject *x_arrobj, *y_arrobj, *z_arrobj;
    if (!PyArg_ParseTuple(
        args, "O!O!O!",
        &PyArray_Type, &x_arrobj,
        &PyArray_Type, &y_arrobj,
        &PyArray_Type, &z_arrobj
    )) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected three numpy arrays");
        return nullptr;
    }
    // check the input data
    npy_intp size_x = get_size_checked(x_arrobj);
    if (size_x == -1) return nullptr;
    npy_intp size_y = get_size_checked(y_arrobj);
    if (size_y == -1) return nullptr;
    npy_intp size_z = get_size_checked(z_arrobj);
    if (size_z == -1) return nullptr;
    // ensure that the arrays have matching size
    if (size_x != size_y || size_x != size_z) {
        PyErr_SetString(PyExc_IndexError, "input arrays must have equal size");
        return nullptr;
    }

    // create the output numpy arrays with the same size and datatype
    PyObject *ra_obj = PyArray_EMPTY(1, &size_x, NPY_FLOAT64, 0);
    PyObject *dec_obj = PyArray_EMPTY(1, &size_x, NPY_FLOAT64, 0);
    if (!ra_obj || !dec_obj) {
        PyErr_SetString(PyExc_RuntimeError, "failed to allocate output arrays");
        Py_XDECREF(ra_obj);
        Py_XDECREF(dec_obj);
        return nullptr;
    }

    // get pointers to the arrays
    double *x_array = static_cast<double *>(PyArray_DATA(x_arrobj));
    double *y_array = static_cast<double *>(PyArray_DATA(y_arrobj));
    double *z_array = static_cast<double *>(PyArray_DATA(z_arrobj));
    double *ra_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ra_obj)));
    double *dec_array = static_cast<double *>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(dec_obj)));
    if (!ra_array || !dec_array || !x_array || !y_array || !z_array) {
        PyErr_SetString(PyExc_ValueError, "failed to access numpy array data");
        return nullptr;
    }

    // compute the new coordinates
    const double PI2 = 2.0 * M_PI;
    for (npy_intp i = 0; i < size_x; ++i) {
        double x = x_array[i];
        double y = y_array[i];
        double z = z_array[i];

        // case where dec == +/- pi/2
        if (x == 0 && y == 0) {
            ra_array[i] = 0.0;
            dec_array[i] = SGN(z) * M_PI_2;
        }

        else {
            double xsq_plus_ysq = x*x + y*y;
            double r_d2 = sqrt(xsq_plus_ysq);
            double r_d3 = sqrt(xsq_plus_ysq + z*z);
            double ra_unwrapped = SGN(y) * acos(x / r_d2);
            ra_array[i] = fmod(ra_unwrapped, PI2);
            dec_array[i] = asin(z / r_d3);
        }
    }

    // return the arrays as tuple
    PyObject *pyresult = Py_BuildValue("OO", ra_obj, dec_obj);
    Py_XDECREF(ra_obj);
    Py_XDECREF(dec_obj);
    return pyresult;
}


extern "C" PyObject *radius_from_coord_sky(PyObject *self, PyObject *args) {
    /**
     * TODO
     */
    // pyargs: ra (array f64), dec (array f64), x_center (double), y_center (double), z_center (double)
    PyArrayObject *ra_arrobj, *dec_arrobj;
    double x_center, y_center, z_center;
    if (!PyArg_ParseTuple(
        args, "O!O!ddd",
        &PyArray_Type, &ra_arrobj,
        &PyArray_Type, &dec_arrobj,
        &x_center, &y_center, &z_center
    )) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected two numpy arrays and three doubles");
        return nullptr;
    }

    // check the input data
    npy_intp size_ra = get_size_checked(ra_arrobj);
    if (size_ra == -1) return nullptr;
    npy_intp size_dec = get_size_checked(dec_arrobj);
    if (size_ra == -1) return nullptr;
    // ensure that the arrays have matching size
    if (size_ra != size_dec) {
        PyErr_SetString(PyExc_IndexError, "input arrays must have equal size");
        return nullptr;
    }

    // get pointers to the arrays
    double *ra_array = static_cast<double *>(PyArray_DATA(ra_arrobj));
    double *dec_array = static_cast<double *>(PyArray_DATA(dec_arrobj));
    if (!ra_array || !dec_array) {
        PyErr_SetString(PyExc_ValueError, "failed to access numpy array data");
        return nullptr;
    }

    // compute convert to cartesian coordinates and compute maximum Euclidean distance
    double maxdist = 0.0;
    for (npy_intp i = 0; i < size_ra; ++i) {
        // convert coordinates
        double sin_ra = sin(ra_array[i]);
        double sin_dec = sin(dec_array[i]);
        double cos_ra = sqrt(1 - sin_ra * sin_ra);
        double cos_dec = sqrt(1 - sin_dec * sin_dec);
        double x = cos_ra * cos_dec;
        double y = sin_ra * cos_dec;
        double z = sin_dec;

        // compute the current maximum Euclidean distance
        double dist = euclidean_distance(x, y, z, x_center, y_center, z_center);
        maxdist = MAX(dist, maxdist);
    }

    // return float
    return PyFloat_FromDouble(maxdist);
}


extern "C" PyObject *radius_from_coord_sphere(PyObject *self, PyObject *args) {
    /**
     * TODO
     */
    // pyargs: x (array f64), y (array f64), z (array f64), x_center (double), y_center (double), z_center (double)
    PyArrayObject *x_arrobj, *y_arrobj, *z_arrobj;
    double x_center, y_center, z_center;
    if (!PyArg_ParseTuple(
        args, "O!O!O!ddd",
        &PyArray_Type, &x_arrobj,
        &PyArray_Type, &y_arrobj,
        &PyArray_Type, &z_arrobj,
        &x_center, &y_center, &z_center
    )) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected three numpy arrays and three doubles");
        return nullptr;
    }

    // check the input data
    npy_intp size_x = get_size_checked(x_arrobj);
    if (size_x == -1) return nullptr;
    npy_intp size_y = get_size_checked(y_arrobj);
    if (size_y == -1) return nullptr;
    npy_intp size_z = get_size_checked(z_arrobj);
    if (size_z == -1) return nullptr;
    // ensure that the arrays have matching size
    if (size_x != size_y || size_x != size_z) {
        PyErr_SetString(PyExc_IndexError, "input arrays must have equal size");
        return nullptr;
    }

    // get pointers to the arrays
    double *x_array = static_cast<double *>(PyArray_DATA(x_arrobj));
    double *y_array = static_cast<double *>(PyArray_DATA(y_arrobj));
    double *z_array = static_cast<double *>(PyArray_DATA(z_arrobj));
    if (!x_array || !y_array || !z_array) {
        PyErr_SetString(PyExc_ValueError, "failed to access numpy array data");
        return nullptr;
    }

    // compute the new coordinates
    double maxdist = 0.0;
    for (npy_intp i = 0; i < size_x; ++i) {
        // compute the current maximum Euclidean distance
        double dist = euclidean_distance(
            x_array[i], y_array[i], z_array[i], x_center, y_center, z_center
        );
        maxdist = MAX(dist, maxdist);
    }

    // return float
    return PyFloat_FromDouble(maxdist);
}


// Module method table
static PyMethodDef module_methods[] = {
    {
        "_coord_sky_to_sphere",
        coord_sky_to_sphere,
        METH_VARARGS,
        "Convert right ascension and declination in radians to Euclidean xyz coordinates.",
    },
    {
        "_coord_sphere_to_sky",
        coord_sphere_to_sky,
        METH_VARARGS,
        "Convert Euclidean xyz coordinates to right ascension and declination in radians.",
    },
    {
        "_radius_from_coord_sky",
        radius_from_coord_sky,
        METH_VARARGS,
        "Compute the Euclidean distance from a center point for points given in right ascension and declination in radians.",
    },
    {
        "_radius_from_coord_sphere",
        radius_from_coord_sphere,
        METH_VARARGS,
        "Compute the Euclidean distance from a center point for points given in Euclidean xyz coordinates.",
    },
    {nullptr, nullptr, 0, nullptr} // Sentinel
};


// Module definition
static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_coordinates",     // Module name
    nullptr,            // Module documentation
    -1,                 // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    module_methods
};


// Module initialization function
PyMODINIT_FUNC PyInit__coordinates(void) {
    import_array(); // Initialize NumPy

    // Return the created module
    return PyModule_Create(&module_definition);
}
