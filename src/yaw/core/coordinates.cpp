#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define SGN(x) ((x) < 0 ? -1 : 1)


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


extern "C" PyObject *coord_sky_to_sphere(PyObject *self, PyObject *args) {
    // Parse the input arguments
    PyArrayObject *ra_arrobj, *dec_arrobj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &ra_arrobj, &PyArray_Type, &dec_arrobj)) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected two numpy arrays");
        return nullptr;
    }
    // check the input data
    npy_intp size_ra = get_size_checked(ra_arrobj);
    if (size_ra == -1) return nullptr;
    npy_intp size_dec = get_size_checked(dec_arrobj);
    if (size_ra == -1) return nullptr;
    if (size_ra != size_dec) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // create the output numpy arrays with the same size and datatype
    PyObject *x_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    PyObject *y_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    PyObject *z_obj = PyArray_EMPTY(1, &size_ra, NPY_FLOAT64, 0);
    if (!x_obj || !y_obj || !z_obj) {
        // Cleanup in case of an error
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

    // compute the new coordinates
    double sin_ra, sin_dec, cos_ra, cos_dec;
    for (npy_intp i = 0; i < size_ra; ++i) {
        sin_ra = sin(ra_array[i]);
        sin_dec = sin(dec_array[i]);
        cos_ra = sqrt(1 - sin_ra * sin_ra);
        cos_dec = sqrt(1 - sin_dec * sin_dec);
        // compute final coordinates
        x_array[i] = cos_ra * cos_dec;
        y_array[i] = sin_ra * cos_dec;
        z_array[i] = sin_dec;
    }

    // return the arrays holding the new coordinates
    PyObject *pyresult = Py_BuildValue("OOO", x_obj, y_obj, z_obj);
    Py_XDECREF(x_obj);
    Py_XDECREF(y_obj);
    Py_XDECREF(z_obj);
    return pyresult;
}


extern "C" PyObject *coord_sphere_to_sky(PyObject *self, PyObject *args) {
    // Parse the input arguments
    PyArrayObject *x_arrobj, *y_arrobj, *z_arrobj;
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &x_arrobj, &PyArray_Type, &y_arrobj, &PyArray_Type, &z_arrobj)) {
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
    if (size_x != size_y || size_x != size_z) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // create the output numpy arrays with the same size and datatype
    PyObject *ra_obj = PyArray_EMPTY(1, &size_x, NPY_FLOAT64, 0);
    PyObject *dec_obj = PyArray_EMPTY(1, &size_x, NPY_FLOAT64, 0);
    if (!ra_obj || !dec_obj) {
        // Cleanup in case of an error
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

    // compute the new coordinates
    double x, y, z;
    double xsq_plus_ysq, r_d2, r_d3, ra_unwrapped;
    double pi2 = 2.0 * M_PI;
    for (npy_intp i = 0; i < size_x; ++i) {
        x = x_array[i];
        y = y_array[i];
        z = z_array[i];
        // compute final coordinates
        if (x == 0 && y == 0) {
            ra_array[i] = 0.0;
            dec_array[i] = SGN(z) * M_PI_2;
        } else {
            xsq_plus_ysq = x*x + y*y;
            r_d2 = sqrt(xsq_plus_ysq);
            r_d3 = sqrt(xsq_plus_ysq + z*z);
            ra_unwrapped = SGN(y) * acos(x / r_d2);
            ra_array[i] = fmod(ra_unwrapped, pi2);
            dec_array[i] = asin(z / r_d3);
        }
    }

    // return the arrays holding the new coordinates
    PyObject *pyresult = Py_BuildValue("OO", ra_obj, dec_obj);
    Py_XDECREF(ra_obj);
    Py_XDECREF(dec_obj);
    return pyresult;
}


extern "C" PyObject *radius_from_coord_sky(PyObject *self, PyObject *args) {
    // Parse the input arguments
    PyArrayObject *ra_arrobj, *dec_arrobj;
    double x_center, y_center, z_center;
    if (!PyArg_ParseTuple(
            args, "O!O!ddd",
            &PyArray_Type, &ra_arrobj, &PyArray_Type, &dec_arrobj,
            &x_center, &y_center, &z_center)) {
        PyErr_SetString(PyExc_TypeError, "invalid arguments, expected two numpy arrays and three doubles");
        return nullptr;
    }
    // check the input data
    npy_intp size_ra = get_size_checked(ra_arrobj);
    if (size_ra == -1) return nullptr;
    npy_intp size_dec = get_size_checked(dec_arrobj);
    if (size_ra == -1) return nullptr;
    if (size_ra != size_dec) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // get pointers to the arrays
    double *ra_array = static_cast<double *>(PyArray_DATA(ra_arrobj));
    double *dec_array = static_cast<double *>(PyArray_DATA(dec_arrobj));

    // compute convert to cartesian coordinates and compute maximum Euclidean distance
    double maxdist = 0.0;
    double dist;
    double sin_ra, sin_dec, cos_ra, cos_dec;
    double x, dx, y, dy, z, dz;
    for (npy_intp i = 0; i < size_ra; ++i) {
        sin_ra = sin(ra_array[i]);
        sin_dec = sin(dec_array[i]);
        cos_ra = sqrt(1 - sin_ra * sin_ra);
        cos_dec = sqrt(1 - sin_dec * sin_dec);
        // compute the distance
        x = cos_ra * cos_dec;
        y = sin_ra * cos_dec;
        z = sin_dec;
        dx = x - x_center;
        dy = y - y_center;
        dz = z - z_center;
        dist = sqrt(dx*dx + dy*dy + dz*dz);
        maxdist = MAX(dist, maxdist);
    }
    PyObject* result = PyFloat_FromDouble(maxdist);
    return result;
}


extern "C" PyObject *radius_from_coord_sphere(PyObject *self, PyObject *args) {
    // Parse the input arguments
    PyArrayObject *x_arrobj, *y_arrobj, *z_arrobj;
    double x_center, y_center, z_center;
    if (!PyArg_ParseTuple(
            args, "O!O!O!ddd",
            &PyArray_Type, &x_arrobj, &PyArray_Type, &y_arrobj, &PyArray_Type, &z_arrobj,
            &x_center, &y_center, &z_center)) {
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
    if (size_x != size_y || size_x != size_z) {
        PyErr_SetString(PyExc_ValueError, "input arrays must have equal size");
        return nullptr;
    }

    // get pointers to the arrays
    double *x_array = static_cast<double *>(PyArray_DATA(x_arrobj));
    double *y_array = static_cast<double *>(PyArray_DATA(y_arrobj));
    double *z_array = static_cast<double *>(PyArray_DATA(z_arrobj));

    // compute the new coordinates
    double maxdist = 0.0;
    double dist;
    double x, dx, y, dy, z, dz;
    for (npy_intp i = 0; i < size_x; ++i) {
        // compute the distance
        x = x_array[i];
        y = y_array[i];
        z = z_array[i];
        dx = x - x_center;
        dy = y - y_center;
        dz = z - z_center;
        dist = sqrt(dx*dx + dy*dy + dz*dz);
        maxdist = MAX(dist, maxdist);
    }
    PyObject* result = PyFloat_FromDouble(maxdist);
    return result;
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
