#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>


#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))


struct Point3D {
    double x;
    double y;
    double z;
};


struct Point3D radec_to_3d(double ra, double dec) {
    double sin_ra = sin(ra);
    double sin_dec = sin(dec);
    // fast cosine from Pythagorean identity
    double cos_ra = sqrt(1 - sin_ra * sin_ra);
    double cos_dec = sqrt(1 - sin_dec * sin_dec);

    struct Point3D point = {
        .x = cos_ra * cos_dec,
        .y = sin_ra * cos_dec,
        .z = sin_dec,
    };
    return point;
}


double distance_points(struct Point3D point1, struct Point3D point2) {
    double dx = point1.x - point2.x;
    double dy = point1.y - point2.y;
    double dz = point1.z - point2.z;
    return sqrt(dx*dx + dy*dy + dz*dz);
}


int check_ra_dec_arrays_raise_pyerr(PyArrayObject *np_ra, PyArrayObject *np_dec) {
    // Check if inputs are NumPy arrays
    if (!PyArray_Check(np_ra)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array for 'ra'");
        return 1;
    }
    if (!PyArray_Check(np_dec)) {
        PyErr_SetString(PyExc_TypeError, "Expected a NumPy array for 'dec'");
        return 1;
    }

    // Check the array types
    int dtype_ra = PyArray_TYPE(np_ra);
    if (dtype_ra != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "unsupported data type for 'ra', expected 'float64'");
        return 1;
    }
    int dtype_dec = PyArray_TYPE(np_dec);
    if (dtype_dec != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "unsupported data type for 'dec', expected 'float64'");
        return 1;
    }

    // Check the array sizes
    npy_intp size_ra = PyArray_SIZE(np_ra);
    npy_intp size_dec = PyArray_SIZE(np_dec);
    if (size_ra == 0) {
        PyErr_SetString(PyExc_ValueError, "data is empty");
        return 1;
    }
    if (size_ra != size_dec) {
        PyErr_SetString(PyExc_IndexError, "lengths of 'ra' and 'dec' do not match");
        return 1;
    }
    return 0;
}


static PyObject* compute_center(PyObject* self, PyObject* args) {
    // pyargs: ra (array f64), dec (array f64)
    PyArrayObject *np_ra, *np_dec;
    if (!PyArg_ParseTuple(args, "OO", &np_ra, &np_dec)) {
        return NULL;
    }
    if (check_ra_dec_arrays_raise_pyerr(np_ra, np_dec)) {
        return NULL;
    }
    npy_intp size = PyArray_SIZE(np_ra);
    double* ra = PyArray_DATA(np_ra);
    double* dec = PyArray_DATA(np_dec);

    // compute convert to cartesian coordinates and compute the running mean
    struct Point3D coord;
    struct Point3D center = radec_to_3d(ra[0], dec[0]);
    for (npy_intp i = 1; i < size; i++) {
        coord = radec_to_3d(ra[i], dec[i]);
        double next_size = (double)i + 1.0;
        center.x += (coord.x - center.x) / next_size;
        center.y += (coord.y - center.y) / next_size;
        center.z += (coord.z - center.z) / next_size;
    }
    // Create and return Python tuple
    PyObject *x_center = PyFloat_FromDouble(center.x);
    PyObject *y_center = PyFloat_FromDouble(center.y);
    PyObject *z_center = PyFloat_FromDouble(center.z);
    PyObject *tuple = PyTuple_Pack(3, x_center, y_center, z_center);
    return tuple;
}


static PyObject* compute_radius(PyObject* self, PyObject* args) {
    // pyargs: ra (array f64), dec (array f64), x (double), y (double), z (double)
    PyArrayObject *np_ra, *np_dec;
    struct Point3D center;
    if (!PyArg_ParseTuple(args, "OOddd", &np_ra, &np_dec, &center.x, &center.y, &center.z)) {
        return NULL;
    }
    if (check_ra_dec_arrays_raise_pyerr(np_ra, np_dec)) {
        return NULL;
    }
    npy_intp size = PyArray_SIZE(np_ra);
    double* ra = PyArray_DATA(np_ra);
    double* dec = PyArray_DATA(np_dec);

    // compute convert to cartesian coordinates and compute maximum Euclidean distance
    double maxdist = 0.0;
    struct Point3D coord;
    for (npy_intp i = 0; i < size; i++) {
        coord = radec_to_3d(ra[i], dec[i]);
        double dist = distance_points(coord, center);
        maxdist = MAX(dist, maxdist);
    }
    double angdist = 2.0 * asin(maxdist / 2.0);
    // Create and return Python float
    PyObject* result = PyFloat_FromDouble(angdist);
    return result;
}


static PyMethodDef utils_methods[] = {
    {
        "_compute_center",
        compute_center,
        METH_VARARGS,
        "Compute the mean Euclidean coordinate from R.A. and Dec."
    },
    {
        "_compute_radius",
        compute_radius,
        METH_VARARGS,
        "Compute the maximum angular distance in radians from a Euclidean point and R.A. and Dec."
    },
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef utils_module = {
    PyModuleDef_HEAD_INIT,
    "_utils",      // Module name
    NULL,          // Module documentation
    -1,            // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    utils_methods  // Method table
};


PyMODINIT_FUNC PyInit__utils(void) {
    import_array();  // Initialize NumPy

    return PyModule_Create(&utils_module);
}
