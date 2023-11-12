#include <Python.h>
#include <stdio.h>


#define BUF_SIZE 65536


static PyObject* _count_lines(PyObject* self, PyObject* args) {
    const char *filename;
    char buf[BUF_SIZE];
    FILE *file;
    long count = 0;
    size_t res, i;

    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    file = fopen(filename, "r");
    if (file == NULL) {
        PyErr_SetString(PyExc_FileNotFoundError, "File not found");
        return NULL;
    }

    for(;;)
    {
        res = fread(buf, 1, BUF_SIZE, file);
        for(i = 0; i < res; i++)
            if (buf[i] == '\n')
                count++;
        if (feof(file)) {
            break;
        }
    }

    fclose(file);

    return PyLong_FromLong(count);
}


static PyObject* _estimate_lines(PyObject* self, PyObject* args) {
    const char *filename;
    char buf[BUF_SIZE];
    FILE *file;
    long size, read = 0, count = 0;
    size_t res, i;
    float frac;

    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    file = fopen(filename, "r");
    if (file == NULL) {
        PyErr_SetString(PyExc_FileNotFoundError, "File not found");
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size = ftell(file);
    fseek(file, 0, SEEK_SET);

    for(;;)
    {
        res = fread(buf, 1, BUF_SIZE, file);
        read += res;
        for(i = 0; i < res; i++)
            if (buf[i] == '\n')
                count++;
        frac = (float)read / size;
        if (frac > 0.01) {
            count = (long)(count / frac);
            break;
        }
        if (feof(file)) {
            break;
        }
    }

    fclose(file);

    return PyLong_FromLong(count);
}


static PyMethodDef _streaming_methods[] = {
    {"_count_lines", _count_lines, METH_VARARGS, "Count lines in a file"},
    {"_estimate_lines", _estimate_lines, METH_VARARGS, "Estimate lines in a file by reading 1 percent of the file size"},
    {NULL, NULL, 0, NULL}  // Sentinel
};


static struct PyModuleDef _streaming_module = {
    PyModuleDef_HEAD_INIT,
    "_streaming",       // Module name
    NULL,               // Module documentation
    -1,                 // Size of per-interpreter state of the module, or -1 if the module keeps state in global variables.
    _streaming_methods  // Method table
};


PyMODINIT_FUNC PyInit__streaming(void) {
    return PyModule_Create(&_streaming_module);
}
