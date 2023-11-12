#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>


#define NUM_THREADS 4


struct ThreadData {
    char *fileContent;
    size_t start;
    size_t end;
    size_t lineCount;
};


void* count_lines_in_chunk(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;

    for (size_t i = data->start; i < data->end; i++) {
        if (data->fileContent[i] == '\n') {
            data->lineCount++;
        }
    }

    pthread_exit(NULL);
}


static PyObject* _count_lines(PyObject* self, PyObject* args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    int file = open(filename, O_RDONLY);
    if (file == -1) {
        PyErr_SetString(PyExc_FileNotFoundError, "File not found");
        return NULL;
    }

    struct stat fileStat;
    if (fstat(file, &fileStat) == -1) {
        PyErr_SetString(PyExc_OSError, "Error getting file size");
        close(file);
        return NULL;
    }

    char *fileContent = mmap(NULL, fileStat.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    if (fileContent == MAP_FAILED) {
        PyErr_SetString(PyExc_OSError, "Error mapping file to memory");
        close(file);
        return NULL;
    }

    pthread_t threads[NUM_THREADS];
    struct ThreadData threadData[NUM_THREADS];

    size_t chunkSize = fileStat.st_size / NUM_THREADS;

    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].fileContent = fileContent;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i == NUM_THREADS - 1) ? fileStat.st_size : (i + 1) * chunkSize;
        threadData[i].lineCount = 0;

        pthread_create(&threads[i], NULL, count_lines_in_chunk, (void*)&threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    size_t count = 0;
    for (size_t i = 0; i < NUM_THREADS; i++) {
        count += threadData[i].lineCount;
    }

    // Clean up
    munmap(fileContent, fileStat.st_size);
    close(file);

    return PyLong_FromSize_t(count);
}


static PyObject* _estimate_lines(PyObject* self, PyObject* args) {
    const char *filename;
    if (!PyArg_ParseTuple(args, "s", &filename)) {
        return NULL;
    }

    int file = open(filename, O_RDONLY);
    if (file == -1) {
        PyErr_SetString(PyExc_FileNotFoundError, "File not found");
        return NULL;
    }

    struct stat fileStat;
    if (fstat(file, &fileStat) == -1) {
        PyErr_SetString(PyExc_OSError, "Error getting file size");
        close(file);
        return NULL;
    }

    char *fileContent = mmap(NULL, fileStat.st_size, PROT_READ, MAP_PRIVATE, file, 0);
    if (fileContent == MAP_FAILED) {
        PyErr_SetString(PyExc_OSError, "Error mapping file to memory");
        close(file);
        return NULL;
    }

    pthread_t threads[NUM_THREADS];
    struct ThreadData threadData[NUM_THREADS];

    size_t chunkSize = (size_t)ceil(fileStat.st_size / NUM_THREADS / 100.0);
    size_t end = (size_t)ceil(fileStat.st_size / 100.0);

    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].fileContent = fileContent;
        threadData[i].start = i * chunkSize;
        threadData[i].end = (i == NUM_THREADS - 1) ? end : (i + 1) * chunkSize;
        threadData[i].lineCount = 0;

        pthread_create(&threads[i], NULL, count_lines_in_chunk, (void*)&threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    size_t count = 0;
    for (size_t i = 0; i < NUM_THREADS; i++) {
        count += threadData[i].lineCount;
    }

    // Clean up
    munmap(fileContent, fileStat.st_size);
    close(file);

    return PyLong_FromSize_t(count * 100);
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
