# Improved catalog I/O

## Other improvements

- Add requirements(_dev).txt for easy setup of the environment
- Store one version of the tree
    - Create a placeholder class for the tree that stores the cache path for
      cheap passing between threads
- Prepare for MPI compatible code
    - Create a MPI version of the progress bar
    - Create a scheduler that can work with either MPI or multiprocessing
    - Add code that determines if MPI should be used (e.g. mpirun), see
      corresponding GH gist

## Required changes

- Check for impact of API changes, if too complex, release new major version
  once MPI is implemented
- Update the documentation and check for outdated text
- update the `__all__` statements
