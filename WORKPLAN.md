# Improved catalog I/O

## From in-memory data

- Get the data and rename columns
- Figure out the patch mode
    - Create patches from a sparse sample, as needed
    - Create the patch column, as needed
- Run a groupby to create patches
    - Write the patch data
    - Unloaded the data as needed
    - Collect the patch meta data
- Save the patch meta data

## From a file

- Get a list of column to load
- Figure out the patch mode
    - Create patches from a sparse sample (patches * 1000), read on the fly, as
      needed
- Run the full data in batches
    - Rename the columns
    - Create the patch column, as needed
    - Sink the data in patch files
- Read all patch files and collect the meta data
- Save the patch meta data

## Other improvements

- Add requirements(_dev).txt for easy setup of the environment
- Allow data access when patch is not loaded
    - When requesting an attribute, load only the required column(s) from the
      data
- Store one version of the tree
    - When caching, store the currently used binning, probably as json file
    - Build the trees once per patch and store them in the cache as pickle files
    - When the binning changes, rebuild the trees and replace the binning and
      pickle files
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
