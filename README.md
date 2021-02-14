# KOSMA GILDAS DLC

This repository contains a number of functions using the gildas-python library.
This includes
   * spline fitting of spectra
   * despike spectra and storing result in an associated array
   * storing the index and quality measured in python/pandas table
   * plotting this python/pandas table with various groupings/histogram
   * interactive plotting of data (show spectra on click)
   * filter spectra using complex data base like queries
   * add assocated line windows to spectra using a map as an input
   * return FFT power at a given frequency
   * plot FFT using numpy libraries

The test folder contains example of the commands.

# Installation

Install with

    sudo -E make  install

The script will add links to the class build directory, if this is installed by root you may have persmission problems
and load it in class with

    @kosma-init
 
after that the command type:
    help  KOSMA\
And you should see the newly availble commands. Type help <COMMAND> to get the class help, or alternatively <COMMAND> /help to get the python help.






## generate index table from class and plotting values

    @kosma-init
    file in "tests/data/multiline.kosma"
    find
    set unit v
    set win 0 1
    get_index /get_rms /fft_focus_freq 170 /fft_focus_freq 20 /parameter tau_signal /reload
    ! plot rms and group by telescope
    plot_index /xaxis number /yaxis rms /group_by telescope
    ! plot rms and group by telescope
    plot_index /xaxis number /yaxis rms /group_by telescope /histogram 100
    ! for more options
    plot_index /help
    ! click on point to show spectra

# getting python-gildas to work

see instructions here:

<https://www.iram.fr/IRAMFR/GILDAS/doc/pdf/gildas-python.pdf>

## Anaconda problems

**excrept**

         - Conflict with Anaconda: when installed from the downloadable
           installer, Anaconda comes with a lot of development tools and
           libraries which override the default system ones, but not in a
           consistent way. This is proved to break software compilation and/or
           execution (Gildas and others). For example:
             astro: /home/user/anaconda/lib/libstdc++.so.6: version
                    GLIBCXX_3.4.21 not found (required by /home/user/
                    gildas-exe-jan17a/x86_64-ubuntu16.04-gfortran/lib/libatm.so)
           This happens because Anaconda gives precedence to its own libraries
           (duplicate but different versions of system ones) in the user's
           environment. There are several ways out of this issue:
             a) Install Anaconda from your OS repositories (instead of custom
                installation in the user account). This way, there should be
                a correct integration of Anaconda within the OS.
             b) Keep Anaconda in the user account, but "hide" it during Gildas
                installation and execution. In other words, you have to ensure
                that there is no reference to Anaconda installation paths in
                the environment variables $PATH and $LD_LIBRARY_PATH. Once this
                is done recompile, install, and try executing Gildas. If it
                runs fine, a permanent solution could be to use a shell
                function which loads Anaconda environment only when needed,
                e.g.
                  function anaconda() {
                      export PATH=...
                      export LD_LIBRARY_PATH=...
                      anaconda
                  }


