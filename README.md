=================================================================
	CS 6886: Systems Engineering for Deep Learning	
=================================================================

Assignment 2
Submitted by: Sooryakiran P
	      ME17B174
	      sooryakiran@smail.iitm.ac.in

-----------------------------------------------------------------

To compile everything,
    cd src
    make all

To run dummy inference on alexnet once,
    cd src && ./alexnet

To run dummy inference on alexnet 100 times,
    cs src && ./alexnet_100

-----------------------------------------------------------------

Directory Structure
===================

.
├── Figures		     // Directory containing all figures
│   └── ...	
│	
├── ME17B174_A2.pdf	     // Copy of report
├── Question.pdf	     // Assignment Questions
│
├── README.md		     // Goto line 1
├── data		     // Directory containing data used for
│			        plotting
├── report.odt		     // Editable report
└── src
    ├── Makefile	     // Makefile for compilation
    ├── TODO		     // TODO
    ├── alexnet.cpp	     // Alexnet source code
    ├── alexnet_100.cpp	     // Alexnet inf. 100 times source code
    ├── test.cpp	     // Source code for testind individual
    │				layers
    ├── util.cpp	     // Main implimentation of layers
    └── util.h		     // Header file for layer definitions

=================================================================

