# Semiconductor Solver (Equilibrium and steady state simulation)

This project solves Poisson's eqaution in 1D and 2D (with cylindrical and cartesian geometry) to calculate equilibrium potential inside semiconductor. Steady state solution calculation for 1D semiconductor is also implemented.

Both Maxwell-Boltzmann and Fermi-Dirac statistics are available for calculating equilibrium carrier population. Poisson equation with Maxwell-Boltzmann statistics is called Poisson-Boltzmann equation which is non-linear. So, there are no analytic solutions available for most devices (analytical solutions available are only for special 1D doping profiles and boundary conditions).

Both potential(dirichlet) and potential derivative (Normal electric field) boundary conditions are considered. In most cases, boundary conditions should be consistent with charge conservation. Electric flux over the boundary of domain determines the charge within the domain. For charge neutrality, Electric flux over the boundary of domain should be 0. For 1D, this means same electric field at both ends of domain.  For 2D simulations, left and right bounadries have always reflecting boundary conditions. Top and bottom boundaries are set in input files.

For steady state solution poisson's equation along with electron and hole conservation equations are solved. SRH recombination-generation model is implemented with common lifetime for electrons and holes. In addition to SRH recombination, additional carrier generation can also be used by defining generation function in input file. Concept of quasi-fermi level along with Maxwell-Boltzman statistics is used to estimate non-equilibrium carrier distribution. Electric current, carrier population and potential profile are calculated. In addition to poisson's equation, two conservation equations (hole conservation and electron conservation) are used. Then, the three coupled equations relating potential,electron concentration and hole concentration are solved simultaneously.

For equilibrium, input file contains structure definition which contains mesh nodes with mesh spacing around nodes, boundary conditions on sections between nodes and a function which defines doping profile in the structure. Then, output of this structure definition (which is mesh and structure parameters) are set as input to the solve function. For steady state calculations, only 1D structures are allowed. First, the structure of interest needs to solved at equilibrium. Then, solution output should be loaded as input to steady state solver along with specifying biasing conditions at contacts. 

Constants for Silicon are entered in **constants_Si.py**.

Sample input and output can be viewed in **solver.ipynb**.

[Sample input and output](solver.md)

[Report](doc/report.pdf)

[Presentation](doc/slide.pdf)


