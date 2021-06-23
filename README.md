# ADAL-ineq
Extended ADAL for Maximum Clique and Graph Coloring problems
------------------------------------------------------------

The present software package includes scripts for solving
SDP problems in general form:

         min <C,X> 
         s.t. <Ai,X> <= bi, i=1,...,l;
              <Aj,X>  = bj, j=l+1,...,m;
              X >= L
              X psd
         
In particular it contains the ADMM for large-scale SDPs in general form and the post-processing 
procedure able to compute a valid bound on the primal optimal value detailed in the manuscript:

Federico Battista, Marianna De Santis "Dealing with inequality constraints in large scale 
                                       semidefinite relaxations for graph coloring and maximum 
                                       clique problems"

---------------------------------------------------------------------------------
The MAIN code for the Alternating Direction Augmented Lagrangian is
  
  * ADALplus_bounds.m 

N.B. the post-processing procedure calls the LP solver of Gurobi.
     You need to install Gurobi in order to run it
   
---------------------------------------------------------------------------------
INSTANCES:
---------------------------------------------------------------------------------
  The instances used in the numerical experience are available in three different 
  folders

