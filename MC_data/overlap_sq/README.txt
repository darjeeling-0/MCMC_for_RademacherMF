This folder includes the element wise square of the M times M overlap between the average ground truth and Posterior sample obtained from MCMC. 
column 1 : list of SNRs 
column 2~M^2+1 : element wise square of the overlap matrix in flattened format, row-major order
The last row (with Inf SNR) is given by the element wise square of the self-overlap of the ground truth.
