# Thin Group Matrix Investigation
## By Gabriel Konar-Steenberg and Michaela Polley
Some computational work related to a math research project undertaken with Dr. Eric Egge at Carleton College. The problem simplifies to this:
 * Let `A = [[1,1,2],[0,1,1],[0,-3,-2]]` and `B = [[-2,0,-1],[-5,1,-1],[3,0,1]]`.
 * Let `H` be the set of all products of `A` and `B`: `{A, B, AB, BA, AAB, ...}`.
 * Are there infinite [left cosets](https://en.wikipedia.org/wiki/Coset) of `H` within the set of all 3x3 matrices with determinant 1?

Most of the code here relates to examining what `H` consists of. For instance, we have that `A^3` and `B^3` are the identity matrix, so multiple strings of `A` and `B` can evaluate to the same matrix (e.g., `AAAB = B`). We call the shortest string that evaluates to a particular matrix its "reduced" form, but we can computationally verify that these are non-unique also: there are multiple strings of the same length that evaluate to the same matrix even when there are no shorter strings they can be reduced to (e.g. `AABBABBAABAABB = BABABBABAABABA` and both of those are reduced). `reduction_investigation.py` is a more-or-less brute force analysis that evaluates all possible reduced strings up to a given length, counts how many of them are actually reduced, and logs them for further analysis. A naïve approach would suggest that there are 2^n strings to be analyzed for string length n, but by using various optimization techniques and tools like [Numba](https://numba.pydata.org/), we are able to complete this analysis for strings of length up to n=38 without specialized computing hardware in just a few hours.

The repository also contains code to analyze how elements of `H` behave modulo small integers, how the magnitude of elements of matrices in `H` changes as string length increases, and more.

This problem, first posed in Kontorovich et al. 2019 (who cite Long et al. 2011; see below) is related to the concept of thin groups and Zariski topology. Portions of this codebase also explore more general problems related to the one defined above.

References:
 * Kontorovich, A., Long, D. D., Lubotzky, A., & Reid, A. W. (2019). WHAT IS...a Thin Group? Notices of the American Mathematical Society, 66(06), 1. https://doi.org/10.1090/noti1900
 * Long, D. D., Reid, A. W., & Thistlethwaite, M. (2011). Zariski dense surface subgroups in SL(3,Z). Geometry & Topology, 15(1), 1–9. https://doi.org/10.2140/gt.2011.15.1