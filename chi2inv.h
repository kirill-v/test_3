// Implementation is copied from https://github.com/MRPT/mrpt
// https://github.com/MRPT/mrpt/blob/498f78eee2f78afdf450a506f03c536fcdd3e32c/libs/math/src/math.cpp#L43

#ifndef CHI2INV_H_
#define CHI2INV_H_

/** The "quantile" of the Chi-Square distribution, for dimension "dim" and
  * probability 0<P<1 (the inverse of chi2CDF)
  * An aproximation from the Wilson-Hilferty transformation is used.
  *  \note Equivalent to MATLAB chi2inv(), but note that this is just an
  * approximation, which becomes very poor for small values of "P".
  */
double chi2inv(double P, unsigned int dim = 1);

#endif
