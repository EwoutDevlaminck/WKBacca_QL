cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG E02BCF
      SUBROUTINE E02BCF(NCAP7, LAMDA, C, X, LEFT, S, IFAIL)
c     E02BCF evaluates a cubic spline and its first three derivatives from its B-spline representation.
c     replaced by successive calls to bvalue 
c     this routine and its dependencies are obtained from netlib.org
c     parameters
c     input
c           ncap7        number of intervals of spline + 7 must be .le. 8
c           lamda(ncap7) knots at which b-spline coefficients are given
c           c(ncap7)     values of b-spline coefficients
c           x            point at which spline and derivatives must be evaluated
c           left         .eq.1 for left hand values (not relevant in this implementation)
c     output
c           s(4)         function value and first three derivates
      INTEGER NCAP7, LEFT, IFAIL
      REAL LAMDA(NCAP7), C(NCAP7), X, S(4)
c     number of spline coefficents
      n = ncap7-4
c     order of spline
      k = 4
      do 10000 ideriv = 1, 4
c       difinition: real function bvalue ( t, bcoef, n, k, x, jderiv )
        jderiv = ideriv - 1
        s( ideriv ) = bvalue( lamda, c, n, k, x, jderiv )
10000 continue
      ifail = 0
      return
      end
