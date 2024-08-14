cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG D01BAF
      REAL FUNCTION D01BAF(D01XXX, A, B, N, FUN, IFAIL)
      INTEGER N, IFAIL
      REAL A, B, FUN
      EXTERNAL D01XXX, FUN
c     D01BAF computes an estimate of the definite integral of a function
c     of known analytical form, using a Gaussian quadrature formula with 
c     a specified number of abscissae. Formulae are provided for a 
c     finite interval (Gauss–Legendre), a semi-infinite interval (Gauss–
c     Laguerre, Gauss–Rational), and an infinite interval (Gauss–Hermite).
c     Parameters
c     D01XXX – SUBROUTINE, supplied by the Library. External Procedure
c     The name of the routine indicates the quadrature formula:
c     D01BAZ, for Gauss–Legendre weights and abscissae;
c     D01BAY, for Gauss–Rational weights and abscissae; not needed
c     D01BAX, for Gauss–Laguerre weights and abscissae; not needed
c     D01BAW, for Gauss–Hermite weights and abscissae. not needed
      d01baf = 0.0
      stop 'D01BAF not yet implemented'
      return
      end
