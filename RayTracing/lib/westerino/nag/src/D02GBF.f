cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG D02GBF
      SUBROUTINE D02GBF(A, B, N, TOL, FCNF, FCNG, C, D, GAM, MNP, X, Y, 
     1 NP, W, LW, IW, LIW, IFAIL)
      INTEGER N, MNP, NP, LW, IW(LIW), LIW, IFAIL
      REAL A, B, TOL, C(N,N), D(N,N), GAM(N), X(MNP), Y(N,MNP),
     1 W(LW)
      EXTERNAL FCNF, FCNG
c     D02GBF solves a general linear two-point boundary value problem 
c     for a system of ordinary differential equations, using a deferred 
c     correction technique.
      stop 'D02GBF not yet implemented'
      return
      end
