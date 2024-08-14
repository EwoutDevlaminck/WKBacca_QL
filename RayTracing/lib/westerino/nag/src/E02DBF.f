cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement of NAG E02DBF and E02DEF
      SUBROUTINE E02DBF(M,PX,PY,X,Y,FF,LAMDA,MU,POINT,NPOINT,C,NC,IFAIL)
      INTEGER M,PX,PY,NPOINT,POINT(NPOINT),NC,IFAIL
      REAL X(M),Y(M),FF(M),LAMDA(PX),MU(PY),C(NC)
      REAL WRK(1000)
      INTEGER IWRK(1000)
C     parameters for B2VAL
C     evaluate function value itself: IDX,Y = 0
      IDX = 0
      IDY = 0
C     CUBIC SPLINES HENCE KX, KY = 3+ 1
      KX = 4
      KY = 4
C
      NX = PX - KX
      NY = PY - KY
c     check that input is in proper domain
      do 100 i = 1, m
        if( x(i) .lt. lamda(4) ) x(i) = lamda(4)
        if( x(i) .gt. lamda(px-3) ) x(i) = lamda(px-3)
        if( y(i) .lt. mu(4) ) y(i) = mu(4)
        if( y(i) .gt. mu(py-3) ) y(i) = mu(py-3)
C       CALL B2VAL TO EVALUATE THE SPLINE INTERPOLANT AT THIS POINT
C       NOTE THAT THE STORAGE OF THE SPLINE COEFFICIENTS IN C (NAG)
C       IS THE MIRROR IMAGE OF THE ONE EXPECTED BY B2VAL
        FF(I) = B2VAL(Y(I),X(I),IDY,IDX,MU,LAMDA,NY,NX,
     *  KY,KX,C,WRK)
  100 continue
cdel      stop 'E02DBF not yet implemented'
cdel      CALL E02DEF(M,PX,PY,X,Y,LAMDA,MU,C,FF,WRK,IWRK,IFAIL)
C     when no error was encountered set IFAIL = 0
      IFAIL = 0
      RETURN
      END


cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG E02ZAF
      SUBROUTINE E02ZAF(PX, PY, LAMDA, MU, M, X, Y, POINT, NPOINT,
     1 ADRES, NADRES, IFAIL)
      INTEGER PX, PY, M, POINT(NPOINT), NPOINT, ADRES(NADRES),
     1 NADRES, IFAIL
      REAL LAMDA(PX), MU(PY), X(M), Y(M)
c     this routine is obsolete
      return
      end
