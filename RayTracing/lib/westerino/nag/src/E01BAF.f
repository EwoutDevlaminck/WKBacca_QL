cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG E01BAF
      SUBROUTINE E01BAF(M, X, Y, LAMDA, C, LCK, WRK, LWRK, IFAIL)
c     E01BAF determines a cubic spline interpolant (B-spline) to a given set of data.
c     routine is replaced by splint from *a practical guide to splines * by C. De Boor
c     this routine and its dependencies are obtained from netlib.org
c
c     parameters
c     input
c           m           number of data points .le. 4
c           x(m)        array of abcissae in strictly increasing order
c           y(m)        array of function values
c     output
c           lamba(lck)  array of knots
c           c(lck)      array of b-spline coefficients
c           lck         dimension of lamda and c in calling programme
c                       lck must be at least m+4
c           wrk(lwrk)   work space
c           lwrk        must be at least 7*m
c           ifail       = 0 for success
      INTEGER M, LCK, LWRK, IFAIL
      REAL X(M), Y(M), LAMDA(LCK), C(LCK), WRK(LWRK)
      
c     order of spline interpolant in splint
      k = 4
      iflag = 0

c     calculate the knots to be used (input to splint)
      if ( m .lt. 4 ) then
        ifail = 1
        stop 'E01BAF M .LT. 4'
      endif
      lamda(1) = x(1)
      lamda(2) = x(1)
      lamda(3) = x(1)
      lamda(4) = x(1)
      lamda(m+1) = x(m)
      lamda(m+2) = x(m)
      lamda(m+3) = x(m)
      lamda(m+4) = x(m)
      do 100 i = 5, m
        lamda(i) = x(i-2)
  100 continue

c     now we are ready to call splint
      call splint( x, y, lamda, m, k, wrk, c, iflag )

c     test for succes
      if( iflag .eq. 1 ) then
        ifail = 0
      else
        ifail = 1
        stop 'failure in E01BAF'
      endif

      return
      end
      
