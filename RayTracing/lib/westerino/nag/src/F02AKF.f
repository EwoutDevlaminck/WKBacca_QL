cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG F02AKF and F02GBF
      SUBROUTINE F02AKF( ar,N,ai,M,K,rr,ri,vr,I,vi,J,int,ifail )
      implicit complex (c)
      dimension ar(3,3), ai(3,3), rr(3), ri(3), vr(3,3), vi(3,3), int(3)
      complex amat(3,3), weig(3), veig(3,3)
      dimension cwork(1000), rwork(1000)

c     copy the input to the complex matrix
      ci1 = cmplx( 0.0, 1.0 )
      do 100 ii = 1, 3
        do 100 jj = 1, 3
          amat( ii, jj ) = ar( ii, jj) + ci1 * ai( ii, jj )
  100 continue
  
c     call the new nag routine (to be replaced)
      stop 'F02AKF not yet implemented'
cdel      call F02GBF( 'N',3,amat,3,weig,veig,3,rwork,cwork,1000,ifail )

c     now split the results
      do 200 ii = 1, 3
        rr(ii) = real( weig(ii) )
        ri(ii) = aimag( weig(ii) ) 
        do 200 jj = 1, 3
          vr(ii,jj) = real( veig(ii,jj) )
          vi(ii,jj) = aimag( veig(ii,jj) )
  200 continue

      return
      end
