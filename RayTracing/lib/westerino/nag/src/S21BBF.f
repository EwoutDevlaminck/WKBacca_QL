cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG S21BBF
c     replaced by numerical recipes RF
      FUNCTION S21BBF(X, Y, Z, IFAIL)
c     symmetrised elliptic integral of the first kind
c
      integer ifail
      real s21bbf,x,y,z,errtol,tiny,big,third,c1,c2,c3,c4
      PARAMETER (errtol=.0025,tiny=1.e-37,big=1.e+37,third=1./3.,
     * cl=1./24.,c2=0.1,c3=3./44.,c4=1./14.)
c     Computes Carlson's elliptic integral of the first kind, RF(X, y, z). 
c     x, y, and z must be nonnegative, and at most one can be zero. 
c     TINY must be at least 5 times the machine underflow limit
c     BIG at most one fifth the machine overflow limit.
      real alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt
      if( min(x,y,z) .lt. 0.0 .or. min(x+y,x+z,y+z) .lt. tiny .or.
     +    max(x,y,z) .gt. big ) stop 'invalid arguments in s21bbf'
      xt=x
      yt=y
      zt=z
    1 continue
        sqrtx=sqrt(xt)
        sqrty=sqrt(yt)
        sqrtz=sqrt(zt)
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
        xt=.25*(xt+alamb)
        yt=.25*(yt+alamb)
        zt=.25*(zt+alamb)
        ave=third*(xt+yt+zt)
        delx=(ave-xt)/ave
        dely=(ave-yt)/ave
        delz=(ave-zt)/ave
        if( max(abs(delx),abs(dely),abs(delz)) .gt. errtol) goto 1
      e2=delx*dely-delz**2
      e3=delx*dely*delz
      s21bbf=(1.+(Cl*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave)
      ifail = 0.0
      return
      end
      
      
