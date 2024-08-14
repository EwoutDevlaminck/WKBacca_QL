cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG S21BCF
c     replaced by numerical recipes RD
      FUNCTION S21BCF(X, Y, Z, IFAIL)
c     symmetrised elliptic integral of the second kind
      integer ifail
      real s21bcf,x,y,z,errtol,tiny,big,cl,c2,c3,c4,c5,c6
      PARAMETER (errtol=.002,tiny=1.e-25,big=4.5E21,cl=3./14.,c2=1./6.,
     *           c3=9./22. ,c4=3./26.,c5=.25*c3,c6=1.5*c4)
c     Computes Carlson's elliptic integral of the second kind, RD(x, y, z). 
c     x and y must be nonnegative, and at most one can be zero. z must be positive. 
c     TINY must be at least twice the negative 2/3 power of the machine overflow limit. 
c     BIG must be at most 0.1 x ERRTOL times the negative 2/3 power of the machine underflow limit.
      real alamb,ave,delx,dely,delz,ea,eb,ec,ed,ee,fac,sqrtx,sqrty,
     *     sqrtz,sum,xt,yt,zt
      if( min(x,y) .lt. 0.0 .or. min(x+y,z).lt. TINY .or.
     *    max(x,y,z) .gt. BIG) stop 'invalid arguments in rd'
      xt=x
      yt=y
      zt=z
      sum=0.
      fac=1.
    1 continue
        sqrtx=sqrt(xt)
        sqrty=sqrt(yt)
        sqrtz=sqrt(zt)
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
        sum=sum+fac/(sqrtz*(zt+alamb))
        fac=.25*fac
        xt=.25*(xt+alamb)
        yt=.25*(yt+alamb)
        zt=.25*(zt+alamb)
        ave=.2*(xt+yt+3.*zt)
        delx=(ave-xt)/ave
        dely=(ave-yt)/ave
        delz=(ave-zt)/ave
        if( max(abs(delx),abs(dely),abs(delz)) .gt. errtol ) goto 1
      ea=delx*dely
      eb=delz*delz
      ec=ea-eb
      ed=ea-6.*eb
      ee=ed+ec+ec
      s21bcf=3.*sum+fac*(1.+ed*(-Cl+C5*ed-C6*delz*ee)
     * +delz*(C2*ee+delz*(-C3*ec+delz*C4*ea)))/(ave*sqrt(ave))
      ifail = 0.0
      return
      end
      
      
