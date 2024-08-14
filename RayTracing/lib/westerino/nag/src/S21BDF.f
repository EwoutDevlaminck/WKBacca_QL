cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c     replacement for NAG S21BDF
c     replaced by numerical recipes RJF
      FUNCTION S21BDF(X, Y, Z, P, IFAIL)
c     symmetrised elliptic integral of the third kind
      integer ifail
      real s21bdf,p,x,y,z,errtol,tiny,big,cl,c2,c3,c4,c5,c6,c7,c8
      PARAMETER(errtol=.002,tiny=2.5e-13,big=9.0e11,cl=3./14.,c2=1./3.,
     * c3=3./22.,c4=3./26. ,c5=.75*c3,c6=1.5*c4,c7=.5*c2,c8=c3+c3)
c     USES rc, rf (s21bbf)
c     Computes Carlson's elliptic integral of the third kind, RJ(x,y,z,p). 
c     x, y, and z must be nonnegative, and at most one can be zero. 
c     p must be nonzero. If p < 0, the Cauchy principal value is returned. 
c     TINY must be at least twice the cube root of the machine underflow limit, 
c     BIG at most one fifth the cube root of the machine overflow limit.
      REAL a,alamb,alpha,ave,b,beta,delp,delx,dely,delz,ea,eb,ec,
     *     ed,ee,fac,pt,rcx,rho,sqrtx,sqrty,sqrtz,sum,tau,xt,
     *     yt,zt,rc,rf
      if( min(x,y,z) .lt. 0.0 .or. min(x+y,x+z,y+z,abs(p)) .lt. TINY 
     * .or. max(x,y,z,abs(p)) .gt. BIG ) stop 'invalid arguments in rj'
      sum=0.
      fac=1.
      if(p .gt. 0.0)then
        xt=x
        yt=y
        zt=z
        pt=p
      else
        xt=min(x,y,z)
        zt=max(x,y,z)
        yt=x+y+z-xt-zt
        a=1. / (yt-p)
        b=a*(zt-yt)* (yt-xt)
        pt=yt+b
        rho=xt*zt/yt
        tau=p*pt/yt
        rcx=rc(rho,tau)
      endif
    1 continue
        sqrtx=sqrt(xt)
        sqrty=sqrt(yt)
        sqrtz=sqrt(zt)
        alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz
        alpha=(pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz)**2
        beta=pt*(pt+alamb)**2
        sum=sum+fac*rc(alpha,beta)
        fac=.25*fac
        xt=.25*(xt+alamb)
        yt=.25*(yt+alamb)
        zt=.25*(zt+alamb)
        pt=.25*(pt+alamb)
        ave=.2*(xt+yt+zt+pt+pt)
        delx=(ave-xt)/ave
        dely=(ave-yt)/ave
        delz=(ave-zt)/ave
        delp=(ave-pt)/ave
        if(max(abs(delx),abs(dely),abs(delz),abs(delp)) .gt. errtol) 
     .     goto 1
      ea=delx*(dely+delz)+dely*delz
      eb=delx*dely*delz
      ec=delp**2
      ed=ea-3.*ec
      ee=eb+2.*delp*(ea-ec)
      s21bdf=3.*sum+fac*(1.+ed*(-Cl+C5*ed-C6*ee)+eb*(C7+delp*
     * (-C8+delp*C4))+delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave))
      if (p.le.0.0) s21bdf=a*(b*s21bdf+3.*(rcx-s21bbf(xt,yt,zt)))
      ifail = 0.0
      return
      end

      FUNCTION rc(x,y)
      REAL rc,x,y,ERRTOL,TINY,SQRTNY,BIG,TNBG,COMP1,COMP2,THIRD,
     * Cl,C2,C3,C4
      PARAMETER (ERRTOL=.04,TINY=1.6e-38,SQRTNY=1.3e-19,BIG=3.E37,
     * TNBG=TINY*BIG,COMP1=2.236/SQRTNY,COMP2=TNBG*TNBG/25.,
     * THIRD=1./3.,Cl=.3,C2=1./7.,C3=.375,C4=9./22.)
c     Computes Carlson's degenerate elliptic integral, Rc{x,y). 
c     x must be nonnegative and y must be nonzero. 
c     If y < 0, the Cauchy principal value is returned. 
c     TINY must be at least 5 times the machine underflow limit, 
c     BIG at most one fifth the machine maximum overflow limit.
      REAL alamb,ave,s,w,xt,yt
      if(x.lt.0.0 .or. y.eq.0.0 .or. (x+abs(y)).lt.TINY .or. 
     *   (x+abs(y)).gt.BIG .or. (y.lt.-COMP1 .and. x.gt.0.0 .and.
     *   x.lt.COMP2)) stop 'invalid arguments in rc'
      if(y.gt.0.0)then
        xt=x
        yt=y
        w=1.0
      else
        xt=x-y
        yt=-y
        w=sqrt (x)/sqrt (xt)
      endif
    1 Continue
        alamb=2.*sqrt(xt)*sqrt(yt)+yt
        xt=.25*(xt+alamb)
        yt=.25*(yt+alamb)
        ave=THIRD*(xt+yt+yt)
        s=(yt-ave)/ave
        if(abs(s) .gt. ERRTOL) goto 1
      rc=w*(1.+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave)
      return
      end
