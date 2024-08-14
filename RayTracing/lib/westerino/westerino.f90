!     ABSORPTION ROUTINE
!
!     IMPLICIT NONE
!
!      REAL*8 THETA,NR,NI,ALFA,BETA,VTE,MODE,PI,ME,NI,TE,
!     +       NIA,VF
!      INTEGER I
!
!C     PI AND THE ELECTRON MASS (MKS)
!      PI = 4.*ATAN(1.)
!      ME = 9.1E-31
!
!c     LOEPJE
!      DO 1 I = 1, 200
!
!c     THETA IS THE ANGEL BETWEEN THE WAVE VECTOR AND THE
!c     THE MAGNETIC FIELD. FOR PERPENDICULAR PROPAGATION
!      THETA = PI / 2.
!
!C     ALFA IS RELATED TO THE DENSITY IT IS THE PLASMA
!C     FREQUENCY SQUARED OVER THE WAVE FREQUENCY SQUARED.
!      ALFA = 0.3
!
!C     BETA IS RELATED TO THE MAGNETIC FIELD STRENGTH, IT
!C     IS THE CYCLOTRON FREQUENCY SQUARED OVER THE WAVE
!C     FREQUENCY SQUARED.
!      BETA = 1. + I / 1E4
!      (first harmonic)
!
!C     NR IS THE REAL PART OF THE REFRACTIVE INDEX (INPUT)
!C     IS TO BE CALCULATED FROM THE DISPERSION RELATION.
!      NR = SQRT(1.-ALFA)
!
!C     MODE SELECTS THE MODE. MODE = -1 SELECTS THE X-MODE
!C     MODE = +1 SELECTS THE O-MODE.
!      MODE = +1
!
!C     VTE IS THE THERMAL VELOCITY OF THE ELECTRONS  NORMALIZED
!C     TO THE SPEED OF LIGHT AND THEREFORE
!C     RELATED TO THE ELECTRON TEMPERATURE TE. BELOW TE IS GIVEN
!C     IN KEV, VTE IS THEN CALCULATED
!      TE = 1
!      VTE = SQRT(3.2E-16*TE/ME)/3.0e8
!
!C     CALL THE ROUTINE
!      CALL DAMPBQ(THETA,NR,NI,ALFA,BETA,VTE,MODE)
!
!C     CALCULATE ALFA FOR THE PERPENDICULAR PROPAGATING O-MODE
!C     FROM ANALYTIC FORMULA (approx)
!      VF  = 511.*(SQRT(BETA)-1)/ TE
!      NIA = 2*SQRT(PI)*ALFA*SQRT(1-ALFA)*VF**2.5*EXP(-VF)/15.
!
!C     THE OUTPUT IS THE ABSORPTION COEFFICIENT STORED IN NI
!      WRITE(*,*)NI,NIA
!
!     This is the absolute value of the imagnary part of the
!     refractive index. To obtain the imaginary part of the
!     wave vector one has to multiply with omega/c. This
!     imaginary wave vector lies along the real part. The
!     power is given by
!
!       P = 1. - exp [ - 2 \int  k_i \cdot {\rm d}{\bf s} ]
!
!     where k_i = ni * \omega / c. The {\rm d} {\bf s} is in
!     the direction of the wave propagation and therefore in
!     the direction of the group velocity. (the direction of
!     wave propagation is of course directly calculated in
!     the code) The wave vector is not necessarily in this
!     direction. Therefore, an additional cos(ph) appears
!     where ph is the angle between the wave vector and the
!     direction of propagation. The differential equation is
!     then
!
!       dP / d s = 2 NI * \omega / c * cos(ph) * (1 - P )
!
!     This can be integrated with the ODE routine.
!
!
!1     CONTINUE
!
!      END
!
!     *********************************************************************
      subroutine DAMPBQ(PTHETA,PNR,PNI,PALFA,PBETA,PVTE,PMODE,ICALLED)
!     *********************************************************************
!       subroutine TO CALCULATE DAMPING FOR WAVES NEAR FUNDAMENTAL
!       CYCLOTRON FREQ--E. WESTERHOF (WEAKLY RELATIVISTIC APPR.)
!
!   THIS ROUTINE USES THE BIQUADRATIC EQUATION TO SOLVE FOR N PERP.
!   THE SECOND HARMONIC ABSORPTION IS INCLUDED.
!
! NR      = THE REAL PART OF THE WAVE NUMBER
! NI      = THE IMAGINARY PART OF THE WAVE NUMBER
!              PLASMA EFFECTS
! THETA   = THE ANGLE BETWEEN THE WAVE VECTOR AND THE MAGNETIC FIELD
! VTE     = THE ELECTRON THERMAL VELOCITY
!
!
!               2   2
!       ALFA = W  /W
!               PE
!
!
!               2   2
!       BETA = W  /W
!               CE
!
!
!***********************

      IMPLICIT   COMPLEX (C)

      COMPLEX    PCEX , PCEY
      COMPLEX    CEX,CEZ

      LOGICAL L1, L2

      DIMENSION  CE( 3 , 3 )

      COMMON / II / CI1

      COMMON / COMMBQ / NQUAD, NBEGIN, NSIGN

      COMMON / PP / PI , ROOTPI , TWOPI

      DATA NBEGIN, NSIGN, NQUAD / 0, 1, 0 /
      DIMENSION PLASM1(51),PLASM2(10)

      NBEGIN = ICALLED
!!!      NSIGN = 1
!!!      NQUAD = 0

      DO 1 J = 1, 51
        PLASM1(J) = 0.
1     CONTINUE
      DO 2 J = 1, 10
        PLASM2(J) = 0.
2     CONTINUE

      PI     = 4.0 * ATAN( 1.0 )
      ROOTPI = SQRT( PI )
      TWOPI  = 2.0 * PI
      CI1    = ( 0.0 , 1.0 )

      ZWFR = 1.0
      ZFCYCL = SQRT( PBETA )
      ZPLFR = SQRT( PALFA )
      ZMU = 2.0 / PVTE**2
      ZRIPAR = PNR * COS( PTHETA )
      IF( ABS(ZRIPAR) .LT. 1.0E-2 ) ZRIPAR = 1.0E-2
      ZRIPER = PNR * SIN( PTHETA )

      PLASM1(  1 ) = ZPLFR
      PLASM1(  2 ) = ZFCYCL
      PLASM1(  3 ) = ZMU
      PLASM1(  4 ) = 0.0
      PLASM1(  5 ) = 0.0
      PLASM1(  6 ) = 0.0

      PLASM2( 2 ) = ZWFR
      PLASM2( 3 ) = ZRIPAR
      PLASM2( 4 ) = ZRIPER

!
!-----------------------------------------------------------------------
!L               2.     CALCULATION OF THE ABSORPTIONCOEFFICIENT
!
!     provisionally put maxit = 10 ; accur = 0.01
      ACCUR = 0.01
      MAXIT = 10
      OLDNP = ZRIPER
      DO 1000  JIT = 1, MAXIT
!       ZRIPER = 0  LEADS TO UNDEFINEDS IN THE CALCULATION!
        ZRIPER  = MAX( OLDNP, ACCUR*0.5 )
        PLASM2( 4 ) = ZRIPER
!   CALCULATE THE DIELECTRIC TENSOR
        CALL DIELTE( PLASM1 , PLASM2 , CE )

!   SOLUTION OF THE BIQUADRATIC EQUATION
!   SOME QUANTITIES TO BE USED
        EZZ0   = 1.0 - PALFA
        CLXX   = CE( 1, 1 ) - ZRIPAR**2
        CLYY   = CE( 2, 2 ) - ZRIPAR**2
        CEXY   = CE( 1, 2 )
        CLXZ   = CE( 1, 3 ) / ZRIPER + ZRIPAR
        CLYZ   = CE( 2, 3 ) / ZRIPER
        CLZZ   = 1.0 + ( EZZ0 - CE( 3, 3 ) ) / ZRIPER**2

        IF( ABS( ZRIPAR ) .GT. 1.0E-3 ) THEN

!   SOLVE THE BIQUADRATIC EQUATION
          CA     = CLXZ**2 + CLXX * CLZZ
          CB     = CLYY * CLXZ**2 + CEXY**2 * CLZZ
          CB     = CB  -  2.0 * CEXY * CLYZ * CLXZ
          CB     = CB  -  CLXX * ( CLYZ**2 - EZZ0 - CLYY*CLZZ )
          CC     = EZZ0 * ( CEXY**2 + CLXX * CLYY )

!   THE DETERMINANT
          CD     = CB**2 - 4.0 * CA * CC
          CRD    = SQRT( CD )

!   CHECK FOR CROSSING FROM QUADRANT 2 TO 3 OR VICE VERSA
          ZRD    = REAL( CD )
          ZID    = AIMAG( CD )
          IF( ( ZRD .GE. 0.0 ) .AND. ( ZID .GE. 0.0 ) ) NNEW = 1
          IF( ( ZRD .LT. 0.0 ) .AND. ( ZID .GE. 0.0 ) ) NNEW = 2
          IF( ( ZRD .LE. 0.0 ) .AND. ( ZID .LT. 0.0 ) ) NNEW = 3
          IF( ( ZRD .GT. 0.0 ) .AND. ( ZID .LT. 0.0 ) ) NNEW = 4

          L1     = ( NQUAD .EQ. 2 ) .AND. ( NNEW .EQ. 3 )
          L2     = ( NQUAD .EQ. 3 ) .AND. ( NNEW .EQ. 2 )
          IF( L1 .OR. L2 ) NSIGN = -NSIGN
          ZGN = NSIGN
          NQUAD  = NNEW

!   THE CORRECT ROOT IS
         CR21 = ( CB + ZGN * CRD ) / ( 2.0 * CA )
!   THE OTHER ROOT IS
         CR22 = ( CB - ZGN * CRD ) / ( 2.0 * CA )

!   WHEN DAMPBQ IS CALLED FOR THE FIRST TIME ALONG A RAY
!   THE CORRECT SIGN OF NSIGN IS TO BE DETERMINED
      IF( NBEGIN .EQ. 0 ) THEN

!   CHOOSE THE ROOT WHICH REAL PART IS CLOSEST TO THE COLD SOLUTION
        ZRR1 = REAL(SQRT(CR21))
        ZRR2 = REAL(SQRT(CR22))
        IF( ABS(ZRIPER-ZRR1) .LE. ABS(ZRIPER-ZRR2) ) THEN
          CR2 = CR21
        ELSE
          CR2 = CR22
          NSIGN = - NSIGN
        ENDIF
        NBEGIN = 1

      ELSE

      CR2 = CR21

      ENDIF

        ELSE
!       FOR PERPENDICULAR PROPAGATION
          IF( PMODE .LT. 0.0 ) THEN
            CR2   = CLYY + CEXY**2 / CLXX
          ELSE
            CR2   = EZZ0 / CLZZ
          ENDIF
        ENDIF

        CR  = SQRT( CR2 )
        ZRIPER = REAL( CR )
        PNI = AIMAG( CR ) * SIN( PTHETA )

        IF( ABS(ZRIPER-OLDNP) .LT. ACCUR ) THEN
          GOTO 2000
        ENDIF

        OLDNP = ZRIPER

 1000 CONTINUE
 2000 CONTINUE

      CR  = SQRT( CR2 )
      ZRIPER = REAL( CR )
      PNI = AIMAG( CR ) * SIN( PTHETA )




!-------------------------------------------------------------------------------------
!             POLARIZATION by N. BERTELLI (04/02/2010)
!
!-------------------------------------------------------------------------------------
!   COMPONENT OF DIELETRIC TENSOR (CORRESPONDING TO EQ.(5) OF Physics Fluids 21 (1978) 
!                                          645 or from Eq.(57) OF TORAY MANUAL (1989))
        CLAMXX = CLXX
        CEPS12 = CEXY
        CLAMXZ = CLXZ*CR
        CLAMYY = CLYY - CR**2
        CLAMYZ = CLYZ*CR
        CLAMZZ = EZZ0 - CLZZ*CR**2

!----------------------------------------------------------------------------------------
!CL    O-MODE POLARIZATION VECTOR (NORMALIZED TO EZ) (SEE EQ.(54) OF TORAY MANUAL (1989))


      PCEX=(CEPS12*CLAMYZ-CLAMXZ*CLAMYY)/&
     (CLAMXX*CLAMYY+CEPS12**2)

      PCEY=(CLAMXX*CLAMYZ+CEPS12*CLAMXZ)/&
     (-1.*CEPS12**2-CLAMXX*CLAMYY)

!     VECTOR NORMALIZATION      

      OMOD = SQRT(ABS(PCEX)**2+ABS(PCEY)**2+1.0)
      PCEXNORMO = PCEX/OMOD
      PCEYNORMO = PCEY/OMOD
      PCEZNORMO = 1.0/OMOD 
 
!----------------------------------------------------------------------------------------
!CL    X-MODE POLARIZATION VECTOR (NORMALIZED TO EY) (SEE EQ.(54) OF TORAY MANUAL (1989))


      CEX = - ( CLAMYZ*CLAMXZ / CLAMZZ + CEPS12 ) /&
     ( CLAMXX - CLAMXZ**2 / CLAMZZ )

      CEZ = - ( CLAMXZ * CEX - CLAMYZ ) / CLAMZZ

!     VECTOR NORMALIZATION      

      XMOD = SQRT(ABS(CEX)**2+1.0+ABS(CEZ)**2)
      PCEXNORMX = CEX/XMOD
      PCEYNORMX = 1.0/XMOD
      PCEZNORMX = CEZ/XMOD    


!     END POLARIZATION BY BERTELLI  
!----------------------------------------------------------------------------------------








!   IT IS POSSIBLE THAT THE WARM DISPERSION FINDS A CUT-OFF
!   WHERE THE COLD ONE DID NOT YET]    IF SO, PUT NI = 0
      IF( ZRIPER .LT. 1.0E-3 ) PNI = 0.0

      return
      end subroutine DAMPBQ



!     *********************************************************************
      subroutine DIELTE( PLASM1 , PLASM2 , CE )
!     *********************************************************************
!
!   THIS ROUTINE EVALUATES THE DIELECTRICTENSOR
!   FOR THE COMPOSITE PLASMA
!
      IMPLICIT   COMPLEX (C)
      COMMON/ COMMU  / ZMU,ZC,JLOSS
      common/comexp/janex       !emp
      common/fwrel/ilrela,imax  !emp
      DIMENSION  PLASM1( 51 ) , PLASM2( 10 ) , CE( 3 , 3 )
      DIMENSION  PLASMA( 51 ) , CZ( 3, 3 ) , CP( 3 , 3 )
      LOGICAL NLRELA
      logical ilrela  !emp
      integer imax    !emp
!
!-----------------------------------------------------------------------
!L               1.     PROLOGUE
!
!     SET PARAMETERS FOR RELATIVISTIC CALCULATIONS (PROVISIONAL)
      NLRELA = ilrela
!      if (janexp.eq.1) NLRELA = .false.
!      AUG
!      NLRELA = .false.
      NMIN = -3
      NMAX = imax
!      NMAX = +2 not always stable!
      NTERM = 5
      NPM  = 16
      NGM  = 16
      NAM  = 16

      DO 101 J1 = 1 , 3
        DO 101 J2 = 1 , 3
          IF ( J1 .EQ. J2 ) THEN
            CE( J1 , J2 ) = 1.0
          ELSE
            CE( J1 , J2 ) = 0.0
          ENDIF
  101 CONTINUE
!
!-----------------------------------------------------------------------
!L                2.     SUMMATION OF ALL CONTRIBUTIONS
      DO 1000 J = 1 , 4
        I10 = 10 * ( J - 1 )
!
!   PARAMETERS FOR THIS COMPONENT
        ZPLFR   = PLASM1( I10 + 1 )
        ZFCYCL  = PLASM1( I10 + 2 )
        ZMU     = PLASM1( I10 + 3 )
        ZBETA   = PLASM1( I10 + 4 )
        KNPAR   = INT( PLASM1( I10 + 5 ) )
        KNPER   = INT( PLASM1( I10 + 6 ) )
        ZWFR    = PLASM2( 2 )
        ZRIPAR  = PLASM2( 3 )
        ZRIPER  = PLASM2( 4 )
        IF ( ZPLFR .LT. 1.0E-6 ) GOTO 1000
!
!   TRANSFORM PARAMETERS TO CO-MOVING FRAME
!   IF BETA =/ 0 IS SPECIFIED.
        IF ( ABS( ZBETA ) .GT. 1.0E-6 ) THEN
          ZGAMMA = 1.0 / SQRT( 1.0 - ZBETA**2 )
          ZWFR = ZWFR * ZGAMMA * ( 1.0 - ZBETA * ZRIPAR )
          ZNPARP = ( ZRIPAR - ZBETA ) / ( 1.0 - ZBETA * ZRIPAR )
          ZNPERP = ZRIPER / ZGAMMA / ( 1.0 - ZBETA * ZRIPAR )
          ZRIPAR = ZNPARP
          IF ( ABS( ZRIPAR ) .LT. 1.0E-3 ) ZRIPAR = 1.0E-3
          ZRIPER = ZNPERP
        ENDIF
!
!   FILL ARRAY FOR EPSILON
         IF( .NOT. NLRELA ) THEN
           PLASMA( 1 ) =  ZPLFR
           PLASMA( 2 ) =  ZWFR
           PLASMA( 3 ) =  ZFCYCL
           PLASMA( 4 ) =  ZMU
           PLASMA( 5 ) =  ZRIPAR
           PLASMA( 6 ) =  ZRIPER
           CALL EPSILON( PLASMA , KNPAR , KNPER , CP )
         ELSE
!
!   INPUTS FOR RELATIVISTIC ROUTINE
           ZX = (ZPLFR/ZWFR)**2
           ZY = ZFCYCL/ZWFR
!          NORMALIZATION OF THE DISTRIBUTION function
!          NOTE THAT THE LEADING TERM IN FK IS NORMALIZED TO 1
!          ANTI-LOSS CONE NOT IMPLEMENTED
           PI    = 4.0 * ATAN( 1.0 )
           JLOSS = KNPER/2
           ZK2 = FK( ZMU, (JLOSS+2), 5 )
           ZK2 = ZK2 * SQRT( 0.5 * PI / ZMU )
           ZC  = ZMU**(JLOSS+1) / (4.0*PI*2.0**JLOSS*FAC(JLOSS)*ZK2)
           CALL FREPSLN(ZX,ZY,ZRIPAR,ZRIPER,ZMU,NMIN,NMAX,NTERM,NPM,NGM, &
                          NAM,CP )
         ENDIF
!
!   TRANSFORM BACK TO LAB.-FRAME
!   IF BETA =/ WAS SPECIFIED.
        IF ( ABS( ZBETA ) .GT. 1.0E-6 ) THEN
          ZFAC1 = 1.0 / ( ZGAMMA * ( 1.0 + ZBETA * ZNPARP ) )**2
          ZFAC2 = ZGAMMA * ZBETA * ZNPERP
          ZFAC3 = ZGAMMA * ( 1.0 + ZBETA * ZNPARP )
          CZ( 1 , 1 ) = ZFAC1 * CP( 1 , 1 )
          CZ( 2 , 2 ) = ZFAC1 * CP( 2 , 2 )
          CZ( 1 , 2 ) = ZFAC1 * CP( 1 , 2 )
          CZ( 2 , 1 ) = - CZ( 1 , 2 )
          CZ( 1 , 3 ) = ZFAC1 * ( ZFAC2*CP( 1, 1 ) + ZFAC3*CP( 1, 3 ) )
          CZ( 3 , 1 ) = CZ( 1 , 3 )
          CZ( 2 , 3 ) =-ZFAC1 * ( ZFAC2*CP( 1, 2 ) - ZFAC3*CP( 2, 3 ) )
          CZ( 3 , 2 ) = - CZ( 2 , 3 )
          CZZ = ZFAC1 * ZFAC3 * ( ZFAC2*CP( 1, 3 ) + ZFAC3*CP( 3, 3 ) )
          CZ( 3 , 3 ) = ZFAC2 * CZ( 1 , 3 ) + CZZ
        ELSE
          DO 201 J3 = 1 , 3
            DO 201 J4 = 1 , 3
              CZ( J3 , J4 ) = CP( J3 , J4 )
  201     CONTINUE
        ENDIF
!
        DO 202 J3 = 1 , 3
          DO 202 J4 = 1 , 3
  202       CE( J3 , J4 ) = CE( J3 , J4 ) + CZ( J3 , J4 )
 1000 CONTINUE
      PLASM1( 41 ) = AIMAG( CE( 1 , 1 ) )
      PLASM1( 42 ) = AIMAG( CE( 1 , 3 ) )
      PLASM1( 43 ) = AIMAG( CE( 3 , 3 ) )
      return
      end subroutine DIELTE



!     *********************************************************************
      subroutine EPSILON( PLASMA , JNPAR , JNPER , CE )
!     *********************************************************************
!
! THIS subroutine CALCULATES THE WEAKLY RELATIVISTIC APPR.
! OF THE DIELECTRICTENSOR - THE UNITY TENSOR OF ((ANTI-)
! LOSSCONE) MAXWELLIAN DISTRIBUTION IN THE CO-MOVING FRAME.
! VERSION 3  07/APR/1986   E. WESTERHOF (FOM)
!
!  LAST UPDATE TO INCLUDE SECOND HARMONIC CONTRIBUTION
!  TO THE DIELECTRIC TENSOR.
!
!
      IMPLICIT COMPLEX (C)
      COMPLEX  ZETA
      COMMON / II / CI1
      COMMON / PP / PI , ROOTPI , TWOPI
      COMMON / NN / NSUM
      COMMON / HH / REALH
      DIMENSION PLASMA( 51 ) , CE( 3 , 3 )
!**
      DIMENSION CF( 50 , 0 : 50 , -1 : 2 )
        DO 1 J1 = 1 , 6 + JNPER/2 + JNPAR
        DO 1 J2 = 0 , 2 + JNPAR
        DO 1 J3 = -1 , 2
   1    CF( J1 , J2 , J3 ) = ( 0.0 , 0.0 )
!**
!
!   THE  XX, XY, XZ AND ZZ ELEMENTS OF THE DIELECTRIC TENSOR
!   FOR A (ANTI-) LOSSCONE MODEL DISTRIBUTION function ARE
!   CALCULATED IN THE W.R. APPROXIMATION AS A GENERALIZATION
!   OF THE WORK OF KRIVENSKI & OREFICE '83 (J. OF PL. P. 30, 125)
!
!-----------------------------------------------------------------------
!L               1.     PROLOGUE
!
!   CONSTANTS AND PLASMA AND WAVE PARAMETERS
!
      NSUM  = 12
      REALH = 0.5
      ZPLFR   = PLASMA( 1 )
      ZWFR    = PLASMA( 2 )
      ZFCYCL  = PLASMA( 3 )
      ZMU     = PLASMA( 4 )
      ZRIPAR  = PLASMA( 5 )
      ZRIPER  = PLASMA( 6 )
!
!-----------------------------------------------------------------------
!L               2.     CALCULATION OF THE F-functionS
!
!**
      DO 2000  JS = -1 , 2
!**
!
!   ARGUMENTS OF THE F-functionS FOR JS
      ZALFA = 0.5 * ZRIPAR**2 - 1.0 + REAL(JS) * ZFCYCL / ZWFR
      ZPSI = ZRIPAR * SQRT( 0.5 * ZMU )
      ZPS2 = ZPSI**2
      CPSI = CMPLX( ZPSI , 0.0 )
      IF (ZALFA .GE. 0.0) THEN
!
!   CALCULATION OF F1/2 AND F3/2 USING EQ.29 AND 30 OF K&O
          ZPHI = SQRT( ZMU * ZALFA )
          CPHI = CMPLX( ZPHI , 0.0 )
          CZ1 = ZETA( CPSI - CPHI )
          CZ2 = ZETA( - CPSI - CPHI )
          CZP = 0.5 * ( CZ1 + CZ2 )
          CZM = 0.5 * ( CZ1 - CZ2 )
          CF( 1 , 0 , JS ) = - CZP
          CF( 2 , 0 , JS ) = - CZM / CPSI
!
!   THE  OTHER F-functionS, AS FAR AS WE NEED THEM,
!   ARE CALCULATED FROM THE RECURRENCE RELATIONS.
          CF( 3 , 0 , JS ) = ( 1.0 + CPHI * CF( 1 , 0 , JS ) - &
          0.5 * CF( 2 , 0 , JS ) ) / CPSI**2
!
!   FOR PSI AND PHI BOTH SMALL (I.E. FOR
!   PERPENDICULAR PROPAGATION AND JS=+1)
!   THE RECURRENCE RELATION CREATE LARGE
!   NUMERICAL ERRORS FOR Q > 5/2.
          IF ( ( ZPS2 .LE. 0.1 ) .AND. ( JS .GE. 1 ) ) GOTO 25
!**
!DEL      IF ( (ZPS2.LE.0.1) .AND. (ABS(CPHI)**2.LE.0.1) ) GOTO 25
!**
!
          DO  21 JQ = 4 , ( 6 + JNPER/2 + JNPAR )
          CF( JQ , 0 , JS ) = ( 1.0 + CPHI**2 * CF( JQ-2 , 0 , JS ) - &
          ( REAL(JQ) - 2.5 ) * CF( JQ-1 , 0 , JS ) ) / CPSI**2
   21     CONTINUE
        ELSE
!
!   FOR NEGATIVE ALFAS F1/2 AND F3/2 ARE REAL AND GIVEN
!   BY EQ.35 OF K&O.
!   BUT FOR VERY LARGE PHI (I.E. FOR JS = -1 OR 0) WE
!   USE AN APPROXIMATE EXPRESSION TO FIRST ORDER (1/MU).
          IF ( JS .LE. 0 ) GOTO 23
          ZPHW = SQRT( - ZMU * ZALFA )
          IF ( ZPHW .GT. 6.0 ) GOTO 23
          CARG = CMPLX( ZPSI , ZPHW )
          CZ = ZETA( CARG )
          ZF12 = AIMAG( CZ )
          ZF32 = - REAL( CZ ) / ZPSI
          CF( 1 , 0 , JS ) = CMPLX( ZF12 , 0.0 )
          CF( 2 , 0 , JS ) = CMPLX( ZF32 , 0.0 )
!
!   THE OTHER F-functionS ARE CALCULATED BY THE RECURRENCE
!   RELATION, BUT WITH THE PROPER SUBSTITUTION OF
!   -I*PHIWIGGLE FOR PHI.
          CF( 3 , 0 , JS ) = ( 1.0 - ZPHW * CF( 1 , 0 , JS ) - &
          0.5 * CF( 2 , 0 , JS ) ) / CPSI**2
!
!   FOR PSI AND PHI BOTH SMALL (I.E. FOR
!   PERPENDICULAR PROPAGATION AND JS=+1)
!   THE RECURRENCE RELATION CREATE LARGE
!   NUMERICAL ERRORS FOR Q > 5/2.
          IF ( ( ZPS2 .LE. 0.1 ) .AND. ( JS .GE. 1 ) ) GOTO 25
!**
!DEL      IF ( (ZPS2.LE.0.1) .AND. (ABS(ZPHW)**2.LE.0.1) ) GOTO 25
!**
!
          DO  22 JQ = 4 , ( 6 + JNPER/2 + JNPAR )
          CF( JQ , 0 , JS ) = ( 1.0 - ZPHW**2 * CF( JQ-2 , 0 , JS ) - &
          ( REAL(JQ) - 2.5 ) * CF( JQ-1 , 0 , JS ) ) / CPSI**2
   22     CONTINUE
          GOTO 241
!
   23     CONTINUE
            DO 24 JQ = 1 , ( 6 + JNPER/2 + JNPAR )
            ZZ = ZMU * ( 0.5 * ZRIPAR**2 - ZALFA )
   24       CF( JQ , 0 , JS ) = CMPLX( ( 1.0 / ZZ ) , 0.0 )
  241     CONTINUE
      ENDIF
      GOTO 29
!
   25 CONTINUE
          CZ = CMPLX( ZMU * ( 0.5 * ZRIPAR**2 - ZALFA ) , 0.0 )
          DO 255 JQ = 4 , ( 6 + JNPER/2 + JNPAR )
          CSUM = ( 0.0 , 0.0 )
          DO 251 JNU = 0 , ( JQ - 2 )
  251     CSUM = CSUM + (-CZ)**JNU * GAMMAH( JQ-JNU-1 ) / GAMMAH( JQ )
          CFA1 = ROOTPI * ( - CZ )**(JQ-2) / GAMMAH( JQ )
          CFA2 = CI1 * SQRT( CZ ) * ZETA( CI1 * SQRT( CZ ) )
          CF( JQ , 0 , JS ) = CSUM + CFA1 * CFA2
  255     CONTINUE
!
   29 CONTINUE
!
!   NOW FILL CF( JQ , JB , JS ) FOR JB > 0.
      DO 201 JQ = 1 , (5 + JNPER/2 + JNPAR)
      CF( JQ, 1, JS ) = ZRIPAR * ( CF( JQ, 0, JS ) - CF( JQ+1, 0, JS ) )
  201 CONTINUE
      DO 250 JB = 2 , (2 + JNPAR)
      DO 250 JQ = 1 , (6 + JNPER/2 + JNPAR - JB)
      CF( JQ,JB,JS ) = ZRIPAR * (CF( JQ,JB-1,JS ) - CF( JQ+1,JB-1,JS ))+ &
       (JB - 1) * CF( JQ+1, JB-2, JS ) / ZMU
  250 CONTINUE
 2000 CONTINUE
!
!-----------------------------------------------------------------------
!L               3.     THE DIELECTRIC TENSOR
!
      NHA3 = ( JNPER + 4 ) / 2
      NHA5 = ( JNPER + 6 ) / 2
      NB0 = JNPAR
      NBM = NB0 - 1
      NB1 = NB0 + 1
      NB2 = NB0 + 2
      ZA = REAL( JNPER )
      ZA2 = ZA / 2.0
      ZB = REAL( JNPAR )
      ZB2 = ZB / 2.0
      ZRTM = 2.0 / ZMU
      ZGB = GAMMAH( 1 + JNPAR/2 )
      ZGA = ZGAMMA( 2 + JNPER/2 )
      ZC = ( ZA + 2 ) / ( PI * 2.0 * ZRTM**(ZA2+ZB2+1.5) * ZGB * ZGA )
      ZKO1 = (ZPLFR / ZWFR)**2 * ZMU * (TWOPI / ZMU)**1.5 * ZC
      ZKO2 = 0.5 * ZRIPER * ZWFR / ZFCYCL
!
      ZG1 = ZGAMMA( 1 + JNPER/2 )
      ZG2 = ZGAMMA( 2 + JNPER/2 )
!
      C1 = ZG1 * ( CF( NHA3 , NB0 , +1 ) + CF( NHA3 , NB0 , -1 ) )
      C2 = ZG1 * ( CF( NHA3 , NB1 , +1 ) + CF( NHA3 , NB1 , -1 ) )
      IF ( NB0 .GT. 0 ) THEN
      C3 = ZG2 * ( CF( NHA5 , NBM , +1 ) + CF( NHA5 , NBM , -1 ) )
      ELSE
      C3 = ( 0.0 , 0.0 )
      ENDIF
      C4 = ZG2 * ( CF( NHA5 , NB0 , +1 ) + CF( NHA5 , NB0 , -1 ) )
      CE(1,1) =  0.25 * ZKO1 * ( ZA * ZRTM**ZA2 * ( C1 - ZRIPAR * C2 ) &
       + ZRTM**(1.0+ZA2) * ( ZB * ZRIPAR * C3 - ZMU * C4 ) )
!
      C1 = ZG1 * ( CF( NHA3 , NB0 , +1 ) - CF( NHA3 , NB0 , -1 ) )
      C2 = ZG1 * ( CF( NHA3 , NB1 , +1 ) - CF( NHA3 , NB1 , -1 ) )
      IF ( NB0 .GT. 0 ) THEN
      C3 = ZG2 * ( CF( NHA5 , NBM , +1 ) - CF( NHA5 , NBM , -1 ) )
      ELSE
      C3 = ( 0.0 , 0.0 )
      ENDIF
      C4 = ZG2 * ( CF( NHA5 , NB0 , +1 ) - CF( NHA5 , NB0 , -1 ) )
      CE(1,2) =-CI1 * 0.25 * ZKO1 * (ZA * ZRTM**ZA2 * (C1 - ZRIPAR * C2) &
       + ZRTM**(1.0+ZA2) * (ZB * ZRIPAR * C3 - ZMU * C4))
!
      C1 = ZG1 * ( CF( NHA3 , NB1 , +1 ) - CF( NHA3 , NB1 , -1 ) )
      C2 = ZG1 * ( CF( NHA3 , NB2 , +1 ) - CF( NHA3 , NB2 , -1 ) )
      C3 = ZG2 * ( CF( NHA5 , NB0 , +1 ) - CF( NHA5 , NB0 , -1 ) )
      C4 = ZG2 * ( CF( NHA5 , NB1 , +1 ) - CF( NHA5 , NB1 , -1 ) )
      CE(1,3) = 0.5 * ZKO1 * ZKO2 * (ZA * ZRTM**ZA2 * (C1 - ZRIPAR * C2) &
       + ZRTM**(1.0+ZA2) * (ZB * ZRIPAR * C3 - ZMU * C4))
!
      C1 = ZG1 * ( CF( NHA3 , NB2 , +1 ) + CF( NHA3 , NB2 , -1 ) )
      C3 = ZG2 * ( CF( NHA5 , NB0 , +1 ) + CF( NHA5 , NB0 , -1 ) )
      C4 = ZG2 * ( CF( NHA5 , NB2 , +1 ) + CF( NHA5 , NB2 , -1 ) )
      C5 = ZG1 * ( ZB * CF( NHA3, NB0, 0 ) - ZMU * CF( NHA3, NB2, 0 ) )
      CE(3,3) = ZKO1 * ( ZRTM**ZA2 * C5 + ZKO2**2 * ( ZRTM**ZA2 * &
       ZA * ZFCYCL / ZWFR * C1 + ZRTM**(1.0+ZA2) * ( ZB * ( 1.0 - &
       ZFCYCL / ZWFR ) * C3 - ZMU * C4 ) ) )
!
!**
!**   WE ADD THE SECOND HARMONIC TERMS HERE
!**
      NHA7 = ( JNPER + 8 ) / 2
      ZG1  = ZGAMMA( 2 + JNPER/2 )
      ZG2  = ZGAMMA( 3 + JNPER/2 )
      ZK3  = (ZPLFR/ZWFR)**2 * (TWOPI/ZMU)**1.5 * ZC * ZMU * 0.25
      ZZ1  = ZA * ZRTM**( 1.0 + ZA2 ) * ZG1
      ZZ2  = ZRTM**( 2.0 + ZA2 ) * ZG2
!
      C1   = ZZ1 * ( CF( NHA5, NB0, 2 ) - ZRIPAR * CF( NHA5, NB1, 2 ) )
      IF ( JNPAR .EQ. 0.0 ) THEN
        C2 = - ZZ2 * ZMU * CF( NHA7 , NB0 , 2 )
      ELSE
        C2 = ZZ2 * ( ZB*ZRIPAR*CF(NHA7,NBM,2) - ZMU*CF(NHA7,NB0,2) )
      ENDIF
      C211 = ZKO2**2 * ZK3 * ( C1 + C2 )
!
      C1   = ZZ1 * ( CF( NHA5, NB1, 2 ) - ZRIPAR * CF( NHA5, NB2, 2 ) )
      C2   = ZZ2 * ( ZB*ZRIPAR*CF(NHA7,NB0,2) - ZMU*CF(NHA7,NB1,2) )
      C213 = ZKO2**3 * ZK3 * ( C1 + C2 )
!
      C1   = ZZ1 * CF( NHA5, NB2, 2 ) * 2.0 * ZFCYCL / ZWFR
      ZZZZ = 1.0 - 2.0 * ZFCYCL / ZWFR
      C2   = ZZ2 * ( ZB*ZZZZ*CF(NHA7,NB0,2) - ZMU*CF(NHA7,NB2,2) )
      C233 = ZKO2**4 * ZK3 * ( C1 + C2 )
!
!   WE HAVE ACCURACY PROBLEMS FOR N= +/- 1 TERMS
!   IF WE ARE TOO FAR FROM THE FUNDAMENTAL RESONANCE
!   WE ARE ALOWED TO USE THE COLD RESULTS THERE.
      IF( ZWFR .GT. 1.3*ZFCYCL ) THEN
        ZHELP = AIMAG( CE( 1 , 1 ) )
        CE( 1 , 1 ) =   - ZPLFR**2 / ( ZWFR**2 - ZFCYCL**2 )
        CE( 1 , 1 ) = CE( 1 , 1 ) + CI1 * ZHELP
        ZHELP = REAL( CE( 1 , 2 ) )
        CE( 1 , 2 ) =   CI1*ZPLFR**2*ZFCYCL / (ZWFR*(ZWFR**2-ZFCYCL**2))
        CE( 1 , 2 ) = CE( 1 , 2 ) + ZHELP
        ZHELP = AIMAG( CE( 1 , 3 ) )
        CE( 1 , 3 ) = CI1 * ZHELP
        ZHELP = AIMAG( CE( 3 , 3 ) )
        CE( 3 , 3 ) = - ( ZPLFR / ZWFR )**2
        CE( 3 , 3 ) = CE( 3 , 3 ) + CI1 * ZHELP
      ENDIF
      CE( 1 , 1 ) =   CE( 1 , 1 ) + C211
      CE( 2 , 2 ) =   CE( 1 , 1 )
      CE( 1 , 2 ) =   CE( 1 , 2 ) - CI1 * C211
      CE( 2 , 1 ) = - CE( 1 , 2 )
      CE( 1 , 3 ) =   CE( 1 , 3 ) + C213
      CE( 2 , 3 ) =   CI1 * CE( 1 , 3 )
      CE( 3 , 1 ) =   CE( 1 , 3 )
      CE( 3 , 2 ) = - CE( 2 , 3 )
      CE( 3 , 3 ) =   CE( 3 , 3 ) + C233
      return
      END
!DECK ZETA
      COMPLEX function ZETA(Z)

!-----------------------------------------------------------------------
!     ALGORITHM PROGRAMMED ACCORDING TO
!     "EFFICIENT COMPUTATION OF THE PLASMA DISPERSION function ZETA(Z)",
!     BY T.WATANABE,
!     INSTITUTE FOR FUSION THEORY,HIROSHIMA UNIVERSITY,
!     HIROSHIMA,JAPAN
!-----------------------------------------------------------------------

      IMPLICIT LOGICAL (A-Z)

      COMMON/HH/ H
      REAL*8 H

      COMPLEX RESULT,Z,ZETA1,ZETA2,ZETA3,ZETA4
      REAL*8 ABSIMZ,ABSREZ,DIFF,IMZ,REZ,LIMIT

      LIMIT=H/4.
      IMZ=AIMAG(Z)
      REZ=Z
      ABSIMZ=ABS(IMZ)
      ABSREZ=ABS(REZ)
      DIFF=ABS(REZ/H-INT(REZ/H))

      IF ( ABSIMZ .LT. LIMIT ) GO TO 20
      IF ( ABSIMZ .GT. 2. ) GO TO 300
10    CONTINUE
      IF ( ABSREZ .GT. 2. ) GO TO 300
      GO TO 100
20    CONTINUE
      IF ( ( DIFF .LT. .25 ) .OR. ( DIFF .GT. .75 ) ) GO TO 30
      GO TO 10
30    CONTINUE
      IF ( ABSREZ .GE. 2.-LIMIT ) GO TO 400
      GO TO 200

100   CONTINUE
      RESULT=ZETA1(Z)
      GO TO 500
200   CONTINUE
      RESULT=ZETA2(Z)
      GO TO 500
300   CONTINUE
      RESULT=ZETA3(Z)
      GO TO 500
400   CONTINUE
      RESULT=ZETA4(Z)
500   CONTINUE
      ZETA=RESULT

      return
!**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
!*F45V1P0*
      COMPLEX function ZETA1(Z)

!-----------------------------------------------------------------------
!     ALGORITHM PROGRAMMED ACCORDING TO
!     "EFFICIENT COMPUTATION OF THE PLASMA DISPERSION function ZETA(Z)",
!     BY T.WATANABE,
!     INSTITUTE FOR FUSION THEORY,HIROSHIMA UNIVERSITY,
!     HIROSHIMA,JAPAN
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!     REGION 1,FORMULA 4
!-----------------------------------------------------------------------

      IMPLICIT LOGICAL (A-Z)

      COMMON/HH/ H
      REAL*8 H

      COMMON/II/ I
      COMPLEX I

      COMMON/NN/ NSUM
      INTEGER NSUM

      COMMON/PP/ PI,ROOTPI,TWOPI
      REAL*8 PI,ROOTPI,TWOPI

      COMPLEX A,B,C,RESIDU,SUM,Z,Z2,SEXP
      INTEGER N
      REAL*8 EXPON,IMZ,LIMIT,NH2

      IMZ=AIMAG(Z)
      Z2=Z**2
      LIMIT=PI/H
      RESIDU=CMPLX(0.,0.)
      SUM=CMPLX(0.,0.)

      DO 100 N=1,NSUM
      NH2=(N*H)**2
      EXPON=EXP(-NH2)
      SUM=SUM+EXPON/(NH2-Z2)
100   CONTINUE
      IF(IMZ.GT.LIMIT) GO TO 600
      A=I*TWOPI*Z/H
      A=1.-SEXP(-A)
      B=SEXP(-Z2)
      B=I*ROOTPI*B
      C=B/A

      IF ( IMZ .EQ. LIMIT ) GO TO 200
      IF ( ABS(IMZ) .LT. LIMIT ) GO TO 300
      IF ( IMZ .EQ. -LIMIT ) GO TO 400
      IF ( IMZ .LT. -LIMIT ) GO TO 500

200   CONTINUE
      RESIDU=C
      GO TO 600
300   CONTINUE
      RESIDU=2.*C
      GO TO 600
400   CONTINUE
      RESIDU=B+C
      GO TO 600
500   CONTINUE
      RESIDU=2.*B

600   CONTINUE
      ZETA1=(-1./Z+2.*Z*SUM)*H/ROOTPI+RESIDU

      return
!**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
!*F45V1P0*
      COMPLEX function ZETA2(Z)

!-----------------------------------------------------------------------
!     ALGORITHM PROGRAMMED ACCORDING TO
!     "EFFICIENT COMPUTATION OF THE PLASMA DISPERSION function ZETA(Z)",
!     BY T.WATANABE,
!     INSTITUTE FOR FUSION THEORY,HIROSHIMA UNIVERSITY,
!     HIROSHIMA,JAPAN
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!     REGION 2,FORMULA 10
!-----------------------------------------------------------------------

      IMPLICIT LOGICAL (A-Z)

      COMMON/HH/ H
      REAL*8 H

      COMMON/II/ I
      COMPLEX I

      COMMON/NN/ NSUM
      INTEGER NSUM

      COMMON/PP/ PI,ROOTPI,TWOPI
      REAL*8 PI,ROOTPI,TWOPI

      COMPLEX A,B,C,RESIDU,SUM,Z,Z2,SEXP
      INTEGER N
      REAL*8 EXPON,IMZ,LIMIT,NH2

      INTEGER NSUM1

      IMZ=AIMAG(Z)
      Z2=Z**2
      LIMIT=PI/H
      RESIDU=CMPLX(0.,0.)
      SUM=CMPLX(0.,0.)

      NH2=H**2/4.
      EXPON=EXP(-NH2)
      SUM=SUM+EXPON/(NH2-Z2)
      NSUM1=NSUM-1

      DO 100 N=1,NSUM1
      NH2=((N+.5)*H)**2
      EXPON=EXP(-NH2)
      SUM=SUM+EXPON/(NH2-Z2)
100   CONTINUE
      IF (IMZ.GT.LIMIT) GO TO 600
      A=I*TWOPI*Z/H
      A=1.+SEXP(-A)
      B=SEXP(-Z2)
      B=I*ROOTPI*B
      C=B/A

      IF ( IMZ .EQ. LIMIT ) GO TO 200
      IF ( ABS(IMZ) .LT. LIMIT ) GO TO 300
      IF ( IMZ .EQ. -LIMIT ) GO TO 400
      IF (IMZ .LT. -LIMIT ) GO TO 500

200   CONTINUE
      RESIDU=C
      GO TO 600
300   CONTINUE
      RESIDU=2.*C
      GO TO 600
400   CONTINUE
      RESIDU=3.*C-B
      GO TO 600
500   CONTINUE
      RESIDU=2.*B

600   CONTINUE
      ZETA2=2.*H*Z*SUM/ROOTPI+RESIDU

      return
!**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
!*F45V1P0*
      COMPLEX function ZETA3(Z)

!-----------------------------------------------------------------------
!     ALGORITHM PROGRAMMED ACCORDING TO
!     "EFFICIENT COMPUTATION OF THE PLASMA DISPERSION function ZETA(Z)",
!     BY T.WATANABE,
!     INSTITUTE FOR FUSION THEORY,HIROSHIMA UNIVERSITY,
!     HIROSHIMA,JAPAN
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!     REGION 3,FORMULA 9
!-----------------------------------------------------------------------

      IMPLICIT LOGICAL (A-Z)

      COMMON/HH/ H
      REAL*8 H

      COMMON/II/ I
      COMPLEX I

      COMMON/NN/ NSUM
      INTEGER NSUM

      COMMON/PP/ PI,ROOTPI,TWOPI
      REAL*8 PI,ROOTPI,TWOPI

      COMPLEX A,B,C,RESIDU,SUM,Z,Z2,SEXP
      INTEGER N
      REAL*8 EXPON,IMZ,LIMIT,NH2

      IMZ=AIMAG(Z)
      Z2=Z**2
      LIMIT=PI/H
      RESIDU=CMPLX(0.,0.)
      SUM=CMPLX(0.,0.)
      IF (IMZ.GT.LIMIT) GO TO 600
      DO 100 N=1,NSUM
      NH2=(N*H)**2
      EXPON=EXP(-NH2)
      SUM=SUM+NH2**2*EXPON/(NH2-Z2)
100   CONTINUE

      A=I*TWOPI*Z/H
      A=1.-SEXP(-A)
      B=SEXP(-Z2)
      B=I*ROOTPI*B
      C=B/A

      IF ( IMZ .EQ. LIMIT ) GO TO 200
      IF ( ABS(IMZ) .LT. LIMIT ) GO TO 300
      IF ( IMZ .EQ. -LIMIT ) GO TO 400
      IF ( IMZ .LT. -LIMIT ) GO TO 500

200   CONTINUE
      RESIDU=C
      GO TO 600
300   CONTINUE
      RESIDU=2.*C
      GO TO 600
400   CONTINUE
      RESIDU=B+C
      GO TO 600
500   CONTINUE
      RESIDU=2.*B

600   CONTINUE
      ZETA3=-1./Z+(-.5+2.*H*SUM/ROOTPI)/Z**3+RESIDU

      return
!**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
!*F45V1P0*
      COMPLEX function ZETA4(Z)

!-----------------------------------------------------------------------
!     ALGORITHM PROGRAMMED ACCORDING TO
!     "EFFICIENT COMPUTATION OF THE PLASMA DISPERSION function ZETA(Z)",
!     BY T.WATANABE,
!     INSTITUTE FOR FUSION THEORY,HIROSHIMA UNIVERSITY,
!     HIROSHIMA,JAPAN
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
!     REGION 4,FORMULA 11
!-----------------------------------------------------------------------

      IMPLICIT LOGICAL (A-Z)

      COMMON/HH/ H
      REAL*8 H

      COMMON/II/ I
      COMPLEX I

      COMMON/NN/ NSUM
      INTEGER NSUM

      COMMON/PP/ PI,ROOTPI,TWOPI
      REAL*8 PI,ROOTPI,TWOPI

      COMPLEX A,B,C,RESIDU,SUM,Z,Z2,SEXP
      INTEGER N
      REAL*8 EXPON,IMZ,LIMIT,NH2

      INTEGER NSUM1

      IMZ=AIMAG(Z)
      Z2=Z**2
      LIMIT=PI/H
      RESIDU=CMPLX(0.,0.)
      SUM=CMPLX(0.,0.)

      NH2=H**2/4.
      EXPON=EXP(-NH2)
      SUM=SUM+NH2**2*EXPON/(NH2-Z2)
      NSUM1=NSUM-1

      DO 100 N=1,NSUM1
      NH2=((N+.5)*H)**2
      EXPON=EXP(-NH2)
      SUM=SUM+NH2**2*EXPON/(NH2-Z2)
100   CONTINUE
      IF (IMZ.GT.LIMIT) GO TO 600
      A=I*TWOPI*Z/H
      A=1.+SEXP(-A)
      B=SEXP(-Z2)
      B=I*ROOTPI*B
      C=B/A

      IF ( IMZ .EQ. LIMIT ) GO TO 200
      IF ( ABS(IMZ) .LT. LIMIT ) GO TO 300
      IF ( IMZ .EQ. -LIMIT ) GO TO 400
      IF ( IMZ .LT. -LIMIT ) GO TO 500

200   CONTINUE
      RESIDU=C
      GO TO 600
300   CONTINUE
      RESIDU=2.*C
      GO TO 600
400   CONTINUE
      RESIDU=3.*C-B
      GO TO 600
500   CONTINUE
      RESIDU=2.*B

600   CONTINUE
      ZETA4=-1./Z+(-.5+2.*H*SUM/ROOTPI)/Z**3+RESIDU

      return
!**  THIS PROGRAM VALID ON FTN4 AND FTN5 **
      END
      COMPLEX function SEXP(ARG)
      COMPLEX ARG
!        THIS IS CEXP WITH UNDERFLOW TREATED AS (0,0)
!        AND OVERFLOW AS EXP(EXPLIM,IMAG(ARG))
      PARAMETER(EXPLIM=700.0)

      IF (REAL(ARG).GT.-EXPLIM) THEN
         IF (REAL(ARG).GT.EXPLIM) THEN
         SEXP=CEXP(CMPLX(EXPLIM,AIMAG(ARG)))
         ELSE
         SEXP=CEXP(ARG)
         END IF
         ELSE
      SEXP=(0.,0.)
      END IF

      return

      END

      REAL*8 function ZGAMMA( KN )
!
!   THIS function CALCULATES THE GAMMA-function
!   OF ARGUMENT KN: G( KN ) = (KN - 1)!.
      IF ( KN .LE. 2 ) THEN
        ZGAMMA = 1
      ELSE
        ZKFAC = 1
        DO 100 JK = 2, ( KN - 1 )
  100   ZKFAC = ZKFAC * REAL( JK )
        ZGAMMA = ZKFAC
      ENDIF
      return
      END
      REAL*8 function GAMMAH( KQ )
!
!   THIS function CALCULATES THE GAMMA-function
!   OF ARGUMENT KQ - 1/2.
      COMMON / PP / PI , ROOTPI , TWOPI
      ZFF = 1.0
      DO 100 JJ = 1 , ( 2 * KQ - 3 ) , 2
  100 ZFF = ZFF * REAL( JJ )
      GAMMAH = ROOTPI * ZFF / 2.0**( KQ - 1 )
      return
      end 



!     *********************************************************************
      subroutine DISTR( PGAMMA, PPPAR, PNPAR, PNY, PF, PU, PW )
!     *********************************************************************
!
!   returnS U AND W FOR A RELATIVISTIC MAXWELLIAN
!   DISTRIBUTION function AT THE INPUT VALUES FOR
!   P PARRALLEL AND GAMMA.
      COMMON/ COMMU  / ZMU,ZC,JLOSS
      ZPPER = SQRT( PGAMMA**2 - PPPAR**2 - 1.0 )
      ZF0   = ZC * EXP( -ZMU * (PGAMMA-1.0) )
      PF    = ZF0
      PU    = -ZMU * ZPPER**(2*JLOSS+1) * ZF0 / PGAMMA
      PW    = -ZMU * PPPAR * ZPPER**(2*JLOSS) * ZF0 / PGAMMA
      IF( JLOSS .GT. 0 ) THEN
        PU     = PU + 2.0*JLOSS * ZPPER**(2*JLOSS-1) * ZF0
        ZF     = -2.0*JLOSS * PPPAR * ZPPER**(2*JLOSS-1) * ZF0
        PU     = PU + PNPAR * ZF / PGAMMA
        PW     = PW -  PNY  * ZF / (PGAMMA * ZPPER)
        PF     = ZPPER**(2*JLOSS) * ZF0
      ENDIF
      return
      END


!     *********************************************************************
      function FK( Z , J , NACCUR )
!     *********************************************************************
!
!   THIS function CALCULATES AN APPROXIMATION
!   TO THE MODIFIED BESSEL function OF THE
!   SECOND KIND OF INDEX J FOR LARGE ARGUMENT Z.
!   THE LEADING TERM IS NORMALIZED TO 1!
!   WE USE EQUATION 8.451.6 OF G&R.
      ZK = 1.
      DO 100  I = 1 , NACCUR
        ZTERM = 1.0 / ( ( 2.0 * Z )**I * FAC( I ) )
        P1 = REAL( J + I )
        P2 = REAL( J - I )
        ZFAC = FGAMMA( P1 + 0.5 ) / FGAMMA( P2 + 0.5 )
        ZK    = ZK + ZFAC * ZTERM
  100 CONTINUE
      FK = ZK
      return
      END



!     *********************************************************************
      function FGAMMA( PQ )
!     *********************************************************************
!
!   THIS function CALCULATES THE GAMMA-function
!   OF ARGUMENT PQ, WHERE PQ = KQ + 1/2.
      PI = 4.0 * ATAN( 1.0 )
      TWOPI = 2.0 * PI
      ROOTPI = SQRT( PI )
      KQ = NINT( PQ - .5 )
      ZFF = 1.0
      IF( KQ .GE. 0 ) THEN
        DO 100 JJ = 1,( 2 * KQ - 1 ),2
  100   ZFF = ZFF * REAL( JJ )
        FGAMMA = ROOTPI * ZFF / 2.0**KQ
        return
      ELSE
        KQ = ABS( KQ )
        DO 200 JJ = 1 , ( 2 * KQ - 1 ) , 2
  200   ZFF = ZFF * REAL( JJ )
        FGAMMA = ( -2. )**KQ * ROOTPI / ZFF
      ENDIF
      return
      END


!     *********************************************************************
      function FAC( N )
!     *********************************************************************
!
!   THIS function CALCULATES N!.
      F = 1.0
      DO 100 J = 1 , N
  100 F = F * REAL( J )
      FAC = F
      return
      end function FAC



!     *********************************************************************
      subroutine FREPSLN( PX,PY,PNPAR,PNPER,PMU,NMIN,NMAX,NTERM,NPM,NGM,NAM,CE)
!     *********************************************************************

!   subroutine FOR COMPLETELY NUMERICAL EVALUATION OF THE
!   DIELECTRIC TENSOR.
!   THE USER MUST SUPLY A ROUTINE DISTR( GAM, PPAR, NPAR, N*Y, F, U, W )
!   WHICH PROVIDES THE VALUES OF U AND W AS DIFINED IN EQ.
!   2.3.20 (BORNATICI ET AL., NUCLEAR FUSION 23 (1983),
!   1153) FOR THE PLASMA DISTRIBUTION function.

!   INPUT:
!         PX     = ( OMEGA P / OMEGA )**2
!         PY     =   OMEGA C / OMEGA
!         PNPAR  = PARALLEL REFRACTIVE INDEX
!         PNPER  = PERPENDICULAR REFRACTIVE INDEX
!         PMU    = MUST CORRESPOND TO THE ASYMPTOTIC BEHAVIOUR
!                  OF THE PLASMA DISTRIBUTION function
!         NMIN, NMAX  MUST SPECIFY THE RANGE OF HARMONICS TO BE INCLUDED

!   OUTPUT:
!        CE(I,J)   = I,J TH ELEMENT OF THE DIELECTRIC TENSOR

      IMPLICIT COMPLEX(C)
      external  D01BAW, D01BAX, D01BAZ
      DIMENSION  CE( 3, 3 )
      DIMENSION  ZPW( 96 ), ZPP( 96 ), ZGW( 64 ), ZGG( 64 )
      DIMENSION ZZW(64), ZZP(64)
      DIMENSION  ZZZZ( 6 ), ZZZSUM( 6 ), ZZSUM( 6 ), ZSUM( 6 )
      DIMENSION  ASUM( 6 ), AASUM( 6 ), ZAW( 64 ), ZAP( 64 )
      DIMENSION  CETA( 3, 3 ), ZHH( 6 ), ESUM( 6 ), EESUM( 6 )
      DIMENSION  ZS( 6 ), ZSS( 6 ), ZCOREC( 6 ), ZZS( 6 ), ZZZ( 6 )
      REAL*8 MMDEI

! WE MUST INITIALIZE ZAP TO PREVENT UNDEFINEDS
      DO 1 I = 1, 64
    1 ZAP( I ) = 0.0

      ZPI   = 4.0 * ATAN( 1.0 )
      CI1    = ( 0.0, 1.0 )
      IFAIL  = 0
      HALFMU = 0.5 * PMU
      Z1MNP2 = 1.0 - PNPAR**2
      ZMUNP  = PMU * PNPAR
      IF( NPM .LE. 0 ) NPM    = 16
      IF( NGM .LE. 0 ) NGM    = 16
      IF( NAM .LE. 0 ) NAM    = 16

!   THE HERMITIAN PART


!   OUTERMOST LOOP: SUMMATION OVER THE HARMONICS
      IF( NMIN .LE. -1 ) THEN
        ZSUM( 1 )   = 0.0
        ZSUM( 2 )   = 0.0
        ZSUM( 3 )   = 0.0
        ZSUM( 4 )   = 0.0
        ZSUM( 5 )   = 0.0
        ZSUM( 6 )   = 0.0
      ELSEIF( NMIN .EQ. 0 ) THEN
        ZSUM( 1 )   = 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 2 )   = 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 3 )   = 0.0
        ZSUM( 4 )   = - 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 5 )   = 0.0
        ZSUM( 6 )   = 0.0
      ELSEIF( NMIN .EQ. 1 ) THEN
        ZSUM( 1 )   = 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 2 )   = 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 3 )   = 1.0 / ( 2.0*ZPI )
        ZSUM( 4 )   = - 1.0 / ( 4.0*ZPI * (1.0+PY) )
        ZSUM( 5 )   = 0.0
        ZSUM( 6 )   = 0.0
      ELSE
        ZSUM( 1 )   = 1.0 / ( 2.0*ZPI * (1.0-PY**2) )
        ZSUM( 2 )   = 1.0 / ( 2.0*ZPI * (1.0-PY**2) )
        ZSUM( 3 )   = 1.0 / ( 2.0*ZPI )
        ZSUM( 4 )   =  PY / ( 2.0*ZPI * (1.0-PY**2) )
        ZSUM( 5 )   = 0.0
        ZSUM( 6 )   = 0.0
      ENDIF

      ASUM( 1 ) = 0.0
      ASUM( 2 ) = 0.0
      ASUM( 3 ) = 0.0
      ASUM( 4 ) = 0.0
      ASUM( 5 ) = 0.0
      ASUM( 6 ) = 0.0

      DO 10000  N = NMIN, NMAX

        ZNY    = N * PY

!   MIDDLE LOOP: THE PPAR INTEGRATION
        ZZSUM( 1 )  = 0.0
        ZZSUM( 2 )  = 0.0
        ZZSUM( 3 )  = 0.0
        ZZSUM( 4 )  = 0.0
        ZZSUM( 5 )  = 0.0
        ZZSUM( 6 )  = 0.0

!       IF A RESONANCE IS PRESENT THE INTEGRATION IS SPLIT INTO PARTS
!       FIRST CALCULATE P MIN AND P PLUS (BORDERS OF MIDDLE RANGE)
        IF( ABS( PNPAR ) .LT. 1.0 ) THEN

          IF( N  .LE. 0 ) THEN
            ZPMIN  = -1.0E+20
            ZPPLUS =  1.0E+20
          ELSE IF( PY .LE. SQRT(Z1MNP2)/REAL(N) ) THEN
            ZPMIN  = -1.0E+20
            ZPPLUS =  1.0E+20
          ELSE
            ZB     = PNPAR * N * PY / Z1MNP2
            ZC     = ( 1.0 - (N*PY)**2 ) / Z1MNP2
            ZPPLUS = ZB + SQRT( ZB**2 - ZC )
            ZPMIN  = ZB - SQRT( ZB**2 - ZC )
          ENDIF

        ELSE IF( ABS( PNPAR ) .EQ. 1.0 ) THEN

          ZPPLUS = 0.5 * ( 1.0/(N*PY) - N*PY )
          ZPMIN  = 0.5 * ( N*PY - 1.0/(N*PY) )

        ELSE
!       ABS( NPAR ) IS GREATER THAN 1

          ZB     = PNPAR * N * PY / Z1MNP2
          ZC     = ( 1.0 - (N*PY)**2 ) / Z1MNP2
          ZPPLUS = ZB + SQRT( ZB**2 - ZC )
          ZPMIN  = ZB - SQRT( ZB**2 - ZC )

        ENDIF

!       FOR ABS( NPAR ) GE 1  WE MUST RESET P MIN AND P PLUS
!       BECAUSE ONLY ONE OF BOTH IS A REAL SOLUTION
        IF( PNPAR .GE. 1.0 ) THEN
          ZPMIN  = ZPPLUS
          ZPPLUS = 1.0E+20
        ENDIF
        IF( PNPAR .LE. -1.0 ) THEN
          ZPPLUS = ZPMIN
          ZPMIN  = -1.0E+20
        ENDIF

!       FIRST CALCULATE THE WEIGTHS AND ABCISSAE FOR
!       THE GAUSS-HERMITE INTEGRATION.
        CALL D01BBF( D01BAW, 0.0, HALFMU, 1, NPM, ZPW, ZPP, IFAIL )

!       CHECK IF ALL POINTS ARE INDIDE THE MIDDLE REGION
        IF( (ZPP(NPM).GT.ZPMIN) .AND. (ZPP(1).LT.ZPPLUS) ) THEN
!         USE ONLY GAUSS-HERMITE IN MIDDLE REGION

          KPM = NPM

        ELSE IF( ABS(PNPAR) .LE. SQRT(1.0/PMU) ) THEN
!         USE GAUSS-HERMITE ON OUTER AND GAUSS-LEGENDRE ON MIDDLE REGION

          CALL D01BBF( D01BAW,ZPMIN,HALFMU,1,2*NPM,ZZW,ZZP,IFAIL )
          DO  10  KP = 1, NPM
            ZPW( KP ) = ZZW( KP+NPM )
            ZPP( KP ) = ZZP( KP+NPM )
   10     CONTINUE
          CALL D01BBF( D01BAZ,ZPMIN,ZPPLUS,1,NPM,ZZW,ZZP,IFAIL )
          DO  15  KP = 1, NPM
            KK        = NPM + KP
            ZPW( KK ) = ZZW( KP )
            ZPP( KK ) = ZZP( KP )
   15     CONTINUE
          CALL D01BBF( D01BAW,ZPPLUS,HALFMU,1,2*NPM,ZZW,ZZP,IFAIL )
          DO  20  KP = 1, NPM
            KK        = 2 * NPM + KP
            ZPW( KK ) = ZZW( KP )
            ZPP( KK ) = ZZP( KP )
   20     CONTINUE
          KPM    = 3 * NPM

        ELSE IF( PNPAR .GT. 0.0 ) THEN
!         THE INTEGRAND FALLS OFF FROM P MIN TO -INFINITY / P PLUS
!         CHECK IF GAUSS-LAGUERRE POINTS ARE ALL INSIDE MIDDLE REGION

          CALL D01BBF( D01BAX,ZPMIN ,ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
          IF( ZZP( NPM ) .LT. ZPPLUS ) THEN
!           USE GAUSS-LAGUERRE ON MIDDLE AND FIRST REGION ONLY

            DO  25  KP = 1, NPM
              ZPW( KP ) = ZZW( KP )
              ZPP( KP ) = ZZP( KP )
   25       CONTINUE
            CALL D01BBF( D01BAX,ZPMIN,-ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  30  KP = 1, NPM
              KK        = NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   30       CONTINUE
            KPM    = 2 * NPM

          ELSE
!           USE GAUSS-LEGENDRE ON MIDDLE AND GAUSS-LAGUERRE ON OUTER

            CALL D01BBF( D01BAX,ZPMIN,-ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  35  KP = 1, NPM
              ZPW( KP ) = ZZW( KP )
              ZPP( KP ) = ZZP( KP )
   35       CONTINUE
            CALL D01BBF( D01BAZ,ZPMIN,ZPPLUS,1,NPM,ZZW,ZZP,IFAIL )
            DO  40  KP = 1, NPM
              KK        = NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   40       CONTINUE
            CALL D01BBF( D01BAX,ZPPLUS,ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  45  KP = 1, NPM
              KK        = 2 * NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   45       CONTINUE
            KPM    = 3 * NPM

          ENDIF

        ELSE
!         PNPAR IS SMALLER THAN 0.0
!         THE INTEGRAND FALLS OFF FROM P PLUS TO P MIN / +INFINITY

          CALL D01BBF( D01BAX,ZPPLUS, ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
          IF( ZZP( NPM ) .GT. ZPMIN ) THEN
!           USE GAUSS-LAGUERRE ON MIDDLE AND THIRD REGION ONLY

            DO  50  KP = 1, NPM
              ZPW( KP ) = ZZW( KP )
              ZPP( KP ) = ZZP( KP )
   50       CONTINUE
            CALL D01BBF( D01BAX,ZPPLUS,-ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  55  KP = 1, NPM
              KK        = NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   55       CONTINUE
            KPM    = 2 * NPM

          ELSE
!           USE GAUSS-LEGENDRE ON MIDDLE AND GAUSS-LAGUERRE ON OUTER

            CALL D01BBF( D01BAX,ZPMIN, ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  60  KP = 1, NPM
              ZPW( KP ) = ZZW( KP )
              ZPP( KP ) = ZZP( KP )
   60       CONTINUE
            CALL D01BBF( D01BAZ,ZPMIN,ZPPLUS,1,NPM,ZZW,ZZP,IFAIL )
            DO  65  KP = 1, NPM
              KK        = NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   65       CONTINUE
            CALL D01BBF( D01BAX,ZPPLUS,-ZMUNP,1,NPM,ZZW,ZZP,IFAIL )
            DO  70  KP = 1, NPM
              KK        = 2 * NPM + KP
              ZPW( KK ) = ZZW( KP )
              ZPP( KK ) = ZZP( KP )
   70       CONTINUE
            KPM    = 3 * NPM

          ENDIF

        ENDIF


        DO 1000  I = 1, KPM
          IF( I .EQ. 0 ) GOTO 1000

          ZPPAR  = ZPP( I )


!   INNERMOST LOOP: THE GAMMA INTEGRATION
          ZGAMS  = N * PY + PNPAR * ZPPAR
          ZGMIN  = SQRT( 1.0 + ZPPAR**2 )
          ZZZSUM( 1 ) = 0.0
          ZZZSUM( 2 ) = 0.0
          ZZZSUM( 3 ) = 0.0
          ZZZSUM( 4 ) = 0.0
          ZZZSUM( 5 ) = 0.0
          ZZZSUM( 6 ) = 0.0

!   FIRST CALCULATE THE WEIGTHS
          CALL D01BBF( D01BAX, ZGMIN, PMU, 1, NGM, ZGW, ZGG, IFAIL )

!   CHECK FOR RESONANCE IN INTEGRATION INTERVAL
          IF( ZGAMS .GT. ZGMIN ) THEN

!           CALCULATE THE 0TH ORDER TERM AROUND THE RESONANCE
            CALL DISTR( ZGAMS, ZPPAR, PNPAR, ZNY, ZF, ZUS, ZWS )
            ZPPERS = SQRT( ZGAMS**2 - ZPPAR**2 - 1.0 )
            ZBS    = PNPER * ZPPERS / PY
            ZJN    = BESSEL( N, ZBS, NTERM )
            ZJNP   = BESSLP( N, ZBS, NTERM )
            ZSS( 1 )    =  ZPPERS * ZUS * ( N * ZJN / ZBS )**2
            ZSS( 2 )    =  ZPPERS * ZUS * ZJNP**2
            ZSS( 3 )    =  ZPPAR  * ZWS * ZJN**2
            ZSS( 4 )    =  ZPPERS * ZUS * N * ZJN * ZJNP / ZBS
            ZSS( 5 )    =  ZPPAR  * ZUS * N * ZJN**2 / ZBS
            ZSS( 6 )    = -ZPPAR  * ZUS * ZJN * ZJNP

          ELSE

            ZSS( 1 )    = 0.0
            ZSS( 2 )    = 0.0
            ZSS( 3 )    = 0.0
            ZSS( 4 )    = 0.0
            ZSS( 5 )    = 0.0
            ZSS( 6 )    = 0.0

          ENDIF

          DO 100  J = 1, NGM

            ZGAMMA = ZGG( J )
            ZPPER  = SQRT( ZGAMMA**2 - ZPPAR**2 - 1.0 )
            CALL DISTR( ZGAMMA, ZPPAR, PNPAR, ZNY, ZF, ZU, ZW )
            ZB     = PNPER * ZPPER / PY
            ZJN    = BESSEL( N, ZB, NTERM )
            ZJNP   = BESSLP( N, ZB, NTERM )
            ZS( 1 )     =  ZPPER * ZU * ( N * ZJN / ZB )**2
            ZS( 2 )     =  ZPPER * ZU * ZJNP**2
            ZS( 3 )     =  ZPPAR * ZW * ZJN**2
            ZS( 4 )     =  ZPPER * ZU * N * ZJN * ZJNP / ZB
            ZS( 5 )     =  ZPPAR * ZU * N * ZJN**2 / ZB
            ZS( 6 )     = -ZPPAR * ZU * ZJN * ZJNP

            DO 90 K = 1, 6
              IF( ZGAMS .GT. ZGMIN ) THEN
                ZZS( K )    = ZSS( K ) * REXP( -PMU * ( ZGAMMA-ZGAMS ) )
              ELSE
                ZZS( K )    = 0.0
              ENDIF
              ZZZZ( K )   = ( ZGAMMA * ZS( K ) - ZGAMS * ZZS( K ) ) / &
                                  ( ZGAMS - ZGAMMA )
              ZZZSUM( K ) = ZZZSUM( K ) + ZZZZ( K ) * ZGW( J )
   90       CONTINUE

  100     CONTINUE

          DO 900  K = 1, 6

!           SUBSTRACT TERM FROM RESONANCE CORRECTION
            ZCOREC( K ) = 0.0
            IF( ZGAMS .GT. ZGMIN ) THEN
              ZCOREC( K ) =     EI(    ( -PMU * (ZGMIN-ZGAMS) )      )
              ZCOREC( K ) = ZGAMS * ZSS( K ) * ZCOREC( K )
              ZZZSUM( K ) = ZZZSUM( K ) + ZCOREC( K )
            ENDIF


            ZZZ( K )   = ZZZSUM( K ) * ZPW( I )
            ZZSUM( K ) = ZZSUM( K ) + ZZZ( K )
  900     CONTINUE

 1000   CONTINUE


!   THE ANTI-HERMITIAN PART AND THE EMISSIVITY TENSOR

!   PERFORM THE INTEGRATION OVER THE RESONANCE
        AASUM( 1 ) = 0.0
        AASUM( 2 ) = 0.0
        AASUM( 3 ) = 0.0
        AASUM( 4 ) = 0.0
        AASUM( 5 ) = 0.0
        AASUM( 6 ) = 0.0

!   SEPARATE THE THREE CASES FOR N PARRALLEL
        IF( ABS( PNPAR ) .LT. 1.0 ) THEN
          IF( N  .LE. 0 )              GOTO 3000
          IF( PY .LE. SQRT(Z1MNP2)/REAL(N) ) GOTO 3000
          ZB     = PNPAR * N * PY / Z1MNP2
          ZC     = ( 1.0 - (N*PY)**2 ) / Z1MNP2
          ZPPLUS = ZB + SQRT( ZB**2 - ZC )
          ZPMIN  = ZB - SQRT( ZB**2 - ZC )
        ELSE IF( ABS( PNPAR ) .EQ. 1.0 ) THEN
          ZPPLUS = 0.5 * ( 1.0/(N*PY) - N*PY )
          ZPMIN  = 0.5 * ( N*PY - 1.0/(N*PY) )
        ELSE
          ZB     = PNPAR * N * PY / Z1MNP2
          ZC     = ( 1.0 - (N*PY)**2 ) / Z1MNP2
          ZPPLUS = ZB + SQRT( ZB**2 - ZC )
          ZPMIN  = ZB - SQRT( ZB**2 - ZC )
        ENDIF
!       FOR ABS( NPAR ) GE 1  WE MUST RESET P MIN AND P PLUS
!       BECAUSE ONLY ONE OF BOTH IS A REAL SOLUTION
        IF( PNPAR .GE. 1.0 ) THEN
          ZPMIN  = ZPPLUS
          ZPPLUS = 1.0E+20
        ENDIF
        IF( PNPAR .LE. -1.0 ) THEN
          ZPPLUS = ZPMIN
          ZPMIN  = -1.0E+20
        ENDIF
        IF( PNPAR .LT. 0.0 ) THEN
          CALL D01BBF( D01BAX,ZPPLUS,(PNPAR*PMU),1,NAM,ZAW,ZAP,IFAIL )
        ELSE IF( PNPAR .GT. 0.0 ) THEN
          CALL D01BBF( D01BAX,ZPMIN ,(PNPAR*PMU),1,NAM,ZAW,ZAP,IFAIL )
        ENDIF
        IF( ABS( PNPAR ) .GE. 1.0 ) THEN

        ELSE IF( ( PNPAR .LT. 0.0 ) .AND. ( ZAP(NAM) .GT. ZPMIN ) ) THEN

        ELSE IF( ( PNPAR .GT. 0.0 ) .AND. ( ZAP(NAM) .LT. ZPPLUS )) THEN

        ELSE
          CALL D01BBF( D01BAZ,ZPMIN,ZPPLUS,1,NAM,ZAW,ZAP,IFAIL )
        ENDIF

        DO  2000  I = 1, NAM

          ZPPAR  = ZAP( I )
          ZGAMS  = N * PY + PNPAR * ZPPAR
          ZPPERS = SQRT( ZGAMS**2 - ZPPAR**2 - 1.0 )
          CALL DISTR( ZGAMS, ZPPAR, PNPAR, ZNY, ZF, ZUS, ZWS )
          ZBS    = PNPER * ZPPERS / PY
          ZJN    = BESSEL( N, ZBS, NTERM )
          ZJNP   = BESSLP( N, ZBS, NTERM )
          ZSS( 1 )    =  ZPPERS * ZUS * ( N * ZJN / ZBS )**2
          ZSS( 2 )    =  ZPPERS * ZUS * ZJNP**2
          ZSS( 3 )    =  ZPPAR  * ZWS * ZJN**2
          ZSS( 4 )    =  ZPPERS * ZUS * N * ZJN * ZJNP / ZBS
          ZSS( 5 )    =  ZPPAR  * ZUS * N * ZJN**2 / ZBS
          ZSS( 6 )    = -ZPPAR  * ZUS * ZJN * ZJNP

          DO 1900  K = 1, 6
            AASUM( K ) = AASUM( K ) + ZGAMS * ZSS( K ) * ZAW( I )
 1900     CONTINUE
 2000   CONTINUE
 3000   CONTINUE


        DO 9000  K = 1, 6
          ZSUM( K )  = ZSUM( K ) + ZZSUM( K )
          ASUM( K )  = ASUM( K ) + AASUM( K )
 9000   CONTINUE

10000 CONTINUE


!   THE FINAL RESULTS ARE:
      CE( 1, 1 ) =      - PX*2.0*ZPI * ( ZSUM( 1 ) + CI1*ZPI*ASUM( 1 ) )
      CE( 2, 2 ) =      - PX*2.0*ZPI * ( ZSUM( 2 ) + CI1*ZPI*ASUM( 2 ) )
      CE( 3, 3 ) =      - PX*2.0*ZPI * ( ZSUM( 3 ) + CI1*ZPI*ASUM( 3 ) )
      CE( 2, 1 ) = -CI1 * PX*2.0*ZPI * ( ZSUM( 4 ) + CI1*ZPI*ASUM( 4 ) )
      CE( 3, 1 ) =      - PX*2.0*ZPI * ( ZSUM( 5 ) + CI1*ZPI*ASUM( 5 ) )
      CE( 3, 2 ) = -CI1 * PX*2.0*ZPI * ( ZSUM( 6 ) + CI1*ZPI*ASUM( 6 ) )
      CE( 1, 2 ) = -CE( 2, 1 )
      CE( 1, 3 ) =  CE( 3, 1 )
      CE( 2, 3 ) = -CE( 3, 2 )

      return
      END



!     *********************************************************************
      function BESSEL( N, PX, KTERM )
!     *********************************************************************
!
!   THIS ROUTINE EVALUTES THE FIRST KTERM TERMS
!   FROM THE SMALL ARGUMENT EXPENSION OF THE
!   BESSEL function OF INTEGER ORDER.
      IF( KTERM .LE. 0.0 ) THEN
        BESSEL = 0.0
        return
      ENDIF
      NABS   = ABS( N )
      SGN    = 1
      IF( N .LT. 0 ) SGN = -1
      ZZ     = -0.25 * PX*PX
      ZFAC   =  1.0 / FAC( NABS )
      ZBES   = ZFAC
      DO 100 I = 1, KTERM-1
        ZFAC   = ZFAC * ZZ / ( I * ( NABS + I ) )
        ZBES   = ZBES + ZFAC
  100 CONTINUE
      BESSEL = SGN**NABS * ( 0.5 * PX )**NABS * ZBES
      return
      end function BESSEL



!     *********************************************************************
      function BESSLP( N, PX, KTERM )
!     *********************************************************************
!
!   THIS ROUTINE EVALUTES THE FIRST KTERM TERMS
!   FROM THE SMALL ARGUMENT EXPENSION OF THE
!   DERIVATIVE OF THE BESSEL function.
      NABS   = ABS( N )
      SGN    = 1
      IF( N .LT. 0 ) SGN = -1
      BESSLP = SGN**NABS * ( NABS * BESSEL(NABS,PX,KTERM) / PX - &
                             BESSEL( NABS+1, PX, KTERM-1 ) )
      return
      end function BESSLP



!     *********************************************************************
      subroutine CALCEI(ARG,RESULT,INT)
!     *********************************************************************
!
! This Fortran 77 packet computes the exponential integrals Ei(x),
!  E1(x), and  exp(-x)*Ei(x)  for real arguments  x  where
!
!           integral (from t=-infinity to t=x) (exp(t)/t),  x > 0,
!  Ei(x) =
!          -integral (from t=-x to t=infinity) (exp(t)/t),  x < 0,
!
!  and where the first integral is a principal value integral.
!  The packet contains three function type subprograms: EI, EONE,
!  and EXPEI;  and one subroutine type subprogram: CALCEI.  The
!  calling statements for the primary entries are
!
!                 Y = EI(X),            where  X .NE. 0,
!
!                 Y = EONE(X),          where  X .GT. 0,
!  and
!                 Y = EXPEI(X),         where  X .NE. 0,
!
!  and where the entry points correspond to the functions Ei(x),
!  E1(x), and exp(-x)*Ei(x), respectively.  The routine CALCEI
!  is intended for internal packet use only, all computations within
!  the packet being concentrated in this routine.  The function
!  subprograms invoke CALCEI with the Fortran statement
!         CALL CALCEI(ARG,RESULT,INT)
!  where the parameter usage is as follows
!
!     Function                  Parameters for CALCEI
!       Call                 ARG             RESULT         INT
!
!      EI(X)              X .NE. 0          Ei(X)            1
!      EONE(X)            X .GT. 0         -Ei(-X)           2
!      EXPEI(X)           X .NE. 0          exp(-X)*Ei(X)    3
!
!  The main computation involves evaluation of rational Chebyshev
!  approximations published in Math. Comp. 22, 641-649 (1968), and
!  Math. Comp. 23, 289-303 (1969) by Cody and Thacher.  This
!  transportable program is patterned after the machine-dependent
!  FUNPACK packet  NATSEI,  but cannot match that version for
!  efficiency or accuracy.  This version uses rational functions
!  that theoretically approximate the exponential integrals to
!  at least 18 significant decimal digits.  The accuracy achieved
!  depends on the arithmetic system, the compiler, the intrinsic
!  functions, and proper selection of the machine-dependent
!  constants.
!
!
!*******************************************************************
!*******************************************************************
!
! Explanation of machine-dependent constants
!
!   beta = radix for the floating-point system.
!   minexp = smallest representable power of beta.
!   maxexp = smallest power of beta that overflows.
!   XBIG = largest argument acceptable to EONE; solution to
!          equation:
!                     exp(-x)/x * (1 + 1/x) = beta ** minexp.
!   XINF = largest positive machine number; approximately
!                     beta ** maxexp
!   XMAX = largest argument acceptable to EI; solution to
!          equation:  exp(x)/x * (1 + 1/x) = beta ** maxexp.
!
!     Approximate values for some important machines are:
!
!                           beta      minexp      maxexp
!
!  CRAY-1        (S.P.)       2       -8193        8191
!  Cyber 180/185
!    under NOS   (S.P.)       2         -975        1070
!  IEEE (IBM/XT,
!    SUN, etc.)  (S.P.)       2        -126         128
!  IEEE (IBM/XT,
!    SUN, etc.)  (D.P.)       2       -1022        1024
!  IBM 3033      (D.P.)      16         -65          63
!  VAX D-Format  (D.P.)       2        -128         127
!  VAX G-Format  (D.P.)       2       -1024        1023
!
!                           XBIG       XINF       XMAX
!
!  CRAY-1        (S.P.)    5670.31  5.45E+2465   5686.21
!  Cyber 180/185
!    under NOS   (S.P.)     669.31  1.26E+322     748.28
!  IEEE (IBM/XT,
!    SUN, etc.)  (S.P.)      82.93  3.40E+38       93.24
!  IEEE (IBM/XT,
!    SUN, etc.)  (D.P.)     701.84  1.79D+308     716.35
!  IBM 3033      (D.P.)     175.05  7.23D+75      179.85
!  VAX D-Format  (D.P.)      84.30  1.70D+38       92.54
!  VAX G-Format  (D.P.)     703.22  8.98D+307     715.66
!
!*******************************************************************
!*******************************************************************
!
! Error returns
!
!  The following table shows the types of error that may be
!  encountered in this routine and the function value supplied
!  in each case.
!
!       Error       Argument         Function values for
!                    Range         EI      EXPEI     EONE
!
!     UNDERFLOW  (-)X .GT. XBIG     0        -         0
!     OVERFLOW      X .GE. XMAX    XINF      -         -
!     ILLEGAL X       X = 0       -XINF    -XINF     XINF
!     ILLEGAL X      X .LT. 0       -        -     USE ABS(X)
!
! Intrinsic functions required are:
!
!     ABS, SQRT, EXP
!
!
!  Author: W. J. Cody
!          Mathematics abd Computer Science Division
!          Argonne National Laboratory
!          Argonne, IL 60439
!
!  Latest modification: September 9, 1988
!
!----------------------------------------------------------------------
      INTEGER I,INT
!D    DOUBLE PRECISION
      REAL*8   &
             A,ARG,B,C,D,EXP40,E,EI,F,FOUR,FOURTY,FRAC,HALF,ONE,P, &
             PLG,PX,P037,P1,P2,Q,QLG,QX,Q1,Q2,R,RESULT,S,SIX,SUMP, &
             SUMQ,T,THREE,TWELVE,TWO,TWO4,W,X,XBIG,XINF,XMAX,XMX0, &
             X0,X01,X02,X11,Y,YSQ,ZERO
      DIMENSION  A(7),B(6),C(9),D(9),E(10),F(10),P(10),Q(10),R(10), &
         S(9),P1(10),Q1(9),P2(10),Q2(9),PLG(4),QLG(4),PX(10),QX(10)
!----------------------------------------------------------------------
!  Mathematical constants
!   EXP40 = exp(40)
!   X0 = zero of Ei
!   X01/X11 + X02 = zero of Ei to extra precision
!----------------------------------------------------------------------
      DATA ZERO,P037,HALF,ONE,TWO/0.0E0,0.037E0,0.5E0,1.0E0,2.0E0/, &
           THREE,FOUR,SIX,TWELVE,TWO4/3.0E0,4.0E0,6.0E0,12.E0,24.0E0/, &
           FOURTY,EXP40/40.0E0,2.3538526683701998541E17/, &
           X01,X11,X02/381.5E0,1024.0E0,-5.1182968633365538008E-5/, &
           X0/3.7250741078136663466E-1/
!D    DATA ZERO,P037,HALF,ONE,TWO/0.0D0,0.037D0,0.5D0,1.0D0,2.0D0/,
!D   1     THREE,FOUR,SIX,TWELVE,TWO4/3.0D0,4.0D0,6.0D0,12.D0,24.0D0/,
!D   2     FOURTY,EXP40/40.0D0,2.3538526683701998541D17/,
!D   3     X01,X11,X02/381.5D0,1024.0D0,-5.1182968633365538008D-5/,
!D   4     X0/3.7250741078136663466D-1/
!----------------------------------------------------------------------
! Machine-dependent constants
!----------------------------------------------------------------------
!!!      DATA XINF/3.40E+38/,XMAX/93.246E0/,XBIG/82.93E0/
      DATA XINF/3.40E+38/,XMAX/700.0E0/,XBIG/82.93E0/
!D    DATA XINF/1.79D+308/,XMAX/716.351D0/,XBIG/701.84D0/
!----------------------------------------------------------------------
! Coefficients  for -1.0 <= X < 0.0
!----------------------------------------------------------------------
      DATA A/1.1669552669734461083368E2, 2.1500672908092918123209E3, &
             1.5924175980637303639884E4, 8.9904972007457256553251E4, &
             1.5026059476436982420737E5,-1.4815102102575750838086E5, &
             5.0196785185439843791020E0/
      DATA B/4.0205465640027706061433E1, 7.5043163907103936624165E2, &
             8.1258035174768735759855E3, 5.2440529172056355429883E4, &
             1.8434070063353677359298E5, 2.5666493484897117319268E5/
!D    DATA A/1.1669552669734461083368D2, 2.1500672908092918123209D3,
!D   1       1.5924175980637303639884D4, 8.9904972007457256553251D4,
!D   2       1.5026059476436982420737D5,-1.4815102102575750838086D5,
!D   3       5.0196785185439843791020D0/
!D    DATA B/4.0205465640027706061433D1, 7.5043163907103936624165D2,
!D   1       8.1258035174768735759855D3, 5.2440529172056355429883D4,
!D   2       1.8434070063353677359298D5, 2.5666493484897117319268D5/
!----------------------------------------------------------------------
! Coefficients for -4.0 <= X < -1.0
!----------------------------------------------------------------------
      DATA C/3.828573121022477169108E-1, 1.107326627786831743809E+1, &
             7.246689782858597021199E+1, 1.700632978311516129328E+2, &
             1.698106763764238382705E+2, 7.633628843705946890896E+1, &
             1.487967702840464066613E+1, 9.999989642347613068437E-1, &
             1.737331760720576030932E-8/
      DATA D/8.258160008564488034698E-2, 4.344836335509282083360E+0, &
             4.662179610356861756812E+1, 1.775728186717289799677E+2, &
             2.953136335677908517423E+2, 2.342573504717625153053E+2, &
             9.021658450529372642314E+1, 1.587964570758947927903E+1, &
             1.000000000000000000000E+0/
!D    DATA C/3.828573121022477169108D-1, 1.107326627786831743809D+1,
!D   1       7.246689782858597021199D+1, 1.700632978311516129328D+2,
!D   2       1.698106763764238382705D+2, 7.633628843705946890896D+1,
!D   3       1.487967702840464066613D+1, 9.999989642347613068437D-1,
!D   4       1.737331760720576030932D-8/
!D    DATA D/8.258160008564488034698D-2, 4.344836335509282083360D+0,
!D   1       4.662179610356861756812D+1, 1.775728186717289799677D+2,
!D   2       2.953136335677908517423D+2, 2.342573504717625153053D+2,
!D   3       9.021658450529372642314D+1, 1.587964570758947927903D+1,
!D   4       1.000000000000000000000D+0/
!----------------------------------------------------------------------
! Coefficients for X < -4.0
!----------------------------------------------------------------------
      DATA E/1.3276881505637444622987E+2,3.5846198743996904308695E+4, &
             1.7283375773777593926828E+5,2.6181454937205639647381E+5, &
             1.7503273087497081314708E+5,5.9346841538837119172356E+4, &
             1.0816852399095915622498E+4,1.0611777263550331766871E03, &
             5.2199632588522572481039E+1,9.9999999999999999087819E-1/
      DATA F/3.9147856245556345627078E+4,2.5989762083608489777411E+5, &
             5.5903756210022864003380E+5,5.4616842050691155735758E+5, &
             2.7858134710520842139357E+5,7.9231787945279043698718E+4, &
             1.2842808586627297365998E+4,1.1635769915320848035459E+3, &
             5.4199632588522559414924E+1,1.0E0/
!D    DATA E/1.3276881505637444622987D+2,3.5846198743996904308695D+4,
!D   1       1.7283375773777593926828D+5,2.6181454937205639647381D+5,
!D   2       1.7503273087497081314708D+5,5.9346841538837119172356D+4,
!D   3       1.0816852399095915622498D+4,1.0611777263550331766871D03,
!D   4       5.2199632588522572481039D+1,9.9999999999999999087819D-1/
!D    DATA F/3.9147856245556345627078D+4,2.5989762083608489777411D+5,
!D   1       5.5903756210022864003380D+5,5.4616842050691155735758D+5,
!D   2       2.7858134710520842139357D+5,7.9231787945279043698718D+4,
!D   3       1.2842808586627297365998D+4,1.1635769915320848035459D+3,
!D   4       5.4199632588522559414924D+1,1.0D0/
!----------------------------------------------------------------------
!  Coefficients for rational approximation to ln(x/a), |1-x/a| < .1
!----------------------------------------------------------------------
      DATA PLG/-2.4562334077563243311E+01,2.3642701335621505212E+02, &
               -5.4989956895857911039E+02,3.5687548468071500413E+02/
      DATA QLG/-3.5553900764052419184E+01,1.9400230218539473193E+02, &
               -3.3442903192607538956E+02,1.7843774234035750207E+02/
!D    DATA PLG/-2.4562334077563243311D+01,2.3642701335621505212D+02,
!D    1         -5.4989956895857911039D+02,3.5687548468071500413D+02/
!D    DATA QLG/-3.5553900764052419184D+01,1.9400230218539473193D+02,
!D   1         -3.3442903192607538956D+02,1.7843774234035750207D+02/
!----------------------------------------------------------------------
! Coefficients for  0.0 < X < 6.0,
!  ratio of Chebyshev polynomials
!----------------------------------------------------------------------
      DATA P/-1.2963702602474830028590E01,-1.2831220659262000678155E03, &
             -1.4287072500197005777376E04,-1.4299841572091610380064E06, &
             -3.1398660864247265862050E05,-3.5377809694431133484800E08, &
              3.1984354235237738511048E08,-2.5301823984599019348858E10, &
              1.2177698136199594677580E10,-2.0829040666802497120940E11/
      DATA Q/ 7.6886718750000000000000E01,-5.5648470543369082846819E03, &
              1.9418469440759880361415E05,-4.2648434812177161405483E06, &
              6.4698830956576428587653E07,-7.0108568774215954065376E08, &
              5.4229617984472955011862E09,-2.8986272696554495342658E10, &
              9.8900934262481749439886E10,-8.9673749185755048616855E10/
!D    DATA P/-1.2963702602474830028590D01,-1.2831220659262000678155D03,
!D   1       -1.4287072500197005777376D04,-1.4299841572091610380064D06,
!D   2       -3.1398660864247265862050D05,-3.5377809694431133484800D08,
!D   3        3.1984354235237738511048D08,-2.5301823984599019348858D10,
!D   4        1.2177698136199594677580D10,-2.0829040666802497120940D11/
!D    DATA Q/ 7.6886718750000000000000D01,-5.5648470543369082846819D03,
!D   1        1.9418469440759880361415D05,-4.2648434812177161405483D06,
!D   2        6.4698830956576428587653D07,-7.0108568774215954065376D08,
!D   3        5.4229617984472955011862D09,-2.8986272696554495342658D10,
!D   4        9.8900934262481749439886D10,-8.9673749185755048616855D10/
!----------------------------------------------------------------------
! J-fraction coefficients for 6.0 <= X < 12.0
!----------------------------------------------------------------------
      DATA R/-2.645677793077147237806E00,-2.378372882815725244124E00, &
             -2.421106956980653511550E01, 1.052976392459015155422E01, &
              1.945603779539281810439E01,-3.015761863840593359165E01, &
              1.120011024227297451523E01,-3.988850730390541057912E00, &
              9.565134591978630774217E00, 9.981193787537396413219E-1/
      DATA S/ 1.598517957704779356479E-4, 4.644185932583286942650E00, &
              3.697412299772985940785E02,-8.791401054875438925029E00, &
              7.608194509086645763123E02, 2.852397548119248700147E01, &
              4.731097187816050252967E02,-2.369210235636181001661E02, &
              1.249884822712447891440E00/
!D    DATA R/-2.645677793077147237806D00,-2.378372882815725244124D00,
!D   1       -2.421106956980653511550D01, 1.052976392459015155422D01,
!D   2        1.945603779539281810439D01,-3.015761863840593359165D01,
!D   3        1.120011024227297451523D01,-3.988850730390541057912D00,
!D   4        9.565134591978630774217D00, 9.981193787537396413219D-1/
!D    DATA S/ 1.598517957704779356479D-4, 4.644185932583286942650D00,
!D   1        3.697412299772985940785D02,-8.791401054875438925029D00,
!D   2        7.608194509086645763123D02, 2.852397548119248700147D01,
!D   3        4.731097187816050252967D02,-2.369210235636181001661D02,
!D   4        1.249884822712447891440D00/
!----------------------------------------------------------------------
! J-fraction coefficients for 12.0 <= X < 24.0
!----------------------------------------------------------------------
      DATA P1/-1.647721172463463140042E00,-1.860092121726437582253E01, &
              -1.000641913989284829961E01,-2.105740799548040450394E01, &
              -9.134835699998742552432E-1,-3.323612579343962284333E01, &
               2.495487730402059440626E01, 2.652575818452799819855E01, &
              -1.845086232391278674524E00, 9.999933106160568739091E-1/
      DATA Q1/ 9.792403599217290296840E01, 6.403800405352415551324E01, &
               5.994932325667407355255E01, 2.538819315630708031713E02, &
               4.429413178337928401161E01, 1.192832423968601006985E03, &
               1.991004470817742470726E02,-1.093556195391091143924E01, &
               1.001533852045342697818E00/
!D    DATA P1/-1.647721172463463140042D00,-1.860092121726437582253D01,
!D   1        -1.000641913989284829961D01,-2.105740799548040450394D01,
!D   2        -9.134835699998742552432D-1,-3.323612579343962284333D01,
!D   3         2.495487730402059440626D01, 2.652575818452799819855D01,
!D   4        -1.845086232391278674524D00, 9.999933106160568739091D-1/
!D    DATA Q1/ 9.792403599217290296840D01, 6.403800405352415551324D01,
!D   1         5.994932325667407355255D01, 2.538819315630708031713D02,
!D   2         4.429413178337928401161D01, 1.192832423968601006985D03,
!D   3         1.991004470817742470726D02,-1.093556195391091143924D01,
!D   4         1.001533852045342697818D00/
!----------------------------------------------------------------------
! J-fraction coefficients for  X .GE. 24.0
!----------------------------------------------------------------------
      DATA P2/ 1.75338801265465972390E02,-2.23127670777632409550E02, &
              -1.81949664929868906455E01,-2.79798528624305389340E01, &
              -7.63147701620253630855E00,-1.52856623636929636839E01, &
              -7.06810977895029358836E00,-5.00006640413131002475E00, &
              -3.00000000320981265753E00, 1.00000000000000485503E00/
      DATA Q2/ 3.97845977167414720840E04, 3.97277109100414518365E00, &
               1.37790390235747998793E02, 1.17179220502086455287E02, &
               7.04831847180424675988E01,-1.20187763547154743238E01, &
              -7.99243595776339741065E00,-2.99999894040324959612E00, &
               1.99999999999048104167E00/
!D    DATA P2/ 1.75338801265465972390D02,-2.23127670777632409550D02,
!D   1        -1.81949664929868906455D01,-2.79798528624305389340D01,
!D   2        -7.63147701620253630855D00,-1.52856623636929636839D01,
!D   3        -7.06810977895029358836D00,-5.00006640413131002475D00,
!D   4        -3.00000000320981265753D00, 1.00000000000000485503D00/
!D    DATA Q2/ 3.97845977167414720840D04, 3.97277109100414518365D00,
!D   1         1.37790390235747998793D02, 1.17179220502086455287D02,
!D   2         7.04831847180424675988D01,-1.20187763547154743238D01,
!D   3        -7.99243595776339741065D00,-2.99999894040324959612D00,
!D   4         1.99999999999048104167D00/
!----------------------------------------------------------------------
      X = ARG
      IF (X .EQ. ZERO) THEN
            EI = -XINF
            IF (INT .EQ. 2) EI = -EI
         ELSE IF ((X .LT. ZERO) .OR. (INT .EQ. 2)) THEN
!----------------------------------------------------------------------
! Calculate EI for negative argument or for E1.
!----------------------------------------------------------------------
            Y = ABS(X)
            IF (Y .LE. ONE) THEN
                  SUMP = A(7) * Y + A(1)
                  SUMQ = Y + B(1)
                  DO 110 I = 2, 6
                     SUMP = SUMP * Y + A(I)
                     SUMQ = SUMQ * Y + B(I)
  110             CONTINUE
                  EI = LOG(Y) - SUMP / SUMQ
                  IF (INT .EQ. 3) EI = EI * EXP(Y)
               ELSE IF (Y .LE. FOUR) THEN
                  W = ONE / Y
                  SUMP = C(1)
                  SUMQ = D(1)
                  DO 130 I = 2, 9
                     SUMP = SUMP * W + C(I)
                     SUMQ = SUMQ * W + D(I)
  130             CONTINUE
                  EI = - SUMP / SUMQ
                  IF (INT .NE. 3) EI = EI * EXP(-Y)
               ELSE
                  IF ((Y .GT. XBIG) .AND. (INT .LT. 3)) THEN
                        EI = ZERO
                     ELSE
                        W = ONE / Y
                        SUMP = E(1)
                        SUMQ = F(1)
                        DO 150 I = 2, 10
                           SUMP = SUMP * W + E(I)
                           SUMQ = SUMQ * W + F(I)
  150                   CONTINUE
                         EI = -W * (ONE - W * SUMP / SUMQ )
                        IF (INT .NE. 3) EI = EI * EXP(-Y)
                  END IF
            END IF
            IF (INT .EQ. 2) EI = -EI
         ELSE IF (X .LT. SIX) THEN
!----------------------------------------------------------------------
!  To improve conditioning, rational approximations are expressed
!    in terms of Chebyshev polynomials for 0 <= X < 6, and in
!    continued fraction form for larger X.
!----------------------------------------------------------------------
            T = X + X
            T = T / THREE - TWO
            PX(1) = ZERO
            QX(1) = ZERO
            PX(2) = P(1)
            QX(2) = Q(1)
            DO 210 I = 2, 9
               PX(I+1) = T * PX(I) - PX(I-1) + P(I)
               QX(I+1) = T * QX(I) - QX(I-1) + Q(I)
  210       CONTINUE
            SUMP = HALF * T * PX(10) - PX(9) + P(10)
            SUMQ = HALF * T * QX(10) - QX(9) + Q(10)
            FRAC = SUMP / SUMQ
            XMX0 = (X - X01/X11) - X02
            IF (ABS(XMX0) .GE. P037) THEN
                  EI = LOG(X/X0) + XMX0 * FRAC
                  IF (INT .EQ. 3) EI = EXP(-X) * EI
               ELSE
!----------------------------------------------------------------------
! Special approximation to  ln(X/X0)  for X close to X0
!----------------------------------------------------------------------
                  Y = XMX0 / (X + X0)
                  YSQ = Y*Y
                  SUMP = PLG(1)
                  SUMQ = YSQ + QLG(1)
                  DO 220 I = 2, 4
                     SUMP = SUMP*YSQ + PLG(I)
                     SUMQ = SUMQ*YSQ + QLG(I)
  220             CONTINUE
                  EI = (SUMP / (SUMQ*(X+X0)) + FRAC) * XMX0
                  IF (INT .EQ. 3) EI = EXP(-X) * EI
            END IF
         ELSE IF (X .LT. TWELVE) THEN
            FRAC = ZERO
            DO 230 I = 1, 9
               FRAC = S(I) / (R(I) + X + FRAC)
  230       CONTINUE
            EI = (R(10) + FRAC) / X
            IF (INT .NE. 3) EI = EI * EXP(X)
         ELSE IF (X .LE. TWO4) THEN
            FRAC = ZERO
            DO 240 I = 1, 9
               FRAC = Q1(I) / (P1(I) + X + FRAC)
  240       CONTINUE
            EI = (P1(10) + FRAC) / X
            IF (INT .NE. 3) EI = EI * EXP(X)
         ELSE
            IF ((X .GE. XMAX) .AND. (INT .LT. 3)) THEN
                  EI = XINF
               ELSE
                  Y = ONE / X
                  FRAC = ZERO
                  DO 250 I = 1, 9
                     FRAC = Q2(I) / (P2(I) + X + FRAC)
  250             CONTINUE
                  FRAC = P2(10) + FRAC
                  EI = Y + Y * Y * FRAC
                  IF (INT .NE. 3) THEN
                        IF (X .LE. XMAX-TWO4) THEN
                              EI = EI * EXP(X)
                           ELSE
!----------------------------------------------------------------------
! Calculation reformulated to avoid premature overflow
!----------------------------------------------------------------------
                              EI = (EI * EXP(X-FOURTY)) * EXP40
                        END IF
                  END IF
            END IF
      END IF
      RESULT = EI
      return
      end subroutine CALCEI



!     *********************************************************************
      function EI(X)
!     *********************************************************************
!--------------------------------------------------------------------
!
! This function program computes approximate values for the
!   exponential integral  Ei(x), where  x  is real.
!
!  Author: W. J. Cody
!
!  Latest modification: January 12, 1988
!
!--------------------------------------------------------------------
      INTEGER INT
      REAL*8  EI, X, RESULT
!D    DOUBLE PRECISION  EI, X, RESULT
!--------------------------------------------------------------------
      INT = 1
      CALL CALCEI(X,RESULT,INT)
      EI = RESULT
      return
      end function EI



!     *********************************************************************
      function EXPEI(X)
!     *********************************************************************
!--------------------------------------------------------------------
!
! This func tion program computes approximate values for the
!   function  exp(-x) * Ei(x), where  Ei(x)  is the exponential
!   integral, and  x  is real.
!
!  Author: W. J. Cody
!
!  Latest modification: January 12, 1988
!
!--------------------------------------------------------------------
      INTEGER INT
      REAL*8  EXPEI, X, RESULT
!D    DOUBLE PRECISION  EXPEI, X, RESULT
!--------------------------------------------------------------------
      INT = 3
      CALL CALCEI(X,RESULT,INT)
      EXPEI = RESULT
      return
      end function EXPEI



!     *********************************************************************
      function EONE(X)
!     *********************************************************************
!--------------------------------------------------------------------
!
! This function program computes approximate values for the
!   exponential integral E1(x), where  x  is real.
!
!  Author: W. J. Cody
!
!  Latest modification: January 12, 1988
!
!--------------------------------------------------------------------
      INTEGER INT
      REAL*8  EONE, X, RESULT
!D    DOUBLE PRECISION  EONE, X, RESULT
!--------------------------------------------------------------------
      INT = 2
      CALL CALCEI(X,RESULT,INT)
      EONE = RESULT
      return
      end function EONE



!     *********************************************************************
      function REXP(X)
!     *********************************************************************
      IF(X.GT. 700. .OR. X.LT.-700.) THEN
        REXP=0.0
      ELSE
        REXP=EXP(X)
      ENDIF
      return
      end function REXP
