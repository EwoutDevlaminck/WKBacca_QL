

  subroutine warmdamp(op, oc, Nr, theta, te, imod, imNprw)

    ! ---------------------------------------------------------------------
    ! Wrapper for D. Farina' warmdisp routine.
    ! ---------------------------------------------------------------------

    use ecdisp , only : set_extv, warmdisp, larmornumber

    ! ... implicit none statement ...
    implicit none

    ! ... paramenters ...
    integer, parameter :: imax = 3 ! max harmonics number
    integer, parameter :: fast = 3
    integer, parameter :: r8 = selected_real_kind(15,300)

    ! ... input variables ...
    integer, intent(in)  ::   &
         imod                   ! = 1 for O-mode, = -1 for X-mode
    real(r8), intent(in) ::   &
         op,                  & ! = omega_p^2 / omega^2 
         oc,                  & ! = omega_c^2 / omega^2
         Nr,                  & ! modulus of the refractive index N
         theta,               & ! angle between N and the magnetic field
         te                     ! electron temperature in keV
    
    ! ... output variables ...
    real(r8), intent(out) ::  &
         imNprw                 ! imaginary part of Nprw

    ! ... local variables ...
    integer  ::               &
         err,                 & ! error flag
         nharm,               & ! hamrnic number
         lrm                    ! effective harmonic number
    real(r8) ::               &
         sqoc,                & ! = omega_c / omega
         mu,                  & ! = 511.e0_r8/te
         Nll, Npr               ! parallel and perpendicular N
    complex(r8) ::            &
         Nprw,                & ! Npr obtained from the disp. rel.
         ex, ey, ez             ! polarization unit vector
         
! ... f2py directives ...
!f2py integer intent(in) :: imod    
!f2py real*8 intent(in) :: op, oc, Nr, theta, te
!f2py real*8 intent(out) :: imNprw    

    ! =====================================================================
    ! Executable statements 

    ! ... initialize common variables of the module ecdisp ...
    call set_extv

    ! ... definition of parameters ...
    sqoc = sqrt(oc)
    Nll  = Nr * cos(theta)
    Npr  = sqrt(Nr**2 - Nll**2)
    mu   = 511.e0_r8/te

    ! ... find the harmonic number ...
    nharm = larmornumber(sqoc, Nll, mu)
    lrm = min(imax, nharm)
    
    ! ... estimate the warm-plasma dispersion function ... 
    call warmdisp(op, sqoc, Nll, mu, -imod, fast, lrm, Npr,                &
         Nprw, ex, ey, ez, err)
    
    ! ... extract imaginary part of the refractive index ...
    imNprw = aimag(Nprw)   


    ! ... do not allow negative values, put 0 instead ...
    if (imNprw < 0.) then
       imNprw = 0.
    end if

    return
  end subroutine warmdamp
