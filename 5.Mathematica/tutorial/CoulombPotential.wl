(* ::Package:: *)

BeginPackage["CoulombPotential`"]
Clear[WaveF,WaveR,WaveA];
WaveR::usage="WaveR[Z_,r_,n_,l_]\:8ba1\:7b97\:7535\:5b50\:5728\:5e93\:4ed1\:52bf\:4e2d\:672c\:5f81\:6ce2\:51fd\:6570\:5f84\:5411\:90e8\:5206\:7684\:8868\:793a\:ff1br_\:4ee5\:73bb\:5c14\:534a\:5f84\:4e3a\:5355\:4f4d\:3002"
WaveA::usage="WaveA[theta_,phi_,l_,m_]\:8ba1\:7b97\:7535\:5b50\:5728\:5e93\:4ed1\:52bf\:4e2d\:672c\:5f81\:6ce2\:51fd\:6570\:89d2\:5ea6\:90e8\:5206\:7684\:8868\:793a\:3002"
WaveF::usage="WaveF[Z_,r_,theta_,phi_,n_,l_,m_]\:8ba1\:7b97\:7535\:5b50\:5728\:5e93\:4ed1\:52bf\:4e2d\:672c\:5f81\:6ce2\:51fd\:6570\:7684\:8868\:793a\:3002"

    Begin["`Private`"]
    WaveR[Z_,r_,n_,l_]:=Module[{unit,tmp},
    unit=(1/(2l+1)!)Sqrt[(n+1)!/(2n (n-l-1)!)](((2Z)/n)^(l+3/2));
    tmp=unit r^l Exp[-((Z r)/n)]Hypergeometric1F1[l+1-n,2l+2,(2Z r)/n]]
    WaveA[theta_,phi_,l_,m_]:=Module[{tmp},
    tmp=SphericalHarmonicY[l,m,theta,phi];tmp]
    WaveF[Z_,r_,theta_,phi_,n_,l_,m_]:=Module[{tmp},
    tmp=WaveR[Z,r,n,l] WaveA[theta,phi,l,m];tmp]
    End[]
EndPackage[]
