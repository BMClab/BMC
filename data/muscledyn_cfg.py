
[activation dynamics]
#maximum value for muscle excitation
u_max = 1.0
#minimum value for muscle excitation
u_min = 0.01

[muscle]
#optimal muscle fiber length
lmopt = 1.0
#pennation angle at rest
alpha0 = 0.0
#maximum isometric muscle force
fm0 = 1.0

[contractile element force-length relationship]
#shape factor
gammal = 0.45

[muscle parallel element]
#exponential shape factor
kpe = 5.0
#passive muscle strain due to maximum isometric force
epsm0 = 0.6

[contractile element force-velocity relationship]
#maximum muscle velocity for concentric activation
vmmax = 10.0
#maximum force generated at the lengthening phase
fcemax = 1.4
#shape factor
Af = 0.25

[tendon]
#tendon slack length
ltslack = 1.0
#tendon strain at the maximal isometric muscle force
epst0 = 0.04
#linear scale factor
kttoe = 3.0


