#!/usr/bin/env python

import math
import argparse
import numpy as np
from rkf45 import r8_rkf45          # necessary: it provides RK integration
from matplotlib.pyplot import *     # for plots:    comment out if plots not needed
from mpl_toolkits.mplot3d import *  # for 3d plots: comment out if plots not needed

"""
	Integration of the planetary model for nanoparticles
        Original code by: Nicola Manini of Universita degli studi di Milano
        Subsequent development by: Michael Plesser
"""


def input_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('-d',   action='store_true',                            help='Enter debug mode')
    parser.add_argument('-r',   action='store_true',                            help='Restart from start_config.dat initial info file')
    parser.add_argument('-g',   action='store',      type=float, default=0.0,   help='Gamma value: damping constant')
    parser.add_argument('-T',   action='store',      type=float, default=3.,    help='Initial kinetic temperature distribution for satellite speed [K]')
    parser.add_argument('-n',   action='store',      type=int  , default=1,     help='Number of satellites to begin with')
    parser.add_argument('-l',   action='store',      type=int  , default=1 ,    help='Number of simulations (loops) to run')
    parser.add_argument('-t',   action='store',      type=float, default=0.01,  help='How many seconds simulate')
    parser.add_argument('--dt', action='store',      type=float, default=1e-5,  help='Time interval for data files')
    parser.add_argument('--np', action='store_true',                            help='Batch mode, run code with "No Plots"')

    args = parser.parse_args()

    return args


def normsquared(vec):
    return np.inner(vec,vec)
    
def derivs(t, y):           # Here is all the physics: the left side of the equation
    global collision
    neq   = len(y)
    nhalf = neq/2
    npart = nhalf/dim
    deriv = np.zeros(neq)     # Just to create & initialize the output array
    deriv[:nhalf] = y[nhalf:] # The second half of y is the velocities:
                            #  they go to the first half of the derivs, to map Newton equation to 1st order
                            
# viscous force (UNREALISTIC, DA RIVEDERE!):
    for i in range(npart):
        inmin=nhalf+i*dim
        inmax=inmin+dim
        deriv[inmin:inmax]=-args.g*np.array(y[inmin:inmax])

    for i in range(npart):
        inmin=i*dim
        inmax=inmin+dim
        ri=y[inmin:inmax]
        inmin+=nhalf
        inmax+=nhalf

        for j in range(i+1,npart):
            jnmin=j*dim
            jnmax=jnmin+dim
            rj=y[jnmin:jnmax]
            jnmin+=nhalf
            jnmax+=nhalf

            rvec=np.subtract(ri,rj)  # the vector joining ri to rj
            r2=normsquared(rvec)
            rmod=math.sqrt(r2)
            r3=r2*rmod
            if collision[2]==0 and rmod < radii[i]+radii[j]:
                collision=t,i,j
                print >> sys.stderr, "collision found between particles {0} and {1} at time {2:.4f}".format(i, j, t)
            force=rvec
            force*=couplingconstant*charges[i]*charges[j]/r3
            deriv[inmin:inmax]+=force/masses[i]
            deriv[jnmin:jnmax]+=-force/masses[j]

    return deriv

def cmprops(y):
    nhalf=len(y)/2
    npart = nhalf/dim
    xcm=[0.]*dim
    vcm=[0.]*dim
    for i in range(npart):
        pp=np.multiply(masses[i],y[dim*i:dim*i+dim])
        xcm=np.add(xcm,pp)
    xcm/=np.sum(masses)
    for i in range(npart):
        pp=np.multiply(masses[i],y[nhalf+dim*i:nhalf+dim*i+dim])
        vcm=np.add(vcm,pp)
    vcm/=np.sum(masses)
    return xcm,vcm


def energies(t, y): # used for checking conservations:
    nhalf = len(y)/2  # offset to reach the velocities
    npart = nhalf/dim

    totkinen=0.
    for i in range(npart):
        inmin=nhalf+i*dim
        inmax=inmin+dim
        vi=y[inmin:inmax]
        totkinen+=masses[i]*normsquared(vi)
    totkinen*=0.5

    totpoten=0.
    for i in range(npart):
        inmin=i*dim
        inmax=inmin+dim
        ri=y[inmin:inmax]        
        for j in range(i+1,npart):
            jnmin=j*dim
            jnmax=jnmin+dim
            rj=y[jnmin:jnmax]
            rvec=np.subtract(ri,rj)  # the vector joining ri to rj
            r1=np.linalg.norm(rvec)
            totpoten+=couplingconstant*charges[i]*charges[j]/r1

    toten=totkinen+totpoten
            
    return toten,totkinen,totpoten


def kinen(i, y): # the single-particle kinetic energy
    nhalf=len(y)/2   # offset to reach the velocities

    inmin=nhalf+i*dim
    inmax=inmin+dim
    vi=y[inmin:inmax]
    return 0.5*masses[i]*normsquared(vi)


def poten(i, y): # evaluate the electric potential energy of particle i
                 # due to the interaction with all other charged particles
    nhalf = len(y)/2
    npart = nhalf/dim
    inmin=i*dim
    inmax=inmin+dim
    ri=y[inmin:inmax]        
    pote=0.
    for j in range(0,i)+range(i+1,npart):
        jnmin=j*dim
        jnmax=jnmin+dim
        rj=y[jnmin:jnmax]
        rvec=np.subtract(ri,rj)  # the vector joining ri to rj
        r1=np.linalg.norm(rvec)
        pote+=couplingconstant*charges[i]*charges[j]/r1
            
    return pote


# here is the actual code which runs an entire simulation:
def wholecalculation(loopnumber):

    global args
    global couplingconstant, charges, masses, radii
    global collision

    npart = args.n+1
    collision=0.,0,0

    label = 'data/nsatellites_{0}__gamma_{1}'.format(args.n, args.g)
    filen = '{0}_loop_number_{1}.xyz'.format(label, loopnumber+1)

    print "# Starting simulation for --- {0}".format(label)

### UNITS ###
## fundamental units:
##       length: micrometer (microm) = 1e-6 m
##       mass:   zepto kg (zkg) = 1e-21 kg
##       time:   microseconds (micros) = 1e-6 s
##       charge: elem. charge q_e, but not really used, the coupling constant
##       couplingconstant = q_e^2/(4 pi epsilon_0) is the only needed
##       quantity relevant for Coulomb interactions
##
## derived units:
##       energy: zJ = zkg*microm^2/micros^2 = 1e-21 J
##       force:  fN = zkg*microm/micros^2 = 1e-15 N
#############

    fpi3                    = 4*np.pi/3             # Volume coefficient V = (4/3*pi)*r^3
    couplingconstant        = 0.2307077055899593    # in microm^3*zkg/micros^2
    kB                      = 1.38064852e-2         # in zJ/K
    amu                     = 1.660539040e-6        # in zkg
    silvermass              = 107.8682*amu
    silvernumberdensity     = 58.5643e9             # atoms microm^-3
    diameterbig             = 0.025                 # in microm
    diametersmall           = 0.0025                # in microm
    initialdistancespread   = 0.15                 # in microm

    ## Random satellite parameterization. Generate a radius then find the mass associated from density
    def masssmall():
        diametersmall = 0
        while diametersmall<=0.001:  diametersmall = np.random.normal(0.0025, 0.002)
        return fpi3*math.pow(diametersmall/2,3)*silvernumberdensity*silvermass

    radii       = [diameterbig/2]+[diametersmall/2]*args.n
    massbig     = fpi3*math.pow(diameterbig  /2,3)*silvernumberdensity*silvermass
    masses      = np.array([massbig]+[masssmall() for i in range(args.n)]) # Using randomized satellite masses

    q0          = 1.                    # Satellite charge magnitude
    charges     = [3*q0]+[-q0]*args.n   # Satellites are NEGATIVELY charged

### End of preliminary setting up stuff ###


    restart_file = 'data/start_config.dat'
    if not args.r:         
        ## Generate initial conditions for the satellites
        x0          = np.zeros(dim*npart)                                               # Initial positions
        v0          = np.zeros(dim*npart)	                                        # Initial velocities                    
        x0[dim:]    = np.random.normal(0.0, initialdistancespread, dim*(npart-1))       # Set satellite initial positions 
        v0[dim:]    = np.random.normal(0.0, kB*args.T/masssmall(), dim*(npart-1))       # Set satellite initial velocities
        

        ## Generate initial conditions for the central particle
        #for i in range(dim):            # Set the speed of the heavy particle
        #    for j in range(1,npart):    #  so that the cm speed equals 0
        #        v0[i]+=v0[dim*j+i]
        #    v0[i]*=-0.0025/massbig ### !!! bad <- fix it!
        y = np.append(x0, v0) # the vector containing all dynamical variables

    elif args.r:                        # Reading from file:
        with open(restart_file, 'r') as f:  y = np.loadtxt(f)
        nsat_file  = (len(y)/2/dim) - 1
        if args.n != nsat_file:
            print "ERROR: expected {0} satellites, got initial condition info for {1}!".format(args.n, nsat_file)
            print "Proceeding with {0} satellites, consider your mistakes!".format(nsat_file)
            args.n = nsat_file
            npart  = nsat_file + 1
    
    neq     = len(y)
    nhalf   = neq/2
    yp      = derivs(0.,y)

    xcm, vcm                    = cmprops(y)
    toten, totkinen, totpoten   = energies(0,y)

    ## These parameters control the precision of the numerical RK integration:
    relerr  = 1.e-11
    abserr  = 1.e-14
    flag    = 1

    tlist           = [0]
    xcmlist         = [xcm]
    vcmlist         = [vcm]
    totenlist       = [toten]
    totkinenlist    = [totkinen]
    totpotenlist    = [totpoten]
    xfull           = [y[:neq/2]]
    kinenv          = np.array([kinen(i, y) for i in range(npart)])
    elpotenv        = np.array([poten(i, y) for i in range(npart)])

    np.set_printoptions(precision=5)
    np.set_printoptions(linewidth=9999999999) # prevent newlines in arrays
    
    if not args.r:  print "## initial condition generated randomly"
    else:           print "## initial condition read from {0}".format(restart_file)
    print "## Parameters:"
    print "#  tot time              : {0} micros".format(args.t)
    print "#  npart                 : {0} (i.e. 1 heavy + {1} satellites)".format(args.n+1, args.n)
    print "#  masses                : {0} zkg".format(masses)
    print "#  charges               : {0} elementary charges".format(charges)
    print "#  initial temperature   : {0} K (kB*T = {1} zJ)".format(args.t, kB*args.T)
    print "#  positions             : {0} microm".format(y[:nhalf])
    print "#  velocities            : {0} microm/micros [= m/s]".format(y[nhalf:])

    np.set_printoptions(precision=6)
    print "## Energy info:"
    print "#  total energy          : {0} zJ".format(toten)
    print "#  total kinetic energy  : {0} zJ".format(totkinen)
    print "#  total potential energy: {0} zJ".format(totpoten)
    print "#  com position          : {0} microm".format(xcm)
    print "#  com velocity          : {0} m/s".format(vcm)
    print "#  kinetic   energies    : {0} zJ".format(kinenv)
    print "#  potential energies    : {0}".format(elpotenv)
    
    fil   = open(filen, 'w')
    nstep = int(round(1.*args.t/args.dt))
    for it in range(nstep):
        print "Simulating time step {0}/{1}".format(it+1, nstep)
        ti = it*args.dt
        tf = (it+1)*args.dt
        tlist.append(tf)
        y, yp, t, flag = r8_rkf45( derivs, neq, y, yp, ti, tf, relerr, abserr, flag )
        if flag!=2:
            print "Warning! flag = {0}.... trying to keep on going".format(flag)
            flag=2


        if collision[2]==0:
            print >> fil, npart
            print >> fil, "#time:",tf
            for i in range(npart):
                if i==0:    name="O"
                else:       name="H"
                print >> fil, name, "".join(" "+str(l) for l in y[dim*i:dim*i+dim])

            xcm, vcm = cmprops(y)
            xcmlist.append(xcm)
            vcmlist.append(vcm)
            xfull.append(y[:neq/2])

            toten, totkinen, totpoten = energies(t,y)
            totenlist.append(toten)
            totkinenlist.append(totkinen)
            totpotenlist.append(totpoten)

        else:
            i=collision[1]
            inmin=i*dim
            inmax=inmin+dim
            ri=np.array(y[inmin:inmax])
            vi=np.array(y[nhalf+inmin:nhalf+inmax])
            j=collision[2]
            jnmin=j*dim
            jnmax=jnmin+dim
            rj=np.array(y[jnmin:jnmax])
            vj=np.array(y[nhalf+jnmin:nhalf+jnmax])

            y=np.delete(y,range(nhalf+jnmin,nhalf+jnmax))
            y=np.delete(y,range(jnmin,jnmax))
            neq=len(y)
            nhalf=neq/2
            npart=nhalf/dim  # should give the same as npart-1
            y[inmin:inmax]=(masses[i]*ri+masses[j]*rj)/(masses[i]+masses[j])
            vi=y[nhalf+inmin:nhalf+inmax]=(masses[i]*vi+masses[j]*vj)/(masses[i]+masses[j])
            masses[i]=masses[i]+masses[j]
            charges[i]=charges[i]+charges[j]
            masses=np.delete(masses,j)
            charges=np.delete(charges,j)
            yp=derivs(collision[0],y)
            restarttime=collision[0],1
            collision=0.,0,0
            it=it-1

            print >>sys.stderr,"Restarting simulation at {0:.4f} with {1} satellites remaining".format(restarttime[0], npart-1)
            plot_tracks(xfull)



    fil.close()
    y = np.array(y)

    kinenv      = np.array([kinen(i, y) for i in range(npart)])
    elpotenv    = np.array([poten(i, y) for i in range(npart)])
    energylists = [totenlist, totkinenlist, totpotenlist] 
    ycmlists    = [vcmlist, xcmlist]

    print "## End of the simulation"
    np.set_printoptions(precision=3)
    print "#  positions             : {0}".format(y[:nhalf])
    print "#  velocities            : {0}".format(y[nhalf:])
    np.set_printoptions(precision=6)
    print "#  energy total          : {0} zJ".format(toten)
    print "#  energy kinetic        : {0} zJ".format(totkinen)
    print "#  energy potential      : {0} zJ".format(totpoten)
    print "#  cm position           : {0} microm".format(xcm)
    print "#  cm velocity           : {0} m/s".format(vcm)
    print "#  ind kinetic energies  : {0} zJ".format(kinenv)
    print "#  ind potential energies: {0} zJ".format(elpotenv)

    fil = open("data/final_config.dat", 'w')
    np.savetxt(fil, y)
    fil.close()

    return tlist, xfull, ycmlists, energylists


def plot_tracks(xfull):
        datapl=[]
        for j in range(len(xfull[-1])):
            d=[]
            for i in range(len(xfull)):
                d.append(xfull[i][j])
            datapl.append(d)
        i=len(xfull)-1
        endpoints=np.array(xfull[i]).reshape(len(xfull[-1])/dim,dim).transpose()
        fig = matplotlib.pyplot.figure()
        ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(xfull[-1])/dim):
            ax.plot(datapl[i*dim],datapl[i*dim+1],datapl[i*dim+2], label='p'+str(i))
        ax.scatter(endpoints[0],endpoints[1],endpoints[2],label="end")
        ax.legend()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        matplotlib.pyplot.show()


def main():

    global dim  # Dimension of the simulation
    global args # Simulation arguments
    global collision

    dim  = 3     
    args = input_args()

    if args.dt<=0 or args.T<=0 or args.n<=0: sys.exit("Hey, something is funny in your options! Goon...")

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    npart = args.n+1
    nhalf = 3*args.n
    for loopn in range(args.l):
        tlist, xfull, ycm, energylists = wholecalculation(loopn)
        vcmlist   = ycm[:nhalf]
        xcmlist   = ycm[nhalf:]
        totenlist = energylists[0]
        kinenlist = energylists[1]
        potenlist = energylists[2]

# This graphics part below here is entirely optional.
# It may be worth commenting it out if you are unwilling to install/use
#    matplotlib
# 3D trajectories:
        if not args.np:
           plot_tracks(xfull) 
            
     #    figure( 1 )
            #subplot( 4, 1, 1 )
            #plot( tlist, xcmlist, 'b-x')
            #ylabel( '$x_{cm}\ [\mu$m]' )
            #title( 'check conservation laws for $'+str(npart-1)+'$ satellites, $\gamma='+str(args.g)+'$')
            #legend( ( "xyz" ), loc='upper left' )

     #    figure( 2 )
            #subplot( 4, 1, 2 )
            #plot( tlist, vcmlist, 'b-o')
            #ylabel( '$v_{cm}$ [m/s]' )
            #legend( ( 'xyz' ), loc='upper right' )
     #    figure( 3 )
            #subplot( 4, 1, 3 )
            #plot( tlist, totenlist, 'b-o')
            #ylabel( 'tot energy [zJ]' )
     #    figure( 4 )
            #subplot( 4, 1, 4 )
            #plot( tlist,totkinenlist, 'r-s', tlist,totpotenlist, 'g-x')
            #xlabel( '$t\ [\mu$s]' )
            #legend( ( 'kin','pot' ), loc='upper right' )
            #ylabel( 'energies [zJ]' )

            #show()



if __name__ == "__main__":
    main()

