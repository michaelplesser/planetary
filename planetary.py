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

        ### UNITS ###
        ## fundamental units:
        ##       length: micrometer (microm) = 1e-6 m
        ##       mass            :  zepto kg (zkg) = 1e-21 kg
        ##       time            :  microseconds (micros) = 1e-6 s
        ##       charge          :  elem. charge q_e, but not really used
        ##       couplingconstant:  q_e^2/(4 pi epsilon_0) is the only 
        ##                            quantity relevant for Coulomb interactions
        ##
        ## derived units:
        ##       energy: zJ = zkg*microm^2/micros^2 = 1e-21 J
        ##       force:  fN = zkg*microm/micros^2 = 1e-15 N

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

    ## Just a few misc. checks. You need a positive, non-zero time-step, temperature, and number of satellites, etc
    if args.dt<=0 or args.t<=args.dt or args.T<=0 or args.n<=0: sys.exit("Hey, something is funny in your options! Goon...")

    return args

## Does the physics, differential equations, F=ma, etc, etc
def derivs(t, y):   

    global collision

    neq   = len(y)
    nhalf = neq   // 2
    npart = nhalf // dim
    deriv = np.zeros(neq)       
    deriv[:nhalf] = y[nhalf:]   # The second half of y is the velocities:
                                #  they go to the first half of the derivs, to map Newton equation to 1st order
                            
    ## Viscous force (damping) (UNREALISTIC, DA RIVEDERE!):
    ## (But not too big a deal if gamma (args.g) is 0 or small-ish...)
    for i in range(npart):
        inmin = nhalf + i*dim
        inmax = inmin + dim
        deriv[inmin:inmax] = -args.g * np.array(y[inmin:inmax]) # Diff. eq is dv/dt = -gamma*v, standard drag

    ## Coulomb forces
    for i in range(npart):
        inmin   = i*dim
        inmax   = inmin+dim
        ri      = y[inmin:inmax]
        inmin  += nhalf
        inmax  += nhalf

        for j in range(i+1, npart):
            jnmin   = j * dim
            jnmax   = jnmin + dim
            rj      = y[jnmin:jnmax]
            jnmin  += nhalf
            jnmax  += nhalf

            rvec    = np.subtract(ri,rj)  # the vector joining ri to rj
            r2      = np.inner(rvec, rvec)
            rmod    = math.sqrt(r2)
            r3      = r2 * rmod
            if collision[2]==0 and rmod < radii[i]+radii[j]:
                collision = t,i,j
                print("\n")
                print("Collision found between particles {0} and {1} at time {2:.4f}".format(i, j, t))
            force   = rvec
            force  *= couplingconstant * charges[i] * charges[j] / r3
            deriv[inmin:inmax] +=  force / masses[i]
            deriv[jnmin:jnmax] += -force / masses[j]

    return deriv

## Center of Mass properies, find the center of mass position and velocity 
def cmprops(y):

    nhalf = len(y) // 2
    npart = nhalf  // dim
    xcm   = [0.] * dim
    vcm   = [0.] * dim
    for i in range(npart):
        pp_x  = np.multiply(masses[i], y[        dim*i:        dim*i+dim])
        pp_v  = np.multiply(masses[i], y[nhalf + dim*i:nhalf + dim*i+dim])
        xcm   = np.add(xcm, pp_x)
        vcm   = np.add(vcm, pp_v)
    xcm /= np.sum(masses)
    vcm /= np.sum(masses)

    return xcm,vcm

## Calculate energies of the system
def energies(t, y):         # used for checking conservations:

    nhalf = len(y) // 2     # offset to reach the velocities
    npart = nhalf  // dim

    totkinen = 0.
    totpoten = 0.
    for i in range(npart):
        
        ## Kinetic energy calculation
        k_inmin     = nhalf   + i*dim
        k_inmax     = k_inmin +   dim
        vi          = y[k_inmin:k_inmax]
        totkinen   += masses[i] * np.inner(vi, vi)
        
        ## Potential energy calculation
        p_inmin = i * dim
        p_inmax = p_inmin + dim
        ri      = y[p_inmin:p_inmax]        
        for j in range(i+1,npart):
            p_jnmin   = j * dim
            p_jnmax   = p_jnmin + dim
            rj        = y[p_jnmin:p_jnmax]
            rvec      = np.subtract(ri,rj)  # the vector joining ri to rj
            r1        = np.linalg.norm(rvec)
            totpoten += couplingconstant * charges[i] * charges[j] / r1

    totkinen *= 0.5
    toten     = totkinen + totpoten
            
    return toten, totkinen, totpoten

## Calculate the kinetic energy of a single particle
def kinen(i, y):        
    
    nhalf = len(y) // 2 # Offset to reach the velocities

    inmin = nhalf + i*dim
    inmax = inmin +   dim
    vi    = y[inmin:inmax]

    return 0.5 * masses[i] * np.inner(vi, vi)


def poten(i, y):        # Evaluate the electric potential energy of particle i
                        # due to the interaction with all other charged particles
    nhalf = len(y) // 2
    npart = nhalf  // dim
    inmin = i * dim
    inmax = inmin + dim
    ri    = y[inmin:inmax]        
    pote  = 0.
    for j in range(npart):
        if i == j: continue         # Don't try to calculate self-energy... I ain't no Schwinger!
        jnmin = j * dim
        jnmax = jnmin + dim
        rj    = y[jnmin:jnmax]
        rvec  = np.subtract(ri,rj)  # the vector joining ri to rj
        r1    = np.linalg.norm(rvec)
        pote += couplingconstant * charges[i] * charges[j] / r1
            
    return pote

def simulate_that_ish(loopnumber):

    global args
    global couplingconstant, charges, masses, radii
    global collision

    npart     = args.n + 1
    collision = 0.,0,0

    label = 'data/nsatellites_{0}__gamma_{1}'.format(args.n, args.g)
    filen = '{0}_loop_number_{1}.xyz'.format(label, loopnumber+1)

    print('\n')
    print("### Starting simulation for --- {0}".format(label))

    fpi3                    = 4*np.pi/3             # Volume coefficient V = (4/3*pi)*r^3
    couplingconstant        = 0.2307077055899593    # in microm^3*zkg/micros^2
    kB                      = 1.38064852e-2         # in zJ/K
    amu                     = 1.660539040e-6        # in zkg
    silvermass              = 107.8682*amu
    silvernumberdensity     = 58.5643e9             # atoms microm^-3
    diameterbig             = 0.025                 # in microm
    initialdistancespread   = 0.1                   # in microm

    ## Random satellite parameterization. Generate a radius then find the mass associated from density
    ## Requires a diameter above a threshold, IE 0. But it's good to use something like 0.001 for physicality
    sat_diameters = []
    def masssmall():
        diametersmall = 0
        while diametersmall<=0.001:  diametersmall = np.random.normal(0.0025, 0.002)   
        sat_diameters.append(diametersmall)
        return fpi3 * math.pow(diametersmall/2,3) * silvernumberdensity * silvermass

    massbig     = fpi3 * math.pow(diameterbig/2,3) * silvernumberdensity * silvermass
    masses      = np.array([massbig] + [masssmall() for i       in range(args.n)])      # Using randomized satellite masses
    radii       = [diameterbig/2]    + [d_small/2   for d_small in sat_diameters]

    q0          = 1.                    # Satellite charge magnitude
    charges     = [3*q0] + [-q0]*args.n # Satellites are NEGATIVELY charged

    ## End of preliminary, setting up stuff

    ## Generate initial conditions

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
            print("ERROR: expected {0} satellites, got initial condition info for {1}!".format(args.n, nsat_file))
            print("Proceeding with {0} satellites, consider your mistakes!".format(nsat_file))
            args.n = nsat_file
            npart  = nsat_file + 1
    
    neq     = len(y)
    nhalf   = neq//2
    yp      = derivs(0.,y)

    xcm, vcm                    = cmprops(y)
    toten, totkinen, totpoten   = energies(0,y)

    ## These parameters control the precision of the numerical RK integration:
    relerr  = 1.e-11
    abserr  = 1.e-14
    flag    = 1

    ## Initialize the various data arrays that will be used
    tlist           = [0]
    xcmlist         = [xcm]
    vcmlist         = [vcm]
    totenlist       = [toten]
    totkinenlist    = [totkinen]
    totpotenlist    = [totpoten]
    xfull           = [y[:neq//2]]
    kinenv          = np.array([kinen(i, y) for i in range(npart)])
    elpotenv        = np.array([poten(i, y) for i in range(npart)])

    np.set_printoptions(precision=5)
    np.set_printoptions(linewidth=9999999999) # prevent newlines in arrays
    
    if not args.r:  print("### initial condition generated randomly")
    else:           print("### initial condition read from {0}".format(restart_file))
    print("\n")
    print("## Parameters:")
    print("#  tot time              : {0} micros".format(args.t))
    print("#  npart                 : {0} (i.e. 1 heavy + {1} satellites)".format(args.n+1, args.n))
    print("#  masses                : {0} zkg".format(masses))
    print("#  charges               : {0} elementary charges".format(charges))
    print("#  initial temperature   : {0} K (kB*T = {1} zJ)".format(args.T, kB*args.T))
    print("#  positions             : {0} microm".format(y[:nhalf]))
    print("#  velocities            : {0} microm/micros [= m/s]".format(y[nhalf:]))
    np.set_printoptions(precision=6)
    print("## Energy info:")
    print("#  total energy          : {0} zJ".format(toten))
    print("#  total kinetic energy  : {0} zJ".format(totkinen))
    print("#  total potential energy: {0} zJ".format(totpoten))
    print("#  com position          : {0} microm".format(xcm))
    print("#  com velocity          : {0} m/s".format(vcm))
    print("#  kinetic   energies    : {0} zJ".format(kinenv))
    print("#  potential energies    : {0}".format(elpotenv))
    print("\n")
    
    restarttime = 0., 0     # For a satellite collision, we merge particles and restart at the collision time
                            # Tuple has form <restart_time>,<restart_flag>, Flag==0 means no restart, ==1 means restart!
    fil   = open(filen, 'w')
    nstep = int(round(1.*args.t/args.dt))
    for it in range(nstep):
        tf = (it+1)*args.dt
        if restarttime[1] == 0:
            print("Simulating time step {0}/{1}".format(it+1, nstep), end='\r')
            ti = it*args.dt
            tlist.append(tf)
        else:
            ti=restarttime[0]
            restarttime=0.,0

        y, yp, t, flag = r8_rkf45( derivs, neq, y, yp, ti, tf, relerr, abserr, flag )
        if flag!=2:
            print("Warning! flag = {0}.... trying to keep on going".format(flag))
            flag=2


        if collision[2]==0:
            print(npart, end="", file=fil)
            print("#time: "+str(tf), end="", file=fil)
            for i in range(npart):
                if i==0:    name="O"
                else:       name="H"
                print(name+"".join(" "+str(l) for l in y[dim*i:dim*i+dim]), end="", file=fil)

            xcm, vcm = cmprops(y)
            xcmlist.append(xcm)
            vcmlist.append(vcm)
            xfull.append(y[:neq//2])

            toten, totkinen, totpoten = energies(t,y)
            totenlist.append(toten)
            totkinenlist.append(totkinen)
            totpotenlist.append(totpoten)

        else:
            i     = collision[1]
            j     = collision[2]
            inmin = i*dim
            jnmin = j*dim
            inmax = inmin+dim
            jnmax = jnmin+dim
            ri    = np.array(y[inmin:inmax])
            rj    = np.array(y[jnmin:jnmax])
            vi    = np.array(y[nhalf+inmin:nhalf+inmax])
            vj    = np.array(y[nhalf+jnmin:nhalf+jnmax])

            y     = np.delete(y,range(nhalf+jnmin,nhalf+jnmax))
            y     = np.delete(y,range(jnmin,jnmax))
            
            neq             = len(y)
            nhalf           = neq//2
            npart           = nhalf//dim
            y[inmin:inmax]  = (masses[i]*ri+masses[j]*rj)/(masses[i]+masses[j])
            vi              = y[nhalf+inmin:nhalf+inmax]=(masses[i]*vi+masses[j]*vj)/(masses[i]+masses[j])
            masses[i]       = masses[i]+masses[j]
            charges[i]      = charges[i]+charges[j]
            
            masses          = np.delete(masses,j)
            charges         = np.delete(charges,j)
            
            yp              = derivs(collision[0],y)
            restarttime     = collision[0],1
            collision       = 0., 0, 0
            it              = it-1

            if not args.np: plot_tracks(xfull)
            print("Restarting simulation at {0:.4f} with {1} satellites remaining".format(restarttime[0], npart-1))
            print('\n')

            xfull = [ np.concatenate([x[:jnmin], x[jnmax:]]) for x in xfull]    # Remove previous position data on collided satellite
                                                                                # Otherwise plot_tracks gets confused :(

    fil.close()
    y = np.array(y)

    kinenv      = np.array([kinen(i, y) for i in range(npart)])
    elpotenv    = np.array([poten(i, y) for i in range(npart)])
    energylists = [totenlist, totkinenlist, totpotenlist] 
    ycmlists    = [vcmlist, xcmlist]

    print("\n")
    print("## End of the simulation")
    np.set_printoptions(precision=3)
    print("#  positions             : {0}".format(y[:nhalf]))
    print("#  velocities            : {0}".format(y[nhalf:]))
    np.set_printoptions(precision=6)
    print("#  energy total          : {0} zJ".format(toten))
    print("#  energy kinetic        : {0} zJ".format(totkinen))
    print("#  energy potential      : {0} zJ".format(totpoten))
    print("#  cm position           : {0} microm".format(xcm))
    print("#  cm velocity           : {0} m/s".format(vcm))
    print("#  ind kinetic energies  : {0} zJ".format(kinenv))
    print("#  ind potential energies: {0} zJ".format(elpotenv))
    print("\n")

    fil = open("data/final_config.dat", 'w')
    np.savetxt(fil, y)
    fil.close()

    return tlist, xfull, ycmlists, energylists


## Currently plot_tracks only supports dim==3. Generalization TBD
def plot_tracks(xfull):
        datapl = []
        for j in range(len(xfull[-1])):
            d  = []
            for i in range(len(xfull)):
                d.append(xfull[i][j])
            datapl.append(d)
        i         = len(xfull)-1
        endpoints = np.array(xfull[i]).reshape(len(xfull[-1])//dim,dim).transpose()
        fig       = matplotlib.pyplot.figure()
        ax        = fig.gca(projection='3d')
        ax        = fig.add_subplot(111, projection='3d')
        for i in range(len(xfull[-1])//dim):
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

    np.set_printoptions(precision=5)
    np.set_printoptions(suppress=True)

    npart = args.n+1
    nhalf = args.n*dim
    for loopn in range(args.l):
        tlist, xfull, ycm, energylists = simulate_that_ish(loopn)
        vcmlist   = ycm[:nhalf]
        xcmlist   = ycm[nhalf:]
        totenlist = energylists[0]
        kinenlist = energylists[1]
        potenlist = energylists[2]

        ## This graphics below are entirely optional
        ## Comment it out if you don't want plots / to use matplotlib
        if not args.np:
           plot_tracks(xfull) 

            ## Energy conservation plots, not too interesting except as a conservation check...
            
            ## Figure( 1 )
            #subplot( 4, 1, 1 )
            #plot( tlist, xcmlist, 'b-x')
            #ylabel( '$x_{cm}\ [\mu$m]' )
            #title( 'check conservation laws for $'+str(npart-1)+'$ satellites, $\gamma='+str(args.g)+'$')
            #legend( ( "xyz" ), loc='upper left' )

            ## Figure( 2 )
            #subplot( 4, 1, 2 )
            #plot( tlist, vcmlist, 'b-o')
            #ylabel( '$v_{cm}$ [m/s]' )
            #legend( ( 'xyz' ), loc='upper right' )
            
            ## Figure( 3 )
            #subplot( 4, 1, 3 )
            #plot( tlist, totenlist, 'b-o')
            #ylabel( 'tot energy [zJ]' )
     
            ## Figure( 4 )
            #subplot( 4, 1, 4 )
            #plot( tlist,totkinenlist, 'r-s', tlist,totpotenlist, 'g-x')
            #xlabel( '$t\ [\mu$s]' )
            #legend( ( 'kin','pot' ), loc='upper right' )
            #ylabel( 'energies [zJ]' )

            #show()

if __name__ == "__main__":
    main()

