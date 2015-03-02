"""Muscle modeling and simulation."""

from __future__ import division, print_function
import numpy as np
from scipy.integrate import ode
import warnings
import configparser


__author__ = 'Marcos Duarte, https://github.com/demotu/BMC'
__version__ = 'muscles.py v.1 2015/03/01'


class Thelen2003():
    """ Thelen (2003) muscle model.
    """

    def __init__(self, parameters=None, states=None):
        if parameters is not None:
            self.set_parameters(parameters)
        if states is not None:
            self.set_states(states)

        self.lm_data = []
        self.lm_data2 = []
        self.a_data = []


    def set_parameters(self, var=None):
        """Load and set parameters for the muscle model.
        """
        
        if var is None:
            var = './../data/muscle_parameter.txt'
        if isinstance(var, str):
            self.P = self.config_parser(var, 'parameters')
        elif isinstance(var, dict):
            self.P = var
        else:
            raise ValueError('Wrong parameters!')    
            
        print('The parameters were successfully loaded ' +
              'and are stored in the variable P.')
        

    def set_states(self, var=None):
        """Load and set states for the muscle model.
        """
        
        if var is None:
            var = './../data/muscle_state.txt'
        if isinstance(var, str):
            self.S = self.config_parser(var, 'states')
        elif isinstance(var, dict):
            self.S = var
        else:
            raise ValueError('Wrong states!')    
            
        print('The states were successfully loaded ' +
              'and are stored in the variable S.')
              

    def config_parser(self, filename, var):

        parser = configparser.ConfigParser()
        parser.optionxform = str  # make option names case sensitive
        parser.read(filename)
        if not parser:
            raise ValueError('No %s file found!' %var)
        #if not 'Muscle' in parser.sections()[0]:
        #    raise ValueError('Wrong %s file!' %var)
        var = {}
        for key, value in parser.items(parser.sections()[0]):
            if key.lower() in ['name', 'id']:
                var.update({key: value})
            else:
                try:
                    value = float(value)
                except ValueError:
                    print('"%s" value "%s" was replaced by NaN.' %(key, value))
                    value = np.nan
                var.update({key: value})
        
        return var   
        

    def force_l(self, lm, gammal=None):
        """Thelen (2003) force of the contractile element vs. muscle length.

        Parameters
        ----------
        lm : float
            normalized muscle fiber length
        gammal : float, optional (default from parameter file)
            shape factor

        Returns
        -------
        fl : float
            normalized force of the muscle contractile element
        """

        if gammal is None: gammal = self.P['gammal']

        fl = np.exp(-(lm-1)**2/gammal)
            
        return fl


    def force_pe(self, lm, kpe=None, epsm0=None):
        """Thelen (2003) force of the muscle parallel element vs. muscle length.
        
        Parameters
        ----------
        lm : float
            normalized muscle fiber length
        kpe : float, optional (default from parameter file)
            exponential shape factor
        epsm0 : float, optional (default from parameter file)
            passive muscle strain due to maximum isometric force
    
        Returns
        -------
        fpe : float
            normalized force of the muscle parallel (passive) element
        """
        
        if kpe is None: kpe = self.P['kpe']
        if epsm0 is None: epsm0 = self.P['epsm0']

        if lm <= 1:
            fpe = 0
        else:
            fpe = (np.exp(kpe*(lm-1)/epsm0)-1)/(np.exp(kpe)-1)
        
        return fpe
        
    
    def force_se(self, lt, ltslack=None, epst0=None, kttoe=None):
        """Thelen (2003) force-length relationship of tendon vs. tendon length.
        
        Parameters
        ----------
        lt : float
            tendon length (normalized or not)
        ltslack : float, optional (default from parameter file)
            tendon slack length (normalized or not)
        epst0 : float, optional (default from parameter file)
            tendon strain at the maximal isometric muscle force
        kttoe : float, optional (default from parameter file)
            linear scale factor
    
        Returns
        -------
        fse : float
            normalized force of the tendon series element
        """
    
        if ltslack is None: ltslack = self.P['ltslack']
        if epst0 is None: epst0 = self.P['epst0']
        if kttoe is None: kttoe = self.P['kttoe']

        epst = (lt-ltslack)/ltslack
        fttoe = .33
        # values from OpenSim Thelen2003Muscle
        epsttoe =  .99*epst0*np.e**3/(1.66*np.e**3 - .67)
        ktlin =  .67/(epst0 - epsttoe)
        #
        if epst <= 0:
            fse = 0
        elif epst <= epsttoe:
            fse = fttoe/(np.exp(kttoe)-1)*(np.exp(kttoe*epst/epsttoe)-1)
        else:
            fse = ktlin*(epst-epsttoe) + fttoe
            
        return fse
        
        
    def velo_fm(self, fm, a, fl, lmopt=None, vmmax=None, fmlen=None, af=None):
        """Thelen (2003) velocity of the force-velocity relationship vs. CE force.
        
        Parameters
        ----------
        fm : float
            normalized muscle force
        a : float
            muscle activation level
        fl : float
            normalized muscle force due to the force-length relationship
        lmopt : float, optional (default from parameter file)
            optimal muscle fiber length
        vmmax : float, optional (default from parameter file)
            normalized maximum muscle velocity for concentric activation
        fmlen : float, optional (default from parameter file)
            normalized maximum force generated at the lengthening phase
        af : float, optional (default from parameter file)
            shape factor
    
        Returns
        -------
        vm : float
            velocity of the muscle
        """

        if lmopt is None: lmopt = self.P['lmopt']
        if vmmax is None: vmmax = self.P['vmmax']
        if fmlen is None: fmlen = self.P['fmlen']
        if af is None: af = self.P['af']

        if fm <= a*fl:  # isometric and concentric activation
            if fm > 0:
                b = a*fl + fm/af
            else:
                b = a*fl
        else:           # eccentric activation
            asyE_thresh = 0.95  # from OpenSim Thelen2003Muscle
            if fm < a*fl*fmlen*asyE_thresh:
                b = (2 + 2/af)*(a*fl*fmlen - fm)/(fmlen - 1)
            else:
                fm0 = a*fl*fmlen*asyE_thresh
                b = (2 + 2/af)*(a*fl*fmlen - fm0)/(fmlen - 1)

        vm = (0.25  + 0.75*a)*1*(fm - a*fl)/b
        vm = vm*vmmax*lmopt

        return vm
        
        
    def force_vm(self, vm, a, fl, lmopt=None, vmmax=None, fmlen=None, af=None):
        """Thelen (2003) force of the contractile element vs. muscle velocity.
        
        Parameters
        ----------
        vm : float
            muscle velocity
        a : float
            muscle activation level
        fl : float
            normalized muscle force due to the force-length relationship
        lmopt : float, optional (default from parameter file)
            optimal muscle fiber length
        vmmax : float, optional (default from parameter file)
            normalized maximum muscle velocity for concentric activation
        fmlen : float, optional (default from parameter file)
            normalized normalized maximum force generated at the lengthening phase
        af : float, optional (default from parameter file)
            shape factor
    
        Returns
        -------
        fvm : float
            normalized force of the muscle contractile element
        """

        if lmopt is None: lmopt = self.P['lmopt']
        if vmmax is None: vmmax = self.P['vmmax']
        if fmlen is None: fmlen = self.P['fmlen']
        if af is None: af = self.P['af']

        vmmax = vmmax*lmopt
        if vm <= 0:  # isometric and concentric activation
            fvm = af*a*fl*(4*vm + vmmax*(3*a + 1))/(-4*vm + vmmax*af*(3*a + 1))
        else:        # eccentric activation
            fvm = a*fl*(af*vmmax*(3*a*fmlen - 3*a + fmlen - 1) + \
                  8*vm*fmlen*(af + 1)) / \
                  (af*vmmax*(3*a*fmlen - 3*a + fmlen - 1) + 8*vm*(af + 1))          
        
        return fvm
        
        
    def penn_ang(self, lmt, lm, lt=None, lm0=None, alpha0=None):
        """Pennation angle.
        
        Parameters
        ----------
        lmt : float
            muscle-tendon length
        lt : float, optional (default=None)
            tendon length
        lm : float, optional (default=None)
            muscle fiber length
        lm0 : float, optional (default from states file)
            initial muscle fiber length
        alpha0 : float, optional (default from parameter file)
            initial pennation angle

        Returns
        -------
        alpha : float
            pennation angle
        """

        if lm0 is None: lm0 = self.S['lm0']
        if alpha0 is None: alpha0 = self.P['alpha0']

        alpha = alpha0
        if alpha0 != 0:
            w = lm0*np.sin(alpha0)
            if lm is not None:
                cosalpha = np.sqrt(1-(w/lm)**2)
            elif lmt is not None and lt is not None:
                cosalpha = 1/(np.sqrt(1 + (w/(lmt-lt))**2))
            alpha = np.arccos(cosalpha)
            
        if alpha > 1.4706289:  # np.arccos(0.1), 84.2608 degrees
            alpha = 1.4706289
        
        return alpha
        

    def excitation(self, t, u_max=None, u_min=None, t0=0, t1=5):
        """Excitation signal, a square wave.

        Parameters
        ----------
        t : float
            time instant [s]
        u_max : float (0 < u_max <= 1), optional (default from parameter file)
            maximum value for muscle excitation
        u_min : float (0 < u_min < 1), optional (default from parameter file)
            minimum value for muscle excitation
        t0 : float, optional (default=0)
            initial time instant for muscle excitation equals to u_max [s]
        t1 : float, optional (default=5)
            final time instant for muscle excitation equals to u_max [s]

        Returns
        -------
        u : float (0 < u <= 1)
            excitation signal
        """

        if u_max is None: u_max = self.P['u_max']
        if u_min is None: u_min = self.P['u_min']
        
        u = u_min
        if t >= t0 and t <= t1:
            u = u_max

        return u


    def activation_dyn(self, t, a, t_act=None, t_deact=None):
        """Thelen (2003) activation dynamics, the derivative of `a` at `t`.

        Parameters
        ----------
        t : float
            time instant [s]
        a : float (0 <= a <= 1)
            muscle activation
        t_act : float, optional (default from parameter file)
            activation time constant [s]
        t_deact : float, optional (default from parameter file)
            deactivation time constant [s]
    
        Returns
        -------
        adot : float 
            derivative of `a` at `t`
        """

        if t_act is None: t_act = self.P['t_act']
        if t_deact is None: t_deact = self.P['t_deact']
        
        u = self.excitation(t)
        if u > a:
            adot = (u - a)/(t_act*(0.5 + 1.5*a))
        else:
            adot = (u - a)/(t_deact/(0.5 + 1.5*a))

        return adot


    def activation_sol(self, fun=None, t0=0, t1=3, a0=0, u_min=None,
                       t_act=None, t_deact=None, show=False, axs=None):
        """Runge-Kutta (4)5 ODE solver for activation dynamics.

        Parameters
        ----------
        fun : function object, optional (default is None and `actdyn` is used)
            function with ODE to be solved
        t0 : float, optional (default=0)
            initial time instant for the simulation [s]
        t1 : float, optional (default=0)
            final time instant for the simulation [s]
        a0 : float, optional (default=0)
            initial muscle activation
        u_max : float (0 < u_max <= 1), optional (default from parameter file)
            maximum value for muscle excitation
        u_min : float (0 < u_min < 1), optional (default from parameter file)
            minimum value for muscle excitation
        t_act : float, optional (default from parameter file)
            activation time constant [s]
        t_deact : float, optional (default from parameter file)
            deactivation time constant [s]
        show : bool, optional (default = False)
            if True (1), plot data in matplotlib figure
        axs : a matplotlib.axes.Axes instance, optional (default = None)

        Returns
        -------
        data : 2-d array
            array with columns [time, excitation, activation]
            
        """

        if u_min is None: u_min = self.P['u_min']
        if t_act is None: t_act = self.P['t_act']
        if t_deact is None: t_deact = self.P['t_deact']
      
        if fun is None:
            fun = self.activation_dyn
        f = ode(fun).set_integrator('dopri5', nsteps=1, max_step=0.005, atol=1e-8)  
        f.set_initial_value(a0, t0).set_f_params(t_act, t_deact)
        # suppress Fortran warning
        warnings.filterwarnings("ignore", category=UserWarning)
        data = []
        while f.t < t1:
            f.integrate(t1, step=True)
            data.append([f.t, self.excitation(f.t), np.max([f.y, u_min])])
        warnings.resetwarnings()
        data = np.array(data)
        if show:
            self.actvation_plot(data, axs)

        self.a_data = data
        
        return data


    def activation(self, t=None):
        """Activation signal."""
        
        data = self.a_data        
        if t is not None and len(data):
            if t <= self.a_data[0, 0]:
                a = self.a_data[0, 2]
            elif t >= self.a_data[-1, 0]:
                a = self.a_data[-1, 2]
            else:
                a = np.interp(t, self.a_data[:, 0], self.a_data[:, 2])
        else:
            a = 1
            
        return a

        
    def lmt_eq(self, t, lmt0=None):
        """Equation for muscle-tendon length."""

        if lmt0 is None:
            lmt0 = self.S['lmt0']
            
        return lmt0

        
    def vm_eq(self, t, lm, lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0):
        """Equation for muscle velocity."""

        if lm < 0.1*lmopt:
            lm = 0.1*lmopt
        #lt0 = lmt0 - lm0*np.cos(alpha0)        
        a = self.activation(t)
        lmt = self.lmt_eq(t, lmt0)
        alpha = self.penn_ang(lmt=lmt, lm=lm, lm0=lm0, alpha0=alpha0)
        lt = lmt - lm*np.cos(alpha)
        fse = self.force_se(lt=lt, ltslack=ltslack)
        fpe = self.force_pe(lm=lm/lmopt)
        fl = self.force_l(lm=lm/lmopt)
        fce_t = fse/np.cos(alpha) - fpe
        #if fce_t < 0: fce_t=0
        vm = self.velo_fm(fm=fce_t, a=a, fl=fl)
        #print([t, a, lm, fse, fpe, fce_t, fl, vm])
        #fce = self.force_vm(vm=vm, fl=fl, lmopt=lmopt, a=a) + fpe  
        #d = [t, lmt, lm, lt, vm, fce*fm0, fse*fm0, a*fl*fm0, fpe*fm0]
        #self.lm_data2.append(d)
        #print(alpha, lm, lt, fl, fse, fpe, fce)

        return vm


    def lm_sol(self, fun=None, t0=0, t1=3, lm0=None, lmt0=None, ltslack=None, lmopt=None,
               alpha0=None, vmmax=None, fm0=None,  show=False, axs=None):
        """Runge-Kutta (4)5 ODE solver for muscle length."""

        if lm0 is None: lm0 = self.S['lm0']
        if lmt0 is None: lmt0 = self.S['lmt0']
        if ltslack is None: ltslack = self.P['ltslack']
        if alpha0 is None: alpha0 = self.P['alpha0']
        if lmopt is None: lmopt = self.P['lmopt']
        if vmmax is None: vmmax = self.P['vmmax']
        if fm0 is None: fm0 = self.P['fm0']
        
        if fun is None:
            fun = self.vm_eq
        f = ode(fun).set_integrator('dopri5', nsteps=1, max_step=0.005, atol=1e-8)  
        f.set_initial_value(lm0, t0).set_f_params(lm0, lmt0, lmopt, ltslack, alpha0, vmmax, fm0)
        # suppress Fortran warning
        warnings.filterwarnings("ignore", category=UserWarning)
        data = []
        #self.lm_data2 = []
        while f.t < t1:
            f.integrate(t1, step=True)
            d = self.calc_data(f.t, np.max([f.y, 0.1*lmopt]), lm0, lmt0,
                                           ltslack, lmopt, alpha0, fm0)
            data.append(d)

        warnings.resetwarnings()
        data = np.array(data)
        self.lm_data = data
        #self.lm_data2 = np.array(self.lm_data2)
        #data = self.lm_data2
        if show:
            self.lm_plot(data, axs)
        
        return data
        
        
    def calc_data(self, t, lm, lm0, lmt0, ltslack, lmopt, alpha0, fm0):
        """Calculus of muscle-tendon variables."""
        
        a = self.activation(t)
        lmt = self.lmt_eq(t, lmt0=lmt0)
        alpha = self.penn_ang(lmt=lmt, lm=lm, lm0=lm0, alpha0=alpha0)
        lt = lmt - lm*np.cos(alpha)
        fl = self.force_l(lm=lm/lmopt)
        fpe = self.force_pe(lm=lm/lmopt)
        fse = self.force_se(lt=lt, ltslack=ltslack)
        fce_t = fse/np.cos(alpha) - fpe
        vm = self.velo_fm(fm=fce_t, a=a, fl=fl, lmopt=lmopt)
        fce = self.force_vm(vm=vm, fl=fl, lmopt=lmopt, a=a) + fpe   
        data = [t, lmt, lm, lt, vm, fce*fm0, fse*fm0, a*fl*fm0, fpe*fm0]
        
        return data


    def actvation_plot(self, data, axs):
        """Plot results of actdyn_ode45 function."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
            return
            
        if axs is None:
            _, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
                
        axs.plot(data[:, 0], data[:, 1], color=[1, 0, 0, .6], label='Excitation')
        axs.plot(data[:, 0], data[:, 2], color=[0, 0, 1, .6], label='Activation')
        axs.set_xlabel('Time [s]')
        axs.set_ylabel('Level')
        axs.legend()
        plt.title('Activation dynamics')
        plt.tight_layout()
        plt.show()   
            
    
    def muscle_plot(self, a=1, axs=None):
        """Plot muscle-tendon relationships with length and velocity."""

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
            return
        
        if axs is None:
            _, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
        
        lmopt   = self.P['lmopt']
        ltslack = self.P['ltslack']
        vmmax   = self.P['vmmax']
        alpha0  = self.P['alpha0']
        fm0     = self.P['fm0']
        lm0     = self.S['lm0']
        lmt0    = self.S['lmt0']
        lt0     = self.S['lt0']
        if np.isnan(lt0): lt0 = lmt0 - lm0*np.cos(alpha0)
        
        lm  = np.linspace(0, 2, 101)
        lt  = np.linspace(0, 1, 101)*0.05 + 1
        vm  = np.linspace(-1, 1, 101)*vmmax*lmopt
        fl  = np.zeros(lm.size)
        fpe = np.zeros(lm.size)
        fse = np.zeros(lt.size)
        fvm = np.zeros(vm.size)
        
        fl_lm0  = self.force_l(lm0/lmopt)
        fpe_lm0 = self.force_pe(lm0/lmopt)
        fm_lm0  = fl_lm0 + fpe_lm0
        ft_lt0  = self.force_se(lt0, ltslack)*fm0        
        
        for i in range(101):
            fl[i]  = self.force_l(lm[i])
            fpe[i] = self.force_pe(lm[i])
            fse[i] = self.force_se(lt[i], ltslack=1)
            fvm[i] = self.force_vm(vm[i], a=a, fl=fl_lm0)

        lm  = lm*lmopt
        lt  = lt*ltslack
        fl  = fl
        fpe = fpe
        fse = fse*fm0
        fvm = fvm*fm0
            
        xlim = self.margins(lm, margin=.05, minmargin=False)
        axs[0].set_xlim(xlim)
        ylim = self.margins([0, 2], margin=.05)
        axs[0].set_ylim(ylim)
        axs[0].plot(lm, fl, 'b', label='Active')
        axs[0].plot(lm, fpe, 'b--', label='Passive')
        axs[0].plot(lm, fl+fpe, 'b:', label='')
        axs[0].plot([lm0, lm0], [ylim[0], fm_lm0], 'k:', lw=2, label='')
        axs[0].plot([xlim[0], lm0], [fm_lm0, fm_lm0], 'k:', lw=2, label='')
        axs[0].plot(lm0, fm_lm0, 'o', ms=6, mfc='r', mec='r', mew=2, label='fl(LM0)')
        axs[0].legend(loc='best', frameon=True, framealpha=.5)
        axs[0].set_xlabel('Length [m]')
        axs[0].set_ylabel('Scale factor')
        axs[0].locator_params(axis='both', nbins=5)
        axs[0].set_title('Muscle F-L (a=1)')
        
        xlim = self.margins([0, np.min(vm), np.max(vm)], margin=.05, minmargin=False)
        axs[1].set_xlim(xlim)
        ylim = self.margins([0, fm0*1.2, np.max(fvm)*1.5], margin=.025)
        axs[1].set_ylim(ylim)
        axs[1].plot(vm, fvm, label='')
        axs[1].set_xlabel('$\mathbf{^{CON}}\;$ Velocity [m/s] $\;\mathbf{^{EXC}}$')
        axs[1].plot([0, 0], [ylim[0], fvm[50]], 'k:', lw=2, label='')
        axs[1].plot([xlim[0], 0], [fvm[50], fvm[50]], 'k:', lw=2, label='')
        axs[1].plot(0, fvm[50], 'o', ms=6, mfc='r', mec='r', mew=2, label='FM0(LM0)')
        axs[1].plot(xlim[0], fm0, '+', ms=10, mfc='r', mec='r', mew=2, label='')
        axs[1].text(vm[0], fm0, 'FM0')
        axs[1].legend(loc='upper right', frameon=True, framealpha=.5)
        axs[1].set_ylabel('Force [N]')
        axs[1].locator_params(axis='both', nbins=5)
        axs[1].set_title('Muscle F-V (a=1)')

        xlim = self.margins([lt0, ltslack, np.min(lt), np.max(lt)], margin=.05, minmargin=False)
        axs[2].set_xlim(xlim)
        ylim = self.margins([ft_lt0, 0, np.max(fse)], margin=.05)
        axs[2].set_ylim(ylim)
        axs[2].plot(lt, fse, label='')
        axs[2].set_xlabel('Length [m]')
        axs[2].plot([lt0, lt0], [ylim[0], ft_lt0], 'k:', lw=2, label='')
        axs[2].plot([xlim[0], lt0], [ft_lt0, ft_lt0], 'k:', lw=2, label='')
        axs[2].plot(lt0, ft_lt0, 'o', ms=6, mfc='r', mec='r', mew=2, label='FT(LT0)')
        axs[2].legend(loc='upper left', frameon=True, framealpha=.5)
        axs[2].set_ylabel('Force [N]')
        axs[2].locator_params(axis='both', nbins=5)
        axs[2].set_title('Tendon')  
        plt.suptitle('Muscle-tendon mechanics', fontsize=18, y=1.03)
        plt.tight_layout(w_pad=.1)
        plt.show()
        
        
    def lm_plot(self, x, axs):
        """Plot results of actdyn_ode45 function.
            data = [t, lmt, lm, lt, vm, fm*fm0, fse*fm0, fl*fm0, fpe*fm0]
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib is not available.')
            return
        
        if axs is None:
            _, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(10, 6))

        axs[0, 0].plot(x[:, 0], x[:, 1], 'b', label='LMT')
        axs[0, 0].plot(x[:, 0], x[:, 2] + x[:, 3], 'g--', label='LM+LT')
        ylim = self.margins(x[:, 1], margin=.1)
        axs[0, 0].set_ylim(ylim)
        axs[0, 0].set_ylabel('$L_{MT}\,(m)$')
        axs[0, 0].locator_params(axis='both', nbins=5)
        axs[0, 0].legend(framealpha=.5, loc='best')
        
        axs[0, 1].plot(x[:, 0], x[:, 3], 'b')
        #axs[0, 1].plot(x[:, 0], lt0*np.ones(len(x)), 'r')
        ylim = self.margins(x[:, 3], margin=.1)
        axs[0, 1].set_ylim(ylim)
        axs[0, 1].set_ylabel('$L_{T}\,(m)$')
        axs[0, 1].locator_params(axis='both', nbins=5)
        
        axs[1, 0].plot(x[:, 0], x[:, 2], 'b')
        #axs[1, 0].plot(x[:, 0], lmopt*np.ones(len(x)), 'r')
        ylim = self.margins(x[:, 2], margin=.1)
        axs[1, 0].set_ylim(ylim)
        axs[1, 0].set_ylabel('$L_{M}\,(m)$')
        axs[1, 0].locator_params(axis='both', nbins=5)
        
        axs[1, 1].plot(x[:, 0], x[:, 4], 'b')
        ylim = self.margins(x[:, 4], margin=.1)
        axs[1, 1].set_ylim(ylim)
        axs[1, 1].set_ylabel('$V_{CE}\,(m/s)$')
        axs[1, 1].locator_params(axis='both', nbins=5)
        
        axs[2, 0].plot(x[:, 0], x[:, 5], 'b', label='CE')
        axs[2, 0].plot(x[:, 0], x[:, 6], 'g--', label='Tendon')
        axs[2, 0].set_ylabel('$Force\,(N)$')
        ylim = self.margins(x[:, [5, 6]], margin=.1)
        axs[2, 0].set_ylim(ylim)
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].locator_params(axis='both', nbins=5)
        axs[2, 0].legend(framealpha=.5, loc='best')
        
        #axs[2, 1].plot(x[:, 0], x[:, 7], 'b', label='FL')
        axs[2, 1].plot(x[:, 0], x[:, 8], 'g--', label='FPE')
        ylim = self.margins(x[:, [8]], margin=.1)
        axs[2, 1].set_ylim(ylim)
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('$Force\,(N)$')
        axs[2, 1].locator_params(axis='both', nbins=5)
        axs[2, 1].legend(framealpha=.5, loc='best')
        plt.suptitle('Simulation of muscle-tendon mechanics', fontsize=18,
                     y=1.03)
        plt.tight_layout()
        plt.show()
        
        
    def margins(self, x, margin=0.01, minmargin=True):
        """Calculate plot limits with extra margins.
        """
        rang = np.nanmax(x) - np.nanmin(x)
        if rang < 0.001 and minmargin:
            rang = 0.001*np.nanmean(x)/margin
            if rang < 1:
                rang = 1
        lim = [np.nanmin(x) - rang*margin, np.nanmax(x) + rang*margin]

        return lim
