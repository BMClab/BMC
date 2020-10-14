"""Read c3d files.

"""

__author__ = "Marcos Duarte, https://github.com/demotu/"
__version__ = "0.0.1"
__license__ = "MIT"

import os
import copy
import pprint
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import ezc3d

xr.set_options(keep_attrs=True)



def read_c3d(fname, var='POINT', Axis={'ML':0, 'AP':1, 'VT':2}, mm2m=True,
             prm=None):
    """Get data from c3d file.
    var = 'ANALOG' 'POINT' 'ANGLE' 'MOMENT' 'POWER' 'OXFORD'
    """
    data = []
    var = var.upper()
    if prm is None:
        prm = get_parameters(fname)
    c3d = ezc3d.c3d(fname).c3d_swig

    if var in ['ANALOG', 'GRF'] and c3d.parameters().isGroup('ANALOG'):
        data = np.transpose(c3d.get_analogs()[0])
        all_channels_names = list(c3d.channelNames())
        coords = {}
        # frame = 1 at t0; time starts at 1/freq to be compatible with events time
        f0 = 0
        coords['Time'] = np.round(np.arange(prm['frames']['a'][0] + f0,
                                            prm['frames']['a'][1] + f0 +
                                            1*prm['rates']['a']/prm['rates']['p'])/prm['rates']['a'], 6)
        if var == 'ANALOG':
            coords['Var'] = all_channels_names
            data = xr.DataArray(data=data, dims=('Time', 'Var'), coords=coords, name=var)
            data.attrs = {'units': 'a.u.'}
        elif var == 'GRF':
            # TODO
            #c = ezc3d.c3d(fname, extract_forceplat_data=True)  # doesn't work for kistler
            all_channels = np.array([np.array(list(Axis.values()))+6*fp for
                                     fp in range(int(data.shape[1]/6))]).flatten().tolist()
            all_labels = [all_channels_names[a].split('.')[1] for a in all_channels]
            data = data[:, all_channels]
            if np.nanmean(data[:, 2::3]) < 0:  # invert vertical grf if negative
                data[:, 2::3] = -data[:, 2::3]
            if np.nanmin(data[:, 2::3]) < 0:
                data[:, 2::3] = data[:, 2::3] - np.nanmin(data[:, 2::3], axis=0)
            data = np.expand_dims(data, 1)
            coords['Var'] = ['Force' + all_labels[0][2:]]
            coords['Axis'] = list(Axis.keys())
            data = xr.DataArray(data=data, dims=('Time', 'Var', 'Axis'), coords=coords, name=var)
            data.attrs = {'units': prm['units_all']['Force']}
        data['Time'].attrs = {'units': 's', 'rate': prm['rates']['a']}
    elif c3d.parameters().isGroup('POINT'):
        all_labels = c3d.pointNames()
        if var == 'POINT' or c3d.parameters().group('POINT').isParameter(var+'S'):
            if var == 'POINT':
                var_labels = c3d.pointNames()
                idx = list(range(len(var_labels)))
            else:
                var_labels = c3d.parameters().group('POINT').parameter(var+'S').valuesAsString()
                idx = [all_labels.index(channel) for channel in var_labels]
            data = np.transpose(c3d.get_points()[:3, idx, :][list(Axis.values())])
            if var == 'ANGLE':
                var_labels = [v[:-6] if 'Angles' in v else v for v in var_labels]
            var_labels = [v[:-len(var)] if var.capitalize() in v else v for v in var_labels]
            coords = {}
            coords['Time'] = np.round(np.arange(prm['frames']['p'][0],
                                                prm['frames']['p'][1] + 1)/prm['rates']['p'], 6)
            coords['Var'] = var_labels
            coords['Axis'] = list(Axis.keys())
            data = xr.DataArray(data=data, dims=('Time', 'Var', 'Axis'),
                                coords=coords, name=var.capitalize())
            if mm2m and var == 'MOMENT':
                data.values = data.values * prm['units_all']['scale']
                prm['units_all']['Moment'] = 'Nm'
            data.attrs = {'units': prm['units_all'][var.capitalize()]}
            data['Time'].attrs = {'units': 's', 'rate': prm['rates']['p']}
        elif var == 'OXFORD':
            var_labels = ['LHFTBA', 'LFFHFA', 'LFFTBA', 'LHXFFA',
                          'RHFTBA', 'RFFHFA', 'RFFTBA', 'RHXFFA']
            idx = [all_labels.index(channel) for channel in var_labels if channel in all_labels]
            if len(idx) < 8:
                #print('Not all data found for var "{}"'.format('OXFORD'))
                pass
            time = np.arange(prm['frames']['p'][0],
                             prm['frames']['p'][1] + 1)/prm['rates']['p']
            data = np.zeros((time.shape[0], 8, 3))*np.nan
            data[:, :len(idx), :] = np.transpose(c3d.get_points()[:3, idx, :][list(Axis.values())])
            a = Axis['VT']
            if 'LArchHeightIndex' in all_labels:
                data[:, 3, Axis['AP']] = c3d.get_points()[a, all_labels.index('LArchHeightIndex'), :]
            if 'RArchHeightIndex' in all_labels:
                data[:, 7, Axis['AP']] = c3d.get_points()[a, all_labels.index('RArchHeightIndex'), :]
            if 'LFTA' in all_labels:
                data[:, 3, Axis['VT']] = c3d.get_points()[a, all_labels.index('LFTA'), :]
            if 'RFTA' in all_labels:                
                data[:, 7, Axis['VT']] = c3d.get_points()[a, all_labels.index('RFTA'), :]
            coords = {}
            coords['Time'] = np.round(time, 6)
            coords['Var'] = var_labels
            coords['Axis'] = list(Axis.keys())
            data = xr.DataArray(data=data, dims=('Time', 'Var', 'Axis'),
                                coords=coords, name=var.capitalize())
            if mm2m:               
                data.loc[dict(Var='LHXFFA', Axis='AP')] = data.sel(Var='LHXFFA', Axis='AP').values *\
                                                          prm['units_all']['scale']
                data.loc[dict(Var='RHXFFA', Axis='AP')] = data.sel(Var='RHXFFA', Axis='AP').values *\
                                                          prm['units_all']['scale']
            data.attrs = {'units': prm['units_all']['Angle']}
            data['Time'].attrs = {'units': 's', 'rate': prm['rates']['p']}                
        else:
            raise ValueError('Invalid option for var: {}'.format(var))
    else:
        raise ValueError('Invalid option for var: {} for data in: {}'.format(var, fname))
    data.attrs.update(prm)
    data['Time'].attrs.update({'events': prm['events'],
                               'side': prm['events']['0'][1][0]})

    return data


def get_parameters(fname):
    """Get parameters from c3d file.
    """
    missing = np.nan
    c3d = ezc3d.c3d(fname).c3d_swig
    c = c3d.parameters().group
    units_all = {'Point' : c('POINT').parameter('UNITS').valuesAsString()[0],
                 'Mass': 'kg', 'Length': 'm', 'Time': 's', 'g': 9.80665}
    if c('POINT').isParameter('ANGLE_UNITS') and c('POINT').isParameter('FORCE_UNITS'):
        units_all.update({'Angle' : c('POINT').parameter('ANGLE_UNITS').valuesAsString()[0],
                          'Force' : c('POINT').parameter('FORCE_UNITS').valuesAsString()[0],
                          'Moment' : c('POINT').parameter('MOMENT_UNITS').valuesAsString()[0],
                          'Power' : c('POINT').parameter('POWER_UNITS').valuesAsString()[0]
                         })
    else:
        units_all.update({'Angle' : '', 'Force' : '',
                          'Moment' : '', 'Power' : ''})
        print('{} does not have ANGLE_UNITS.'.format(fname))
    if units_all['Point'] == 'cm':
        scale = .01
    elif units_all['Point'] == 'mm':
        scale = .001
    else:
        scale = 1
    units_all['scale'] = scale
    if (c3d.parameters().isGroup('ANALYSIS') and
        c('ANALYSIS').isParameter('NAMES') and
        c('ANALYSIS').isParameter('UNITS')):
        units_all.update(dict(zip(c('ANALYSIS').parameter('NAMES').
                                  valuesAsString(),
                                  c('ANALYSIS').parameter('UNITS').
                                  valuesAsString())))
    else:
        #print('{} does not have ANALYSIS.'.format(fname))
        pass
    LL, FL = {'L': np.nan, 'R': np.nan}, {'L': np.nan, 'R': np.nan}
    if c3d.parameters().isGroup('PROCESSING'):
        if c('PROCESSING').isParameter('Bodymass'):
            mass = np.round(c('PROCESSING').parameter('Bodymass').
                            valuesAsDouble()[0], 3)
        if c('PROCESSING').isParameter('Height'):
            height = np.round(c('PROCESSING').parameter('Height').
                              valuesAsDouble()[0]*units_all['scale'], 3)
        if (c('PROCESSING').isParameter('UpperLegLength') and
            c('PROCESSING').isParameter('LowerLegLength')):
            LL['L'] = np.round((c('PROCESSING').parameter('UpperLegLength').
                                valuesAsDouble()[0] +
                                c('PROCESSING').parameter('LowerLegLength').
                                valuesAsDouble()[0])*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('LLegLength'):
            LL['L'] = np.round(c('PROCESSING').parameter('LLegLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('LegLength'):
            LL['L'] = np.round(c('PROCESSING').parameter('LegLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        if (c('PROCESSING').isParameter('UpperLegLength') and
            c('PROCESSING').isParameter('LowerLegLength')):
            LL['R'] = np.round((c('PROCESSING').parameter('UpperLegLength').
                                valuesAsDouble()[0] +
                                c('PROCESSING').parameter('LowerLegLength').
                                valuesAsDouble()[0])*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('RLegLength'):
            LL['R'] = np.round(c('PROCESSING').parameter('RLegLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('LegLength'):
            LL['R'] = np.round(c('PROCESSING').parameter('LegLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        if c('PROCESSING').isParameter('LFootLength'):
            FL['L'] = np.round(c('PROCESSING').parameter('LFootLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('FootLength'):
            FL['L'] = np.round(c('PROCESSING').parameter('FootLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        if c('PROCESSING').isParameter('RFootLength'):
            FL['R'] = np.round(c('PROCESSING').parameter('RFootLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
        elif c('PROCESSING').isParameter('FootLength'):
            FL['R'] = np.round(c('PROCESSING').parameter('FootLength').
                               valuesAsDouble()[0]*units_all['scale'], 3)
    else:
        mass, height = np.nan, np.nan

    rates = {'p': c3d.header().frameRate(),
             'a': c3d.header().frameRate() * c3d.header().nbAnalogByFrame()}
    frames = {'p': [c3d.header().firstFrame(), c3d.header().lastFrame()],
              'a': [c3d.header().firstFrame() * c3d.header().nbAnalogByFrame(),
                    c3d.header().lastFrame() * c3d.header().nbAnalogByFrame()]}

    events = get_events(fname, missing=missing)

    param = {'filename': os.path.splitext(os.path.basename(fname))[0],
             'mass': mass, 'height': height, 'LL': LL, 'FL': FL,
             'units_all': units_all,
             'rates': rates, 'frames': frames, 'events': events}
    if (c3d.parameters().isGroup('ANALYSIS') and
        c('ANALYSIS').isParameter('NAMES') and
        c('ANALYSIS').isParameter('VALUES')):
        param.update(dict(zip(c('ANALYSIS').parameter('NAMES').valuesAsString(),
                              np.round(c('ANALYSIS').parameter('VALUES').
                                       valuesAsDouble(), 3))))

    return param


def get_events(fname, missing=np.nan):
    """Get events from c3d file.
    """
    c3d = ezc3d.c3d(fname).c3d_swig
    events = {'0': [np.inf, ''], '1': [-np.inf, ''],
              'LFS': [missing], 'LFO': [missing], 'RFS': [missing], 'RFO': [missing]}
    if c3d.parameters().isGroup('EVENT'):
        labels = c3d.parameters().group('EVENT').parameter('LABELS').valuesAsString()
        contexts = c3d.parameters().group('EVENT').parameter('CONTEXTS').valuesAsString()
        times = np.round(np.array(c3d.parameters().group('EVENT')
                                  .parameter('TIMES').valuesAsDouble())[1::2], 6)
        #if times[-1] < times[0]:  # delete possible out-of-order event at the end
        #    times, labels, contexts = times[:-1], labels[:-1], contexts[:-1]
        evs = {}
        for label, context, time in zip(labels, contexts, times):
            key = context + ' ' + label
            key2 = ''.join([i[0] for i in key.strip().split(' ')]).upper()
            evs[key2] = np.r_[evs[key2], np.atleast_1d(time)] if \
                        key2 in evs.keys() else np.atleast_1d(time)
            if time < events['0'][0]:
                events['0'] = [time, key2]
            if time > events['1'][0]:
                events['1'] = [time, key2]
        for ev in evs:
            if len(ev) > 1:
                evs[ev] = np.sort(evs[ev])
        events.update(evs)

    return events


def trimmer(da, evs=None, trim=0):
    """Trim xarray data based on events.

    trim  ['all', 'evs', 'grf']
    """
    if trim:
        if evs is None:
            evs = copy.deepcopy(da.Time.attrs['events'])
        else:
            evs = copy.deepcopy(evs)
        knan = [key for key in evs.keys() if np.isnan(evs[key][0])]
        for key in knan:
            evs.pop(key)
        da = da.sel({'Time': slice(evs['0'][0],
                                   evs['1'][0]+da['Time'][1]-da['Time'][0])})
        # let's force t0 = 0 even if it's not the closest to the first event
        da['Time'] = da['Time'] - da['Time'].values[0]
        #da = da.assign_coords({'Time': da['Time']-da['Time'].values[0]})
        t0 = evs['0'][0]
        for ev in evs:
            if ev in ['0', '1']:
                evs[ev][0] = np.round(evs[ev][0] - t0, 6)
            else:
                evs[ev] = np.round(evs[ev] - t0, 6)
        da['Time'].attrs.update({'events': evs})

    return da


def find_ev_GRFcycle(da, evs=None, side=None):
    """Find the foot strike events of one cycle with GRF data.
    """
    if evs is None:
        evs = copy.deepcopy(da.Time.attrs['events'])
    else:
        evs = copy.deepcopy(evs)
    if side is None:
        if 'side' in da.Time.attrs:
            side = da.Time.attrs['side']
        else:
            side = evs['0'][1][0]

    side2 = 'L' if side == 'R' else 'R'
    sFS = evs[side+'FS']
    while len(sFS) > 2:
        idx = np.argmin([da.sel(Time=slice(sFS[0], sFS[1]), Axis='VT').mean(),
                         np.inf,
                         da.sel(Time=slice(sFS[1], sFS[2]), Axis='VT').mean()])
        evs[side+'FS'] = np.delete(sFS, idx)
        evs['0'] = [np.min(evs[side+'FS']), side+'FS']
        evs['1'] = [np.max(evs[side+'FS']), side+'FS']
        if evs[side+'FO'][0] < evs['0'][0]:
            evs[side+'FO'] = np.delete(evs[side+'FO'], 0)
        if side2+'FO' in evs and evs[side2+'FO'][0] < evs['0'][0]:
            evs[side2+'FO'] = np.delete(evs[side2+'FO'], 0)
        sFS = evs[side+'FS']
    evs['0'] = [np.min(evs[side+'FS']), side+'FS']
    while side2+'FS' in evs and evs[side2+'FS'][0] < evs['0'][0]:
        evs.pop(side2+'FS')
        while side2+'FO' in evs and evs[side2+'FO'][0] < evs['0'][0]:
            evs.pop(side2+'FO')
    while side2+'FS' in evs and evs[side2+'FS'][-1] > evs['1'][0]:
        evs.pop(side2+'FS')
        while side2+'FO' in evs and evs[side2+'FO'][-1] < evs['1'][0]:
            evs.pop(side2+'FO')

    return evs


def normala(da, method, mass=1, LL=1, g=9.80665, value=1, units='a.u.'):
    """Amplitude normalization of data in xarray.
    """
    if method == 'BM':
        da.values, units = da.values/mass, da.attrs['units']+'/kg'
    elif method == 'BW':
        da.values, units = da.values/(mass*g), da.attrs['units']+'/BW'
    elif method == 'BMLL':
        da.values, units = da.values/(mass*LL), da.attrs['units']+'/kg*LL'
    elif method == 'BWLL':
        da.values, units = da.values/(mass*g*LL), da.attrs['units']+'/BW*LL'
    else:
        da.values = da.values/value
    da.attrs.update({'units': units, 'normala': 1})
    return da


def normalt(da, method='scale', value=100, units='% cycle',
            axis=0, step=1, k=3, smooth=0, mask=None, nan_at_ext='delete'):
    """Time normalization of data in xarray.
    """
    period = np.round(da['Time'].values[-1] - da['Time'].values[0], 6)
    if method == 'scale':
        da['Time'] = da['Time']/period * value
        #da = da.assign_coords(Time=da.Time/tend * value)
        evs = copy.deepcopy(da['Time'].attrs['events'])
        for ev in evs:
            if ev in ['0', '1']:
                evs[ev][0] = np.round(evs[ev][0]/period * value, 6)
            else:
                evs[ev] = np.round(evs[ev]/period * value, 6)
        da['Time'].attrs.update({'units': units, 'normalt': 1, 'events': evs,
                                 'rate': da['Time'].attrs['rate'] * period / value})
    elif method == 'tnorm':
        from tnorma import tnorma

        yn, tn, indie = tnorma(da.values, axis=axis, step=step, k=k,
                               smooth=smooth, mask=mask, nan_at_ext=nan_at_ext)
        coords = {}
        coords['Time'] = tn
        coords['Axis'] = da['Axis'].values.tolist()
        attrs = copy.deepcopy(da.attrs)
        tattrs = copy.deepcopy(da['Time'].attrs)
        da = xr.DataArray(data=yn, dims=('Time', 'Axis'), coords=coords, name=da.name)
        da.attrs = attrs
        da['Time'].attrs = tattrs
        evs = tattrs['events']
        for ev in evs:
            if ev in ['0', '1']:
                evs[ev][0] = np.round(evs[ev][0]/period * tn[-1], 6)
            else:
                evs[ev] = np.round(evs[ev]/period * tn[-1], 6)
        da['Time'].attrs.update({'units': units, 'normalt': 1, 'events': evs,
                                 'rate': (len(tn)-1) / period })
    elif method == 'interp':
        tattrs = copy.deepcopy(da['Time'].attrs)
        evs = tattrs['events']
        for ev in evs:
            if ev in ['0', '1']:
                evs[ev][0] = np.round(evs[ev][0]/period * 100, 6)
            else:
                evs[ev] = np.round(evs[ev]/period * 100, 6)
        da['Time'] = np.linspace(0, 100, da['Time'].shape[0])
        da = da.interp(Time=np.linspace(0, 100, int(100/step+1)),
                       method='linear', assume_sorted=True)
        da['Time'].attrs = tattrs
        da['Time'].attrs.update({'units': units, 'normalt': 1, 'events': evs,
                                 'rate': (len(da['Time'])-1) / period})

    return da


def plot_lines(axes, evs, hline=True, evline=True, show=False):
    """plot horizontal line and events.
    """
    cev = {'L':'r', 'R':'b'}
    lev = {'S':'-', 'O':'--'}
    if hasattr(axes, 'ndim') and axes.ndim > 1:
        axes = axes.flatten()
    elif not isinstance(axes, list):
        axes = [axes]
    for ax in axes:
        if hline:
            ax.axhline(y=0, c='k', lw=0.5)
        if evline:
            for e in evs:
                if len(e) > 1:
                    for v in evs[e]:
                        ax.axvline(x=v, c=cev[e[0]], ls=lev[e[-1]], lw=1)
        ax.margins(y=.2)
    if show:
        plt.tight_layout()
        plt.show()


def plot(da, *args, var=None, sharey=False, size=2.5, aspect=1.5, show=True,
         hline=True, evline=True, title='', **kwargs):
    """Plot data from c3d xarray
    """
    if var is None:
        var = da.Var.values
    g = da.sel(Var=var).plot.line(x='Time', row='Var', col='Axis', sharey=sharey,
                                  size=size, aspect=aspect, *args, **kwargs)
    plot_lines(axes=g.axes, evs=da.Time.attrs['events'], hline=hline,
               evline=evline)
    if title:
        g.fig.suptitle(title, y=1)
    if show:
        plt.tight_layout()
        plt.show()
    return g


def printdict(x):
    """Print dictionary elements up to 3 levels in colors.
    """
    # colors
    r = "\033[1;31m"
    b = "\033[1;34m"
    g = "\033[1;32m"
    n = "\033[0m"
    if not hasattr(x, 'keys'):
        print(x)
        return
    print('keys:', r, [key for key in x.keys()], n)
    for k in x.keys():
        print("['{}{}{}']:".format(r, k, n))
        if not hasattr(x[k], 'keys'):
            print(x[k])
            continue
        print(' keys:', g, [key for key in x[k].keys()], n)
        for k2 in x[k].keys():
            print(" ['{}{}{}']['{}{}{}']:".format(r, k, n, g, k2, n))
            if hasattr(x[k][k2], 'keys'):
                print('  keys:', b, [key for key in x[k][k2].keys()], n)
                for k3 in x[k][k2].keys():
                    print("  ['{}{}{}']['{}{}{}']['{}{}{}']:".format(r, k, n, g, k2,
                                                                     n, b, k3, n))
                    if isinstance(x[k][k2][k3], np.ndarray):
                        print('  shape:', x[k][k2][k3].shape)
                    else:
                        print('  ', sep='', end='')
                        pprint.pprint(x[k][k2][k3], indent=2)
            else:
                print('  shape:', x[k][k2].shape)
