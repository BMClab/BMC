import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Read Zebris pressure platform ASCII files

        Marcos Duarte
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Zebris Medical GmbH (https://www.zebris.de/en/) builds devices for measuring pressure on the foot/platform interface.  
        The BMClab (http://demotu.org/) has two zebris FDM 1.5 platforms for measuring pressure and each one has the following specs:  
         - Measuring principle: capacitive force measurement  
         - Dimensions: 158 x 60.5 x 2.1 cm (L x W x H)  
         - Sensor area: 149 x 54.2 cm (L x W)  
         - Number of sensors: 11264  
         - Physical resolution: 1.4 sensors /cm2  (0.714 cm2)
         - Sampling frequency: 100 Hz  
         - Measuring Range: 1 - 120 N/cm2  
         - Accuracy of the calibrated measuring range: (1 – 80 N/cm2), ±5% (FS)  
         - Hysteresis: < 3 % (FS)  
 
        The two pressure platforms can be synchronized and used as a single 3-m platform.  

        The proprietary software to operate the pressure device saves files in ASCII and binary formats with the pressure data. Here are functions for reading most of the files saved in ASCII format. These files have headers with metadata about the patient and acquisition conditions and the data of pressure, force, or center of pressure  depending on the type of acquisition and chosen option to export the data.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## APD file

        The .apd file in ASCII contains the metadata and the maximum values of pressure during the trial only at the regions where there were pressure greater than the threshold (1 N/cm2). This file can be used for making insoles.  
        Here is a typical .apd file:
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo
    import sys
    sys.path.insert(1, r'./../functions')
    return np, plt


@app.cell
def _():
    path2 = './../Data/'
    filename = path2 + 'MDwalk2.apd'
    with open(file=filename, mode='rt', newline='') as f:
        print(f.read())
    return filename, path2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is a function for reading an .apd file from Zebris pressure platform:
        """
    )
    return


@app.cell
def _(np):
    def read_zebris_apd(filename):
        """Reads Zebris pressure platform ASCII files .apd.
        """
        import pprint
        sections = ['General', 'Customer', 'Technical', 'Data']
        _s = 0
        info = {}
        with open(file=filename, mode='rt', newline='') as f:
            for linea in f:
                line = linea.strip('\r[]\n')
                if line == sections[_s]:
                    info[sections[_s]] = {}
                    _s += 1
                elif line:
                    info[sections[_s - 1]][line.split('=')[0]] = line.split('=')[1]
                elif _s == 3:
                    break
            f.readline()
            data = np.loadtxt(f, delimiter='\t')
            data[data == -1] = 0
            print('File %s successfully open.' % filename)
            print('Data has %d rows and %d columns.' % data.shape)
        return (info, data)
    return (read_zebris_apd,)


@app.cell
def _(filename, read_zebris_apd):
    info, data = read_zebris_apd(filename)
    return data, info


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Simple 2D plot of the raw data
        """
    )
    return


@app.cell
def _(data, info, np, plt):
    from matplotlib import cm
    dx = float(info['Technical']['LDistX']) / 10
    dy = float(info['Technical']['LDistY']) / 10
    x = np.arange(1 / 2, data.shape[0] + 1 / 2, 1) * dx
    y = np.arange(1 / 2, data.shape[1] + 1 / 2, 1) * dy
    (X, Y) = np.meshgrid(y, x)
    print('Shapes:')
    print('X:', X.shape, 'Y:', Y.shape, 'data:', data.shape)
    (_fig, _ax) = plt.subplots(figsize=(6, 7))
    _img = _ax.pcolormesh(X, Y, data, cmap=cm.jet)
    _ax.set_aspect('equal')
    _fig.colorbar(_img, label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    _ax.set_title('Plantar pressure during walking')
    plt.show()
    return X, Y, cm, dx, dy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 2D plot with filtering

        Let's use the matplotlib function [`imshow`](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.imshow) for plotting raw and filtered data.  
        For the difference between `imshow` and `pcolormesh`, see https://stackoverflow.com/questions/21166679/when-to-use-imshow-over-pcolormesh.
        """
    )
    return


@app.cell
def _(cm, data, plt):
    (_fig, _ax) = plt.subplots(1, 2, figsize=(12, 7))
    _img0 = _ax[0].imshow(data, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest')
    _ax[0].set_xlabel('Y (cm)')
    _ax[0].set_ylabel('X (cm)')
    _img1 = _ax[1].imshow(data, cmap=cm.jet, aspect='equal', origin='lower', interpolation='bilinear', vmin=0, vmax=40)
    _ax[1].set_xlabel('Y (cm)')
    _ax[1].set_ylabel('X (cm)')
    _fig.colorbar(_img1, ax=list(_ax), label='Pressure (N/cm$^2$)')
    _fig.suptitle('Plantar pressure during walking', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 3D plots
        """
    )
    return


@app.cell
def _(X, Y, cm, data, plt):
    from mpl_toolkits.mplot3d import Axes3D
    _fig = plt.figure(figsize=(14, 6))
    _ax = _fig.gca(projection='3d')
    _surf = _ax.plot_surface(X, Y, data, cmap=cm.jet, rcount=data.shape[0], ccount=data.shape[1], linewidth=0, antialiased=True)
    _ax.view_init(60, 200)
    _fig.colorbar(_surf, orientation='vertical', label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### 3D plot with data filtering
        """
    )
    return


@app.cell
def _(X, Y, data, dx, info, np):
    from scipy import interpolate
    # interpolate data over a four times denser grid
    dxy = float(info['Technical']['LDistX'])/10  # the pressure cell is squared
    x2 = np.arange(1/8, data.shape[0] + 1/8, 1/4)*dx
    y2 = np.arange(1/8, data.shape[1] + 1/8, 1/4)*dx
    X2, Y2 = np.meshgrid(y2, x2)
    tck = interpolate.bisplrep(X, Y, data)
    data2 = interpolate.bisplev(X2[0,:], Y2[:,0], tck).T
    print('Shapes:')
    print('X2:', X2.shape, 'Y2:', Y2.shape, 'data2:', data2.shape)
    return X2, Y2, data2


@app.cell
def _(X2, Y2, cm, data2, plt):
    _fig = plt.figure(figsize=(14, 6))
    _ax = _fig.gca(projection='3d')
    _surf = _ax.plot_surface(X2, Y2, data2, cmap=cm.jet, rcount=data2.shape[0], ccount=data2.shape[1], linewidth=0, antialiased=False)
    _ax.view_init(60, 200)
    _fig.colorbar(_surf, orientation='vertical', label='Pressure (N/cm$^2$)')
    _ax.set_xlabel('Y (cm)')
    _ax.set_ylabel('X (cm)')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Loading several steps of a trial
        """
    )
    return


@app.cell
def _(cm, dx, dy):
    def plot_apd(info, data, ax, title):
        dxy = float(info['Technical']['LDistX']) / 10
        x0 = int(info['Technical']['StartSensY']) * dx
        y0 = int(info['Technical']['StartSensX']) * dy
        xlen = int(info['Technical']['SensCountY'])
        ylen = int(info['Technical']['SensCountX'])
        _img = _ax.imshow(data, cmap=cm.jet, aspect='auto', origin='lower', extent=[x0, x0 + xlen * dx, y0, y0 + ylen * dy], interpolation='nearest', vmin=0, vmax=40)
        _ax.set_title(title)
        _ax.set_xlabel('Y (cm)')
        return _img
    return (plot_apd,)


@app.cell
def _(path2, plot_apd, plt, read_zebris_apd):
    steps = ['MDwalk2' + _step for _step in ['', '_1', '_2', '_3', '_4']]
    (infos, datas) = ({}, {})
    (_fig, _axs) = plt.subplots(1, len(steps), figsize=(14, 7))
    for (_s, _step) in enumerate(steps):
        (infos[_step], datas[_step]) = read_zebris_apd(path2 + _step + '.apd')
        _img = plot_apd(infos[_step], datas[_step], _axs[_s], _step)
    _axs[0].set_ylabel('X (cm)')
    _fig.colorbar(_img, ax=list(_axs), label='Pressure (N/cm$^2$)', orientation='horizontal', pad=0.1, aspect=40)
    plt.show()
    return datas, infos, steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It seems the Zebris software when exporting the pressure data to .apd files of a trial with a sequence of steps, saves first the data of the right foot and then of the left foot.  

        The origin of the pressure platform was at the far right corner in relation to the walking direction of the subject in this trial (see the values at the x and y axes). So, the order of the subject's foot steps were: MDwalk2_4, MDwalk2_1, MDwalk2_3, MDwalk2, and MDwalk2_2.

        The size of the pressure data across steps is not constant, the width varies from 12 to 15 columns and the length varies from 30 to 32 rows of data. Multiply these numbers by 0.846591 cm to have the size of the pressure data in centimeters. Possible reasons for this variation are: 1. foot rotation, 2. differences at how the foot is supported at the ground at each step, and 3. perhaps measurement noise. So, one can't directly compare the images (the pressures), for example, we can't average the data in order to get the mean foot pressure (or other statistics related to the positions of the steps); we will have first to align the data (each step) and account for the different sizes. In image processing, this procedure is part of what is known as [image registration](https://en.wikipedia.org/wiki/Image_registration). For the application of image registration to foot plantar pressure, see Pataky et al. (2008), Oliveira et al. (2010).

        For now, given that there are only few steps and they seemed to be similar, we will only transform the images to have the same size.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Image resize
        """
    )
    return


@app.cell
def _(datas, infos, plot_apd, plt, steps):
    from scipy.misc import imresize
    datas2 = {}
    (_fig, _axs) = plt.subplots(1, len(steps), figsize=(14, 7))
    for (_s, _step) in enumerate(steps):
        maxdata = datas[_step].max()
        datas2[_step] = imresize(datas[_step], size=(120, 60), interp='bilinear')
        datas2[_step] = maxdata * (datas2[_step] / datas2[_step].max())
        print('%s has %d rows and %d columns.' % (_step, *datas2[_step].shape))
        _img = plot_apd(infos[_step], datas2[_step], _axs[_s], _step)
        _axs[_s].set_aspect('equal')
    _axs[0].set_ylabel('X (cm)')
    _fig.colorbar(_img, ax=list(_axs), label='Pressure (N/cm$^2$)', orientation='horizontal', pad=0.1, aspect=40)
    plt.show()
    return (datas2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can now calculate for instance the left and right steps with the maxium pressure of all corresponding steps:
        """
    )
    return


@app.cell
def _(cm, datas2, np, plt, steps):
    max_step_r = np.max(np.stack((datas2[steps[0]], datas2[steps[1]]), 2), 2)
    max_step_l = np.max(np.stack((datas2[steps[2]], datas2[steps[3]], datas2[steps[4]]), 2), 2)
    (_fig, _axs) = plt.subplots(1, 2, figsize=(12, 7))
    _img0 = _axs[0].imshow(max_step_l, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest', vmin=0, vmax=40)
    _axs[0].set_xlabel('Y')
    _axs[0].set_ylabel('X')
    _img1 = _axs[1].imshow(max_step_r, cmap=cm.jet, aspect='equal', origin='lower', interpolation='nearest', vmin=0, vmax=40)
    _axs[1].set_xlabel('Y')
    _fig.colorbar(_img1, ax=list(_axs), label='Pressure (N/cm$^2$)')
    _fig.suptitle('Plantar pressure during walking (maximum values across steps)', fontsize=16)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It doesn't work!  
        We need to perform image registration...
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## References

        - [Pataky TC1, Goulermas JY, Crompton RH. (2008) A comparison of seven methods of within-subjects rigid-body pedobarographic image registration. J Biomech., 20;41(14):3085-9. doi: 10.1016/j.jbiomech.2008.08.001](https://www.ncbi.nlm.nih.gov/pubmed/18790481).  
        - [Oliveira FP1, Pataky TC, Tavares JM (2010) Registration of pedobarographic image data in the frequency domain. Comput Methods Biomech Biomed Engin., 13(6):731-40. doi: 10.1080/10255840903573020](https://www.ncbi.nlm.nih.gov/pubmed/20526916). [PDF](http://paginas.fe.up.pt/~tavares/downloads/publications/artigos/CMBBE_13_6_2010.pdf)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
