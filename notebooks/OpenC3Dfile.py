import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Open C3D files
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Marcos Duarte
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Text and binary files**   
        There are two kinds of computer files: text and binary files. Text files are structured as a sequence of lines of electronic text. The most common formats of a text file are ASCII (with 128 ($2^7$) different characters and UTF-8 (which includes non-English characters). A binary file is a file that is not structured as a text file. Because in fact everything in a computer is stored in binary format (a sequence of zeros and ones), text files are binary files that store text codes.

        To open and read a text file is simple and straightforward. A text file doesn't need additional information to be read, and can be openned by any text-processing software. This is not the case of a binary file, we need to have extra information about how the file is structured to be able to read it. However, binary files can store more information per file size than text files and we can read and write binary files faster than text files. This is one of the reasons why software developers would choose a binary format.

        **C3D format**

        > The C3D format is a public domain, binary file format that has been used in Biomechanics, Animation and Gait Analysis laboratories to record synchronized 3D and analog data since the mid 1980's.  It is supported by all 3D major Motion Capture System manufacturers, as well as other companies in the Biomechanics, Motion Capture and Animation Industries ([http://www.c3d.org/](http://www.c3d.org/)).

        There is a very detailed [technical implementation manual of the C3D file format](http://www.c3d.org/pdf/c3dformat_ug.pdf).

        The C3D file format has three basic components: 

         - Data: at this level the C3D file is simply a binary file that stores raw 3D and analog information.   
         - Standard parameters: default information about the raw 3D and analog data that is required to access the data.   
         - Custom parameters: information specific to a particular manufacturers’ software application or test subject.   

        Regarding to what is useful to the analysis, a C3D file normally has four types of information:

         - 3D point: 3D coordinates of markers or any related biomechanical quantities such as angles, joint forces and moments, etc.
         - Analaog: analog data acquired with an A/D converter.
         - Event: specific frames as events of the acquisition.
         - Metadata: information about the subject, the system configuration, etc.
 
        All C3D files contain a minimum of three sections of information:

         - A single, 512 byte, header section.
         - A parameter section consisting of one, or more, 512-byte blocks. 
         - 3D point/analog data section consisting of one, or more, 512-byte blocks.
 
        To demonstrate the process of reading a binary file, let's read only part of the header of a C3D file:
        """
    )
    return


@app.cell
def _():
    from __future__ import division, print_function
    from struct import unpack       # convert C structs represented as Python strings
    from cStringIO import StringIO  # reads and writes a string buffer

    def getFloat(floatStr,processor):
        "16-bit float string to number conversion"
        if processor > 1: #DEC or SGI
            floatNumber = unpack('f',floatStr[2:] + floatStr[0:2])[0]/4
        else:
            floatNumber = unpack('f',floatStr)[0]
        return floatNumber   

    filename = './../data/Gait.c3d'  # data from sample03.zip @ http://www.c3d.org/sampledata.html
    fid = open(filename, 'rb')       # open file for reading in binary format
    #Header section:
    bytes = fid.read(512)
    buf = StringIO(bytes)
    firstBlockParameterSection, fmt = unpack('BB', buf.read(2))
    if fmt != 80:
        print('This file is not a valid C3D file.')
        fid.close()
    # First block of parameter section:
    firstBlockByte = 512*(firstBlockParameterSection - 1) + 2
    fid.seek(firstBlockByte)
    nparameter_blocks, processor = unpack('BB', fid.read(2))
    processor = processor - 83
    processors = ['unknown', 'Intel', 'DEC', 'SGI']
    #Back to the header section:
    n3Dpoints, = unpack('H', buf.read(2))
    nTotalAnalogDataPerFrame, = unpack('H', buf.read(2))
    nFirstFrame3D, = unpack('H', buf.read(2))
    nLastFrame3D, = unpack('H', buf.read(2))
    maxInterpGap, = unpack('H', buf.read(2))
    scaleFactor = getFloat(buf.read(4), processor)
    dataStartBlock, = unpack('H', buf.read(2))
    nAnalogDataPerFrame, = unpack('H', buf.read(2))
    frameRate = getFloat(buf.read(4), processor)

    print('File "%s":' %filename)
    print('Number of 3D points: %d' % n3Dpoints)
    print('Number of frames: %d' %(nLastFrame3D - nFirstFrame3D + 1))
    print('Frame rate: %.2f' % frameRate)
    print('Analog data per frame: %d' % nAnalogDataPerFrame)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To read a complete C3D file, and fast enough, is not simple. Fortunately, there is a Python package to work with C3D files: the [BTK library](https://code.google.com/p/b-tk/). The same developer of BTK, Arnaud Barré, also wrote another nice software: [Mokka](http://b-tk.googlecode.com/svn/web/mokka/index.html), an *'open-source and cross-platform software to easily analyze biomechanical data'*.   

        [See this page on how to install the BTK library for Python](https://code.google.com/p/b-tk/wiki/PythonBinaries).   

        Let's use BTK to read a C3D file.   
        There is a different version of BTK for each OS; here is a workaround to import the right BTK library according to your OS:
        """
    )
    return


@app.cell
def _():
    import platform, sys

    if platform.system() == 'Windows':
        if sys.maxsize > 2**32:
            sys.path.insert(1, r'./../functions/btk/win64')
        else:
            sys.path.insert(1, r'./../functions/btk/win32')
    elif platform.system() == 'Linux':
        if sys.maxsize > 2**32:
            sys.path.insert(1, r'./../functions/btk/linux64')
        else:
            sys.path.insert(1, r'./../functions/btk/linux32')
    elif platform.system() == 'Darwin':
        sys.path.insert(1, r'./../functions/btk/mac')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The BTK files above were taken from [https://code.google.com/p/b-tk/wiki/PythonBinaries](https://code.google.com/p/b-tk/wiki/PythonBinaries) with the exception of the files for the Mac OS. I compiled these files for the latest Mac OS version (10.9) and Python 2.7.6. Use the Mac files from the BTK website if you have older Mac OS and Python versions.

        With the BTK files and the path to them in the Python path, to use them it's easy:
        """
    )
    return


@app.cell
def _():
    import btk
    return (btk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Read a C3D file:
        """
    )
    return


@app.cell
def _(btk):
    reader = btk.btkAcquisitionFileReader()  # build a btk reader object 
    reader.SetFilename("./../data/Gait.c3d") # set a filename to the reader
    acq = reader.GetOutput()                 # btk aquisition object
    acq.Update()                             # Update ProcessObject associated with DataObject
    return (acq,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Get information about the C3D file content:
        """
    )
    return


@app.cell
def _(acq):
    print('Acquisition duration: %.2f s' %acq.GetDuration()) 
    print('Point frequency: %.2f Hz' %acq.GetPointFrequency())
    print('Number of frames: %d' %acq.GetPointFrameNumber())
    print('Point unit: %s' %acq.GetPointUnit())
    print('Analog frequency: %.2f Hz' %acq.GetAnalogFrequency())
    print('Number of analog channels: %d' %acq.GetAnalogNumber()) 
    print('Number of events: %d' %acq.GetEventNumber())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Get the labels of the 3D points and analog channels:
        """
    )
    return


@app.cell
def _(acq):
    print('Marker labels:')
    for _i in range(0, acq.GetPoints().GetItemNumber()):
        print(acq.GetPoint(_i).GetLabel(), end='  ')
    print('\n\nAnalog channels:')
    for _i in range(0, acq.GetAnalogs().GetItemNumber()):
        print(acq.GetAnalog(_i).GetLabel(), end='  ')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Get all events:
        """
    )
    return


@app.cell
def _(acq):
    for _i in range(0, acq.GetEvents().GetItemNumber()):
        print(acq.GetEvent(_i).GetLabel() + ' at frame %d' % acq.GetEvent(_i).GetFrame())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This file doesn't have any event.

        Get all metadata:
        """
    )
    return


@app.cell
def _(acq):
    for _i in range(acq.GetMetaData().GetChildNumber()):
        print(acq.GetMetaData().GetChild(_i).GetLabel() + ':')
        for j in range(acq.GetMetaData().GetChild(_i).GetChildNumber()):
            print(acq.GetMetaData().GetChild(_i).GetChild(j).GetLabel(), end='  ')
        print('\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Get the 3D position data of all markers as a Numpy array:
        """
    )
    return


@app.cell
def _(acq):
    import numpy as np
    data = np.empty((3, acq.GetPointFrameNumber(), 1))
    for _i in range(0, acq.GetPoints().GetItemNumber()):
        label = acq.GetPoint(_i).GetLabel()
        data = np.dstack((data, acq.GetPoint(label).GetValues().T))
    data = data.T
    data = np.delete(data, 0, axis=0)
    data[data == 0] = np.NaN
    return data, np


@app.cell
def _(data):
    data.shape
    return


@app.cell
def _(data, np):
    print(np.nanmin(data), np.nanmax(data))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are 33 markers (with x, y, z coordinates) and 487 frames.   
        Let's visualize these data.

        First, the vertical ground reaction force:
        """
    )
    return


@app.cell
def _(acq, np):
    import matplotlib.pyplot as plt
    # '%matplotlib inline' command supported automatically in marimo

    ana = acq.GetAnalog("f1z")
    fz1 = ana.GetValues()/ana.GetScale()
    ana = acq.GetAnalog("f2z")
    fz2 = ana.GetValues()/ana.GetScale()
    freq2 = acq.GetAnalogFrequency()
    t = np.linspace(1, len(fz1), num=len(fz1))/freq2
    plt.figure(figsize=(10, 4))
    plt.plot(t, fz1, label='Fz1')
    plt.plot(t, fz2, label='Fz2')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('GRF vertical [N]')
    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's visualize the markers positions in a 3D animation:
        """
    )
    return


@app.cell
def _():
    # Use the IPython magic %matplotlib qt to plot a figure in a separate window
    #  because the Ipython Notebook doesn't play inline matplotlib animations yet.
    # '%matplotlib qt' command supported automatically in marimo
    return


@app.cell
def _(acq, data, np, plt):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    dat = data[:, 130:340, :]
    freq = acq.GetPointFrequency()
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.view_init(10, 150)
    pts = []
    for _i in range(dat.shape[0]):
        pts += ax.plot([], [], [], 'o')
    ax.set_xlim3d([np.nanmin(dat[:, :, 0]), np.nanmax(dat[:, :, 0])])
    ax.set_ylim3d([np.nanmin(dat[:, :, 1]) - 400, np.nanmax(dat[:, :, 1]) + 400])
    ax.set_zlim3d([np.nanmin(dat[:, :, 2]), np.nanmax(dat[:, :, 2])])
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    def animate(i):
        for (pt, xi) in zip(pts, dat):
            (x, y, z) = xi[:_i].T
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])
        return pts
    anim = animation.FuncAnimation(fig, func=animate, frames=dat.shape[1], interval=1000 / freq, blit=True)
    plt.show()
    return (anim,)


@app.cell
def _():
    # get back the inline plot
    # '%matplotlib inline' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This is the animation you see in a separate window when you run the code above:

        <div class='center-align'><figure><img src='https://github.com/BMClab/BMC/blob/master/images/walking.gif?raw=1' width=500 alt='walking'/> </figure></div><br>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To save the animation generated above as a video file (you need to have [FFmpeg](http://www.ffmpeg.org/) or [ImageMagick](http://www.imagemagick.org/script/index.php) installed):
        """
    )
    return


@app.cell
def _(anim):
    anim.save('walking.mp4', writer='ffmpeg', fps=50)
    # or
    anim.save('walking.gif', writer='imagemagick', fps=50)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can also write to a C3D file using BTK; [look at its webpage](http://b-tk.googlecode.com/svn/doc/Python/0.2/_getting_started.html).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
