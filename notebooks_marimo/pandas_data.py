import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # pandas Data

        > Marcos Duarte  
        > Laboratory of Biomechanics and Motor Control ([http://demotu.org/](http://demotu.org/))  
        > Federal University of ABC, Brazil
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    # '%matplotlib notebook' command supported automatically in marimo
    # tk qt notebook inline ipympl
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import sys, os
    sys.path.insert(1, r'./../functions')
    return np, os, pd, plt


@app.cell
def _(os):
    path2 = r'./../../../X/Clau/'
    name = 'WBDS01walkT06mkr.txt'
    fname = os.path.join(path2, name)
    return (fname,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## pandas with one index
        """
    )
    return


@app.cell
def _(fname, np, pd):
    df = pd.read_csv(fname, sep='\t', header=0, index_col=0, dtype=np.float64, engine='c')
    df.columns = df.columns.str.replace('\.', '')
    df.head()
    return (df,)


@app.cell
def _(df, plt):
    _ax = df.plot(y='RASISX', figsize=(8, 3), title='A plot of kinematics')
    _ax.set_ylabel('Position [mm]')
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 1, 0.95])
    return


@app.cell
def _(display, plt):
    def plot_widget(df):
        """general plot widget of a pandas dataframe
        """
        from ipywidgets import widgets
        col_w = widgets.SelectMultiple(options=df.columns, value=[df.columns[0]], description='Column')
        clear_w = widgets.Checkbox(value=True, description='Clear axis')
        container = widgets.HBox(children=[col_w, clear_w])
        display(container)
        (fig, _ax) = plt.subplots(1, 1, figsize=(9, 4))
        if col_w.value:
            df.plot(y=col_w.value[0], ax=_ax)
        plt.tight_layout()
        plt.show()

        def plot(change):
            if clear_w.value:
                _ax.clear()
            for c in col_w.value:
                df.plot(y=c, ax=_ax)
        col_w.observe(plot, names='value')
    return (plot_widget,)


@app.cell
def _(df, plot_widget):
    plot_widget(df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## pandas multiindex

        Data with hierarchical column index ([multiindex](http://pandas.pydata.org/pandas-docs/stable/advanced.html#creating-a-multiindex-hierarchical-index-object)) where columns have multiple levels.
        """
    )
    return


@app.cell
def _(fname, np, pd):
    df_1 = pd.read_csv(fname, sep='\t', header=0, index_col=0, dtype=np.float64, engine='c')
    cols = [s[:-1] for s in df_1.columns.str.replace('\\.', '')]
    df_1.columns = [cols, list('XYZ') * int(df_1.shape[1] / 3)]
    df_1.columns.set_names(names=['Marker', 'Coordinate'], level=[0, 1], inplace=True)
    return (df_1,)


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    df_1['RASIS'].head()
    return


@app.cell
def _(df_1):
    df_1.RASIS.X.head()
    return


@app.cell
def _(df_1):
    df_1.xs('X', level='Coordinate', axis=1).head()
    return


@app.cell
def _(df_1):
    df_1.loc[:, (slice(None), 'X')].head()
    return


@app.cell
def _(df_1):
    df_1.swaplevel(0, 1, axis=1).head()
    return


@app.cell
def _(df_1):
    _ax = df_1.plot(y=('RASIS', 'X'), subplots=True, figsize=(8, 2), rot=0)
    return


@app.cell
def _(df_1, plt):
    _ax = df_1.plot(y='RASIS', subplots=True, sharex=True, figsize=(8, 4), rot=0, title='A plot of kinematics')
    plt.tight_layout(pad=0, h_pad=0, rect=[0, 0, 1, 0.95])
    return


@app.cell
def _(df_1):
    values = df_1.reset_index(drop=False).values
    values[0, :5]
    return


@app.cell
def _(df_1):
    df_1.head()
    return


@app.cell
def _(df_1):
    x = df_1.swaplevel(0, 1, axis=1)
    return (x,)


@app.cell
def _(x):
    x2 = x.unstack(level=-1)
    x2.head()
    return


@app.cell
def _(x):
    x.head()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
