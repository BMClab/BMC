#!/usr/bin/env python

__author__ = "Marcos Duarte <duartexyz@gmail.com>"
__version__ = "DLT.py v.0.0.2 2023/01/13"

"""
Camera calibration and point reconstruction by direct linear transformation, DLT

The fundamental problem here is to find a mathematical relationship between the 
 coordinates of a 3D point and its projection onto the image plane. The DLT 
 (a linear approximation to this problem) is derived from modeling the object 
 and its projection on the image plane as a pinhole camera situation.
In simplistic terms, using a pinhole camera model, it can be found by similar 
 triangles the following relation between the image coordinates (u,v) and the 3D 
 point (X,Y,Z):
    [ u ]   [ L1  L2  L3  L4 ]   [ X ]
    [ v ] = [ L5  L6  L7  L8 ] * [ Y ]
    [ 1 ]   [ L9 L10 L11 L12 ]   [ Z ]
                                 [ 1 ]
The matrix L is kwnown as the camera matrix or camera projection matrix. For a 
 2D point (X,Y), this matrix is 3x3. In fact, the L12 term (or L9 for 2D DLT) 
 is not independent from the other parameters and then there are only 11 
 (or 8 for 2D DLT) independent parameters in the DLT to be determined through
 the calibration procedure.

There are more accurate (but more complex) algorithms for camera calibration
 that also consider lens distortion. For example, OpenCV and Tsai softwares have
 been ported to Python. However, DLT is classic, simple, and effective (fast)
 for most applications.

About DLT, see: https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
"""

import numpy as np

class DLT2():
    """
    Methods for camera calibration and point reconstruction based on DLT.

    DLT is typically used in two steps: 
    1. Camera calibration: L, err = DLTcalib(nd, xyz, uv). 
    2. Object (point) reconstruction: xyz = DLTrecon(n_dims, n_cams, Ls, uvs)

    The camera calibration step consists in digitizing points with known
     coordinates in the real space and find the camera parameters.
    At least 4 points are necessary for the calibration of a plane (2D DLT)
    and at least 6 points for the calibration of a volume (3D DLT). For the 2D
    DLT, at least one view of the object (points) must be entered. For the
     3D DLT, at least 2 different views of the object (points) must be entered.
    These coordinates (from the object and image(s)) are inputed to the DLTcalib 
     algorithm which estimates the camera parameters (8 for 2D DLT and 11 for
     3D DLT).
    Usually it is used more points than the minimum necessary and the
     overdetermined linear system is solved by a least squares minimization
     algorithm. Here this problem is solved using singular value
     decomposition (SVD).
    With these camera parameters and with the camera(s) at the same position
     of the calibration step, we now can reconstruct the real position of any
     point inside the calibrated space (area for 2D DLT and volume for the
     3D DLT) from the point position(s) viewed by the same fixed camera(s).
    This code can perform 2D or 3D DLT with any number of views (cameras).
    For 3D DLT, at least two views (cameras) are necessary.
    """

    def __init__(self):
        #Nothing special to add here
        pass 
    
    def DLTcalib(self, n_dims, xyz, uv):
        """
        Camera calibration by DLT with known object points and image points

        This code performs 2D or 3D DLT camera calibration with any number of
         views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
         n_dims is the number of dimensions of the object space: 3 for 3D DLT
          and 2 for 2D DLT.
         xyz are the coordinates in the object 3D or 2D space of the
          calibration points.
         uv are the coordinates in the image 2D space of these calibration
          points.
         The coordinates (x,y,z and u,v) are given as columns and the different
          points as rows.
         For the 2D DLT (object planar space), only the first 2 columns
          (x and y) are used.
         There must be at least 6 calibration points for the 3D DLT and 4
          for the 2D DLT.
        Outputs:
         L: array of the 8 or 11 parameters of the calibration matrix.
         err: error of the DLT (mean residual of the DLT transformation in units 
          of camera coordinates).
        """
        
        # Convert all variables to numpy array:
        xyz = np.asarray(xyz)
        uv = np.asarray(uv)
        # Number of points:
        n_points = xyz.shape[0]
        # Check the parameters:
        if uv.shape[0] != n_points:
            raise ValueError('xyz (%d points) and uv (%d points) have different number of points.'
                             %(n_points, uv.shape[0]))
        if (n_dims == 2 and xyz.shape[1] != 2) or (n_dims == 3 and xyz.shape[1] != 3):
            raise ValueError('Incorrect number of coordinates (%d) for %dD DLT (it should be %d).'
                             %(xyz.shape[1], n_dims, n_dims))
        if n_dims == 3 and n_points < 6 or n_dims == 2 and n_points < 4:
            raise ValueError('%dD DLT requires at least %d calibration points. Only %d points were entered.'
                             %(n_dims, 2*n_dims, n_points))
            
        # Normalize the data to improve the DLT quality (DLT is dependent on the
        #  system of coordinates).
        # This is relevant when there is a considerable perspective distortion.
        # Normalization: mean position at origin and mean distance equals to 1 
        #  at each direction.
        Txyz, xyzn = self.Normalization(n_dims, xyz)
        Tuv, uvn = self.Normalization(2, uv)
        # Formulating the problem as a set of homogeneous linear equations, M*p=0:
        A = []
        if n_dims == 2: #2D DLT
            for i in range(n_points):
                x, y = xyzn[i, 0], xyzn[i, 1]
                u, v = uvn[i, 0], uvn[i, 1]
                A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
                A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )
        elif n_dims == 3: #3D DLT
            for i in range(n_points):
                x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
                u, v = uvn[i, 0], uvn[i, 1]
                A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
                A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )
     
        # Convert A to array: 
        A = np.asarray(A) 
        # Find the 11 (or 8 for 2D DLT) parameters: 
        U, S, Vh = np.linalg.svd(A)
        # The parameters are in the last line of Vh and normalize them: 
        L = Vh[-1, :] / Vh[-1, -1]
        # Camera projection matrix: 
        H = L.reshape(3, n_dims+1)
        # Denormalization: 
        H = np.dot( np.dot( np.linalg.pinv(Tuv), H ), Txyz )
        H = H / H[-1, -1]
        L = H.flatten()
        # Mean error of the DLT (mean residual of the DLT transformation in 
        #  units of camera coordinates): 
        uv2 = np.dot( H, np.concatenate( (xyz.T, np.ones((1, xyz.shape[0]))) ) ) 
        uv2 = uv2 / uv2[2,:] 
        # Mean distance: 
        err = np.sqrt( np.mean(np.sum( (uv2[0:2, :].T - uv)**2, 1 )) ) 

        return L, err


    def DLTrecon(self, n_dims, n_cams, Ls, uvs):
        """
        Reconstruction of object point from image point(s) based on the DLT parameters.

        This code performs 2D or 3D DLT point reconstruction with any number of
         views (cameras).
        For 3D DLT, at least two views (cameras) are necessary.
        Inputs:
         n_dims is the number of dimensions of the object space: 3 for 3D DLT
          and 2 for 2D DLT.
         n_cams is the number of cameras (views) used.
         Ls (array type) are the camera calibration parameters of each camera 
          (is the output of DLTcalib function). The Ls parameters are given
           as columns and the Ls for different cameras as rows.
         uvs are the coordinates of the point in the image 2D space of each camera.
          The coordinates of the point are given as columns and the different
           views as rows.
        Outputs:
         xyz: point coordinates in space.
        """
        
        # Convert Ls to array: 
        Ls = np.asarray(Ls) 
        # Check the parameters: 
        if Ls.ndim == 1 and n_cams != 1:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (1) are different.'
                             %(n_cams))    
        if Ls.ndim > 1 and n_cams != Ls.shape[0]:
            raise ValueError('Number of views (%d) and number of sets of camera calibration parameters (%d) are different.'
                             %(n_cams, Ls.shape[0]))
        if n_dims == 3 and Ls.ndim == 1:
            raise ValueError('At least two sets of camera calibration parametersd are neede for 3D point reconstruction.')

        if n_cams == 1: # 2D and 1 camera (view), the simplest (and fastest) case. 
            # One could calculate inv(H) and input that to the code to speed up 
            #  things if needed. 
            Hinv = np.linalg.inv( Ls.reshape(3,3) )
            # Point coordinates in space:
            xyz = np.dot( Hinv, [uvs[0], uvs[1],1] ) 
            xyz = xyz[0:2] / xyz[2]       
        else:
            # Formulate problem as a set of homogeneous linear equations, A*p=0: 
            M = []
            for i in range(n_cams):
                L = Ls[i, :]
                u, v = uvs[i][0], uvs[i][1] # indexing works for both list and numpy array 
                if n_dims == 2:      
                    M.append( [L[0]-u*L[6], L[1]-u*L[7], L[2]-u*L[8]] )
                    M.append( [L[3]-v*L[6], L[4]-v*L[7], L[5]-v*L[8]] )
                elif n_dims == 3:  
                    M.append( [L[0]-u*L[8], L[1]-u*L[9], L[2]-u*L[10], L[3]-u*L[11]] )
                    M.append( [L[4]-v*L[8], L[5]-v*L[9], L[6]-v*L[10], L[7]-v*L[11]] )
            
            # Find the xyz coordinates: 
            U, S, Vh = np.linalg.svd(np.asarray(M))
            # Point coordinates in space: 
            xyz = Vh[-1,0:-1] / Vh[-1,-1]
        
        return xyz


    def Normalization(self, n_dims, x):
        """Normalization of coordinates (centroid to the origin and mean distance of sqrt(2 or 3)).

        Inputs:
         n_dims: number of dimensions (2 for 2D; 3 for 3D)
         x: the data to be normalized (directions at different columns and points at rows)
        Outputs:
         Tr: the transformation matrix (translation plus scaling)
         x: the transformed data
        """

        x = np.asarray(x)
        m, s = np.mean(x, 0), np.std(x)
        if n_dims == 2:
            Tr = np.array([[s, 0, m[0]],
                           [0, s, m[1]],
                           [0, 0,   1]])
        else:
            Tr = np.array([[s, 0, 0, m[0]],
                           [0, s, 0, m[1]],
                           [0, 0, s, m[2]],
                           [0, 0, 0,   1]])
            
        Tr = np.linalg.inv(Tr)
        x = np.dot( Tr, np.concatenate( (x.T, np.ones((1,x.shape[0]))) ) )
        x = x[0:n_dims, :].T

        return Tr, x


    def test(self):
        # Tests of DLT
        print('\nTEST #1')
        print('Test of camera calibration and point reconstruction based on'\
              ' direct linear transformation (DLT).')
        print('3D (x, y, z) coordinates (in cm) of the corner of a cube:')
        xyz = [[   0,    0,    0],
               [   0, 12.3,    0],
               [14.5, 12.3,    0],
               [14.5,    0,    0],
               [   0,    0, 14.5],
               [   0, 12.3, 14.5],
               [14.5, 12.3, 14.5],
               [14.5,    0, 14.5]]
        print(np.asarray(xyz))
        print('2D (u, v) coordinates (in pixels) of 4 different views of the cube:')
        uv1 = [[1302, 1147],
               [1110,  976],
               [1411,  863],
               [1618, 1012],
               [1324,  812],
               [1127,  658],
               [1433,  564], 
               [1645,  704]]
        uv2 = [[1094, 1187],
               [1130,  956],
               [1514,  968],
               [1532, 1187],
               [1076,  854],
               [1109,  647],
               [1514,  659],
               [1523,  860]]
        uv3 = [[1073,  866],
               [1319,  761],
               [1580,  896],
               [1352, 1016],
               [1064,  545],
               [1304,  449],
               [1568,  557],
               [1313,  668]]
        uv4 = [[1205, 1511],
               [1193, 1142],
               [1601, 1121],
               [1631, 1487],
               [1157, 1550],
               [1139, 1124],
               [1628, 1100],
               [1661, 1520]]
        print('uv1:')
        print(np.asarray(uv1))
        print('uv2:')
        print(np.asarray(uv2))
        print('uv3:')
        print(np.asarray(uv3))
        print('uv4:')
        print(np.asarray(uv4))

        print('\nUse 4 views to perform a 3D calibration of the camera with 8 points of the cube:')
        n_dims = 3
        n_cams = 4
        L1, err1 = self.DLTcalib(n_dims, xyz, uv1)
        print('Camera calibration parameters based on view #1:')
        print(L1)
        print('Error of the calibration of view #1 (in pixels):')
        print(err1)
        L2, err2 = self.DLTcalib(n_dims, xyz, uv2)
        print('Camera calibration parameters based on view #2:')
        print(L2)
        print('Error of the calibration of view #2 (in pixels):')
        print(err2)
        L3, err3 = self.DLTcalib(n_dims, xyz, uv3)
        print('Camera calibration parameters based on view #3:')
        print(L3)
        print('Error of the calibration of view #3 (in pixels):')
        print(err3)
        L4, err4 = self.DLTcalib(n_dims, xyz, uv4)
        print('Camera calibration parameters based on view #4:')
        print(L4)
        print('Error of the calibration of view #4 (in pixels):')
        print(err4)
        xyz1234 = np.zeros((len(xyz), 3))
        L1234 = [L1, L2, L3, L4]
        for i in range(len(uv1)):
            xyz1234[i, :] = self.DLTrecon( n_dims, n_cams, L1234, [uv1[i], uv2[i], uv3[i], uv4[i]] )
        print('Reconstruction of the same 8 points based on 4 views and the camera calibration parameters:')
        print(xyz1234)
        print('Mean error of the point reconstruction using the DLT (error in cm):')
        print(np.mean(np.sqrt(np.sum((np.array(xyz1234) - np.array(xyz))**2, 1))))

        print('\nTEST #2')
        print('Test of the 2D DLT')
        print('2D (x, y) coordinates (in cm) of the corner of a square:')
        xy = [[   0,    0],
              [   0, 12.3],
              [14.5, 12.3],
              [14.5,    0]]
        print(np.asarray(xy))
        print('2D (u, v) coordinates (in pixels) of 2 different views of the square:')
        uv1 = [[1302, 1147],
               [1110,  976],
               [1411,  863],
               [1618, 1012]]
        uv2 = [[1094, 1187],
               [1130,  956],
               [1514,  968],
               [1532, 1187]]
        print('uv1:')
        print(np.asarray(uv1))
        print('uv2:')
        print(np.asarray(uv2))
        print('')
        print('Use 2 views to perform a 2D calibration of the camera with 4 points of the square:')
        n_dims = 2
        n_cams = 2
        L1, err1 = self.DLTcalib(n_dims, xy, uv1)
        print('Camera calibration parameters based on view #1:')
        print(L1)
        print('Error of the calibration of view #1 (in pixels):')
        print(err1)
        L2, err2 = self.DLTcalib(n_dims, xy, uv2)
        print('Camera calibration parameters based on view #2:')
        print(L2)
        print('Error of the calibration of view #2 (in pixels):')
        print(err2)
        xy12 = np.zeros((len(xy), 2))
        L12 = [L1, L2]
        for i in range(len(uv1)):
            xy12[i, :] = self.DLTrecon( n_dims, n_cams, L12, [uv1[i], uv2[i]] )
        print('Reconstruction of the same 4 points based on 2 views and the camera calibration parameters:')
        print(xy12)
        print('Mean error of the point reconstruction using the DLT (error in cm):')
        print(np.mean(np.sqrt(np.sum((np.array(xy12) - np.array(xy))**2, 1))))

        print('\nTEST #3')
        print('Use only one view to perform a 2D calibration of the camera'\
              ' with 4 points of the square:')
        n_dims = 2
        n_cams = 1
        L1, err1 = self.DLTcalib(n_dims, xy, uv1)
        print('Camera calibration parameters based on view #1:')
        print(L1)
        print('Error of the calibration of view #1 (in pixels):')
        print(err1)
        xy1 = np.zeros((len(xy), 2))
        for i in range(len(uv1)):
            xy1[i,:] = self.DLTrecon( n_dims, n_cams, L1, uv1[i] )
        print('Reconstruction of the same 4 points based on one view and'\
              ' the camera calibration parameters:')
        print(xy1)
        print('Mean error of the point reconstruction using the DLT (error in cm):')
        print(np.mean(np.sqrt(np.sum((np.array(xy1) - np.array(xy))**2, 1))))

if __name__ == '__main__': 
    dlt = DLT2()
    dlt.test()
