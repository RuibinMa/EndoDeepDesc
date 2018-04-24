# Author: True Price <jtprice at cs.unc.edu>

import numpy as np

#-------------------------------------------------------------------------------
#
# Camera
#
#-------------------------------------------------------------------------------

class Camera:
    @staticmethod
    def GetNumParams(type_):
        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            return 3
        if type_ == 1 or type_ == 'PINHOLE':
            return 4
        if type_ == 2 or type_ == 'SIMPLE_RADIAL':
            return 4

        # TODO: not supporting other camera types, currently
        raise Exception('Camera type not supported')

    #---------------------------------------------------------------------------

    @staticmethod
    def GetNameFromType(type_):
        if type_ == 0: return 'SIMPLE_PINHOLE'
        if type_ == 1: return 'PINHOLE'
        if type_ == 2: return 'SIMPLE_RADIAL'

        # TODO: not supporting other camera types, currently
        raise Exception('Camera type not supported')

    #---------------------------------------------------------------------------

    def __init__(self, type_, width_, height_, params):
        self.width = width_
        self.height = height_

        if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
            self.fx, self.cx, self.cy = params
            self.fy = self.fx
            self.has_distortion = False
            self.camera_type = 0
        elif type_ == 1 or type_ == 'PINHOLE':
            self.fx, self.fy, self.cx, self.cy = params
            self.has_distortion = False
            self.camera_type = 1
        elif type_ == 2 or type_ == 'SIMPLE_RADIAL':
            self.fx, self.cx, self.cy, self.k1 = params
            self.fy, self.k2, self.p1, self.p2 = self.fx, 0, 0, 0
            self.has_distortion = True
            self.camera_type = 2
        # TODO (True): removed support for OpenCV model
        #elif type_ == 'OPENCV':
        #    self.fx, self.fy, self.cx, self.cy = params[:4]
        #    self.k1, self.k2, self.p1, self.p2 = params[4:]
        #    self.has_distortion = True
        else:
            # TODO: not supporting other camera types, currently
            raise Exception('Camera type not supported')

    #---------------------------------------------------------------------------

    def __str__(self):
        s = (self.GetNameFromType(self.camera_type) + ' ' +
                str(self.width) + ' ' + str(self.height) + ' ' +
                str(self.fx) + ' ')

        if self.camera_type == 1: # PINHOLE
            s += str(self.fy) + ' '

        s += str(self.cx) + ' ' + str(self.cy)

        if self.camera_type == 2: # SIMPLE_RADIAL
            s += ' ' + str(self.k1)

        return s

    #---------------------------------------------------------------------------

    # return the camera parameters in the same order as the colmap output format
    def get_params(self):
        if self.camera_type == 0:
            return np.array((self.fx, self.cx, self.cy))
        if self.camera_type == 1:
            return np.array((self.fx, self.fy, self.cx, self.cy))
        if self.camera_type == 2:
            return np.array((self.fx, self.cx, self.cy, self.k1))

    #---------------------------------------------------------------------------

    # return the camera matrix
    def get_camera_matrix(self):
        return np.array(
            ((self.fx, 0, self.cx), (0, self.fy, self.cy), (0, 0, 1)))

    #---------------------------------------------------------------------------

    # return the inverse camera matrix
    def get_inv_camera_matrix(self):
        inv_fx, inv_fy = 1. / self.fx, 1. / self.fy
        return np.array(((inv_fx, 0, -inv_fx * self.cx),
                         (0, inv_fy, -inv_fy * self.cy),
                         (0, 0, 1)))

    #---------------------------------------------------------------------------

    # return an (x, y) pixel coordinate grid for this camera
    def get_image_grid(self):
        return np.meshgrid(
            (np.arange(self.width)  - self.cx) / self.fx,
            (np.arange(self.height) - self.cy) / self.fy)

    #---------------------------------------------------------------------------

    # x: array of shape (N,2) or (2,)
    # normalized: False if the input points are in pixel coordinates
    # denormalize: True if the points should be put back into pixel coordinates
    def distort_points(self, x, normalized=True, denormalize=True):
        x = np.atleast_2d(x)

        # put the points into normalized camera coordinates
        if not normalized:
            x -= np.array([[self.cx, self.cy]])
            x /= np.array([[self.fx, self.fy]])

        # undistort, if necessary
        if self.has_distortion:
            x = x * (1. + self.k1 * np.square(x).sum(axis=1)[:,np.newaxis])

        if denormalize:
            x *= np.array([[self.fx, self.fy]])
            x += np.array([[self.cx, self.cy]])

        return x

    #---------------------------------------------------------------------------

    # x: array of shape (N,2) or (2,)
    # normalized: False if the input points are in pixel coordinates
    # denormalize: True if the points should be put back into pixel coordinates
    def undistort_points(self, x, normalized=False, denormalize=True):
        x = np.atleast_2d(x)

        # put the points into normalized camera coordinates
        if not normalized:
            x -= np.array([[self.cx, self.cy]])
            x /= np.array([[self.fx, self.fy]])

        xu = x.copy()
            
        # undistort, if necessary
        if self.has_distortion:
            MAX_NUM_NEWTON_ITERATIONS = 20

            # Newton's method to solve for the undistorted point
            Jinv = np.empty((len(xu), 2, 2))

            for _ in xrange(MAX_NUM_NEWTON_ITERATIONS):
                x_sq = xu * xu
                xy = xu[:,0] * xu[:,1]

                # G = xu * (1 + k*rsq) - xd 
                G = xu * (1. + self.k1 * x_sq.sum(axis=1)[:,np.newaxis]) - x

                # inverse Jacobian of G w.r.t. xu
                #     [ 3 * k * xu^2 + k * yu^2 + 1, 2 * k * xu * yu             ]
                # J = [ 2 * k * xu * yu            , 3 * k * yu^2 + k * xu^2 + 1 ]
                #        [ a b ]         [  d -b ]
                # If J = [ c d ], Jinv = [ -c  a ] / (a*d - b*c)
                Jinv[:,0,0] = self.k1 * (3. * x_sq[:,1] + x_sq[:,0]) + 1.
                Jinv[:,0,1] = -2. * self.k1 * xy
                Jinv[:,1,0] = Jinv[:,0,1]
                Jinv[:,1,1] = self.k1 * (3. * x_sq[:,0] + x_sq[:,1]) + 1.
                Jinv /= (
                        Jinv[:,0,0] * Jinv[:,1,1] - Jinv[:,0,1] * Jinv[:,1,0]
                    )[:,np.newaxis,np.newaxis]

                # apply Jinv.G separately for the xu and yu updates
                xu[:,0] -= (Jinv[:,0,:] * G).sum(axis=1)
                xu[:,1] -= (Jinv[:,1,:] * G).sum(axis=1)

        if denormalize:
            xu *= np.array([[self.fx, self.fy]])
            xu += np.array([[self.cx, self.cy]])

        return xu


