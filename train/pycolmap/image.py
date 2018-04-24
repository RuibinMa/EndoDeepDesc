# Author: True Price <jtprice at cs.unc.edu>

import numpy as np

#-------------------------------------------------------------------------------
#
# Image
#
#-------------------------------------------------------------------------------

class Image:
    def __init__(self, name_, camera_id_, q_, tvec_):
        self.name = name_
        self.camera_id = camera_id_
        self.q = q_
        self.tvec = tvec_

        self.points2D = np.array([], dtype=np.float64) # initally empty
        self.point3D_ids = np.array([], dtype=np.uint64) # initally empty

    #---------------------------------------------------------------------------

    def R(self):
        return self.q.ToR()

    #---------------------------------------------------------------------------

    def C(self):
        return -self.R().T.dot(self.tvec)
