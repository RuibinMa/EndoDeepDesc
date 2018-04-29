import matlab.engine
import numpy as np
eng = matlab.engine.start_matlab()
patches = eng.testTFeat_func()
np_patch = np.array(patches._data).reshape(patches.size, order='F')
print(np_patch[200,20,50])