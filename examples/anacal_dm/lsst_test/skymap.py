config.skyMap = "discrete"

# Position of M31 (Andromeda)
config.skyMap["discrete"].raList = [10.685]  # degrees
config.skyMap["discrete"].decList = [41.269]  # degrees
config.skyMap["discrete"].radiusList = [4.5]  # degrees
config.skyMap["discrete"].pixelScale = 0.2  # LSST arcsec/pixel
config.skyMap["discrete"].patchInnerDimensions = [5000, 5000]  # in pixels
config.skyMap["discrete"].projection = "TAN"
config.skyMap["discrete"].tractOverlap = 0
config.name = "lsst_sim"
