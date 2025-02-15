

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile

'''
Band	Wavelength (nm)	    Resolution	    Use Case
B1	    443	                60m	            Coastal/aerosol, useful for water and atmospheric corrections
B2	    490	                10m	            Blue band, good for water body detection
B3	    560	                10m	            Green band, useful for vegetation mapping
B4	    665	                10m	            Red band, vegetation/chlorophyll absorption
B5	    705	                20m	            Red-edge, useful for vegetation health monitoring
B6	    740	                20m	            Red-edge, sensitive to plant stress
B7	    783	                20m	            Red-edge, useful for land cover classification
B8	    842	                10m	            Near-infrared (NIR), key for vegetation analysis
B8A	    865	                20m	            Narrow NIR, useful for vegetation water content
B9	    945	                60m	            Water vapor absorption, atmospheric correction
B11	    1610	            20m	            SWIR-1, useful for soil moisture and burned areas
B12	    2190	            20m	            SWIR-2, useful for land cover and mineral detection



Class	                Recommended Bands	        Why?
Background	            B2, B3, B4, B8	            General land cover separation
Grassland/Shrubland	    B3, B4, B5, B6, B7, B8A	    Red-edge bands (B5-B8A) are good for vegetation health
Logging	                B4, B5, B8, B11, B12	    SWIR bands (B11, B12) help detect bare soil after tree removal
Mining	                B2, B4, B11, B12	        SWIR (B11, B12) detects minerals, B2 helps with water bodies near mines
Plantation	            B3, B4, B5, B8, B8A	        Similar to vegetation, but more structured patterns
'''
image = tifffile.imread('dataset/train_images/train_175.tif')
image = np.nan_to_num(image)



pad = np.zeros((1024, 50))
imgs = np.zeros((1024, 50))
for c in range(image.shape[2]):
    imgs = np.concatenate([imgs, image[:, :, c], pad], 1)

B3 = image[:, :, 2]
B4 = image[:, :, 3]
B8 = image[:, :, 7]
B11 = image[:, :, 10]
ndvi = (B8 - B4) / (B8 + B4 + 1e-6)   # Vegetation health (good for grassland, plantations)
ndwi = (B3 - B8) / (B8 + B3 + 1e-6) # Water presence (helps in mining detection)
ndbi = (B11 - B8) / (B8 + B11 + 1e-6)
'''
    NDVI > 0.5 → Dense vegetation
    0.2 < NDVI < 0.5 → Grassland/shrubland
    NDVI < 0.2 → Barren land, urban areas, water


    NDWI > 0.3 → Water body
    NDWI < 0.3 → Non-water (soil, vegetation, urban areas)


    NDBI > 0.2 → High likelihood of built-up areas (mines, urban zones, infrastructure).
    NDBI < 0.2 → Vegetation or water.

'''
shrubland = (ndvi > 0.2) & ((ndvi <0.5))

plt.imsave('ndvi.png', ndvi)
plt.imsave('ndwi.png', ndwi)
plt.imsave('ndbi.png', ndbi)
print()

