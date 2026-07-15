# 02_blended_metal

_Smooth blobby metal combos: blob/ellipsoid/spike families blended with + - *._

Material **copper**, coord-frequency ×1.1, clip radius 1.7. See `params.txt` for the full generator configuration.

| view | seed | ftrace expr |
|---|---|---|
| ![](shape_001.png) | 1 | `f_pillow(1.1*x,1.1*y,1.1*z,1.118) * f_superellipsoid(1.1*x,1.1*y,1.1*z,1.309,1.819) + f_pillow(1.1*x,1.1*y,1.1*z,1.838) * f_blob2(1.1*x,1.1*y,1.1*z,1.027,3.069,1.294,0.741) - (f_ellipsoid(1.1*x,1.1*y,1.1*z,1.453,0.968,1.751))` |
| ![](shape_002.png) | 2 | `1.1*z - f_superellipsoid(1.1*x,1.1*y,1.1*z,1.082,1.029) + (f_glob(1.1*x,1.1*y,1.1*z,1.342) + f_torus(1.1*x,1.1*y,1.1*z,0.85,0.324) * f_torus(1.1*x,1.1*y,1.1*z,0.582,0.342))` |
| ![](shape_005.png) | 5 | `f_isect_ellipsoids(1.1*x,1.1*y,1.1*z,2.072,1.137,0.617,0.502) + f_cross_ellipsoids(1.1*x,1.1*y,1.1*z,3.953,3.853,1.086,1.326) * 0.357 + (f_glob(1.1*x,1.1*y,1.1*z,1.618) - (f_torus(1.1*x,1.1*y,1.1*z,0.615,0.222)))` |
| ![](shape_006.png) | 6 | `f_superellipsoid(1.1*x,1.1*y,1.1*z,1.663,0.531) + f_superellipsoid(1.1*x,1.1*y,1.1*z,1.636,2.313) * 1.1*x - (f_ellipsoid(1.1*x,1.1*y,1.1*z,1.733,0.987,1.234) + f_sphere(1.1*x,1.1*y,1.1*z,0.416))` |
| ![](shape_008.png) | 8 | `f_torus_gumdrop(1.1*x,1.1*y,1.1*z,2.051) * 1.1*x + f_superellipsoid(1.1*x,1.1*y,1.1*z,1.982,0.504) + f_sphere(1.1*x,1.1*y,1.1*z,1.028) - (f_cross_ellipsoids(1.1*x,1.1*y,1.1*z,2.034,2.774,0.537,0.683))` |
| ![](shape_010.png) | 10 | `f_torus_gumdrop(1.1*x,1.1*y,1.1*z,1.159) * f_isect_ellipsoids(1.1*x,1.1*y,1.1*z,3.851,3.761,0.549,0.643) - (f_ellipsoid(1.1*x,1.1*y,1.1*z,0.918,0.801,1.672) + 1.1*x - f_blob2(1.1*x,1.1*y,1.1*z,1.142,2.88,1.437,1.429))` |
