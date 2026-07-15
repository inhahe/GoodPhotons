# 01_clean_solids

_Bold, legible closed surfaces. Function-heavy leaves, +/- only, shallow nesting._

Material **gold**, coord-frequency ×1, clip radius 1.6. See `params.txt` for the full generator configuration.

| view | seed | ftrace expr |
|---|---|---|
| ![](shape_001.png) | 1 | `f_kummer_surface_v1(x,y,z,1.118) - f_mitre(x,y,z,1.676) + f_steiners_roman(x,y,z,1.688)` |
| ![](shape_002.png) | 2 | `f_steiners_roman(x,y,z,2.38) - f_devils_curve(x,y,z,1.365) + (f_glob(x,y,z,1.342))` |
| ![](shape_003.png) | 3 | `f_mitre(x,y,z,1.657) + f_glob(x,y,z,2.098) - f_glob(x,y,z,1.35)` |
| ![](shape_004.png) | 4 | `f_ovals_of_cassini(x,y,z,1.996,0.524,0.618,3.77) - f_lemniscate_of_gerono(x,y,z,2.704) - f_cross_ellipsoids(x,y,z,3.873,1.154,1.264,0.541)` |
| ![](shape_005.png) | 5 | `f_cross_ellipsoids(x,y,z,2.072,1.137,0.617,0.502) + f_piriform(x,y,z,2.961) - f_hunt_surface(x,y,z,1.476)` |
| ![](shape_006.png) | 6 | `f_mitre(x,y,z,2.07) - (f_heart(x,y,z,1.562) - f_piriform(x,y,z,2.475))` |
