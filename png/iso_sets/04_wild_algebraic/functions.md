# 04_wild_algebraic

_Deep-nested algebraic tangles: all functions, + - * ops, abs/sin/cos wrap._

Material **chrome**, coord-frequency ×1.2, clip radius 1.8. See `params.txt` for the full generator configuration.

| view | seed | ftrace expr |
|---|---|---|
| ![](shape_001.png) | 1 | `f_quartic_saddle(1.2*x,1.2*y,1.2*z,1.118) - f_quartic_paraboloid(1.2*x,1.2*y,1.2*z,2.243) - (f_glob(1.2*x,1.2*y,1.2*z,2.376) * sin(f_piriform(1.2*x,1.2*y,1.2*z,2.484)) - f_piriform(1.2*x,1.2*y,1.2*z,1.109) - cos(-1.177)) + (1.31 * f_cross_ellipsoids(1.2*x,1.2*y,1.2*z,3.519,2.528,0.822,0.88))` |
| ![](shape_002.png) | 2 | `0.229 - cos(f_steiners_roman(1.2*x,1.2*y,1.2*z,1.365)) - f_steiners_roman(1.2*x,1.2*y,1.2*z,0.943) * (0.634 + f_blob2(1.2*x,1.2*y,1.2*z,1.026,3.783,0.791,1.019)) + f_witch_of_agnesi(1.2*x,1.2*y,1.2*z,1.586,0.874) + (abs(-0.496) * f_quartic_saddle(1.2*x,1.2*y,1.2*z,0.957))` |
| ![](shape_004.png) | 4 | `1.2*r + (-1.05 - (-0.008 * -2.243 * (sin(-1.617) * (0.817 * f_nodal_cubic(1.2*x,1.2*y,1.2*z,2.714) - f_klein_bottle(1.2*x,1.2*y,1.2*z,0.776)))))` |
| ![](shape_005.png) | 5 | `abs(f_spikes(1.2*x,1.2*y,1.2*z,3.43,1.137,0.617,1.005,0.98)) * 0.428 * sin(0.596) - (f_torus2(1.2*x,1.2*y,1.2*z,1.399,0.628,0.3) * f_cushion(1.2*x,1.2*y,1.2*z,0.914) + (f_spikes(1.2*x,1.2*y,1.2*z,3.936,1.976,0.809,3.775,1.569) - cos(f_ovals_of_cassini(1.2*x,1.2*y,1.2*z,1.229,0.918,0.889,2.216)) + f_spikes(1.2*x,1.2*y,1.2*z,2.646,1.881,1.495,2.726,1.527)))` |
| ![](shape_006.png) | 6 | `abs(f_cubic_saddle(1.2*x,1.2*y,1.2*z,2.07)) + (f_nodal_cubic(1.2*x,1.2*y,1.2*z,2.792)) + f_paraboloid(1.2*x,1.2*y,1.2*z,1.707) + sin(f_parabolic_torus(1.2*x,1.2*y,1.2*z,0.673,0.547,0.501)) + (f_isect_ellipsoids(1.2*x,1.2*y,1.2*z,2.582,1.328,0.985,0.845) * f_isect_ellipsoids(1.2*x,1.2*y,1.2*z,1.367,2.304,0.731,1.01)) + 2.296 + f_quartic_saddle(1.2*x,1.2*y,1.2*z,1.323)` |
| ![](shape_007.png) | 7 | `-2.322 * (abs(f_heart(1.2*x,1.2*y,1.2*z,1.588)) + (abs(1.2*y) - f_bicorn(1.2*x,1.2*y,1.2*z,1.454,0.642) - 1.2*x - cos(f_kummer_surface_v1(1.2*x,1.2*y,1.2*z,1.246)) - abs(f_paraboloid(1.2*x,1.2*y,1.2*z,2.348)) - (abs(f_cross_ellipsoids(1.2*x,1.2*y,1.2*z,1.19,1.514,1.42,0.868)))))` |
