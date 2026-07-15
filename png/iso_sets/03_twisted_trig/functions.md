# 03_twisted_trig

_Wavy gyroid-like surfaces: all functions, sin/cos unary wrap, higher frequency._

Material **morpho**, coord-frequency ×1.7, clip radius 1.6. See `params.txt` for the full generator configuration.

| view | seed | ftrace expr |
|---|---|---|
| ![](shape_001.png) | 1 | `f_quartic_saddle(1.7*x,1.7*y,1.7*z,1.118) - f_quartic_paraboloid(1.7*x,1.7*y,1.7*z,2.243) - (f_glob(1.7*x,1.7*y,1.7*z,2.376) * sin(f_piriform(1.7*x,1.7*y,1.7*z,2.484)) - f_piriform(1.7*x,1.7*y,1.7*z,1.109) - cos(-0.942))` |
| ![](shape_002.png) | 2 | `0.183 - sin(f_steiners_roman(1.7*x,1.7*y,1.7*z,1.365)) - f_steiners_roman(1.7*x,1.7*y,1.7*z,0.943) * (0.507 + f_blob2(1.7*x,1.7*y,1.7*z,1.026,3.783,0.791,1.019)) + f_witch_of_agnesi(1.7*x,1.7*y,1.7*z,1.586,0.874)` |
| ![](shape_003.png) | 3 | `f_crossed_trough(1.7*x,1.7*y,1.7*z,1.657) + (1.7*r + 1.7*y) + sin(f_nodal_cubic(1.7*x,1.7*y,1.7*z,1.984)) - f_kampyle_of_eudoxus(1.7*x,1.7*y,1.7*z,0.73,0.844,1.252) * f_steiners_roman(1.7*x,1.7*y,1.7*z,2.205)` |
| ![](shape_004.png) | 4 | `1.7*r + (-0.84 - (-0.006 * -1.794 * (sin(-1.293) * 0.654)))` |
| ![](shape_005.png) | 5 | `cos(f_spikes(1.7*x,1.7*y,1.7*z,3.43,1.137,0.617,1.005,0.98)) * 0.342 * sin(0.476) - (f_torus2(1.7*x,1.7*y,1.7*z,1.399,0.628,0.3) - f_cushion(1.7*x,1.7*y,1.7*z,0.914) + (1.7*r))` |
| ![](shape_006.png) | 6 | `cos(f_cubic_saddle(1.7*x,1.7*y,1.7*z,2.07)) + (f_nodal_cubic(1.7*x,1.7*y,1.7*z,2.792)) + cos(f_paraboloid(1.7*x,1.7*y,1.7*z,1.707)) - (1.7*x + f_klein_bottle(1.7*x,1.7*y,1.7*z,1.265)) + f_kummer_surface_v2(1.7*x,1.7*y,1.7*z,1.066,-0.428,1.189,-0.822)` |
