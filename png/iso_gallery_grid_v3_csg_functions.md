# Isosurface functions for `iso_gallery_grid_v3_csg.png`

Each grid tile mapped to its exact isosurface definition (material + `function`/CSG body), as rendered.

## 01 — Brass bulb cluster  (`01_bulb`)

```
isosurface {
    material brass
    difference {
    difference {
    smooth_union { k 0.039
        smooth_union { k 0.045
        smooth_union { k 0.042
        smooth_union { k 0.038
        smooth_union { k 0.025
        smooth_union { k 0.048
        sphere { center 0.642 0.328 0.521  radius 0.094 }
        sphere { center 0.378 0.387 0.636  radius 0.150 }
    }
        torus { translate 0.616 0.400 0.509  rotate 108 43 74  major 0.207  minor 0.055 }
    }
        sphere { center 0.375 0.339 0.588  radius 0.101 }
    }
        sphere { center 0.638 0.355 0.454  radius 0.132 }
    }
        torus { translate 0.519 0.454 0.499  rotate 86 150 69  major 0.170  minor 0.049 }
    }
        sphere { center 0.328 0.351 0.559  radius 0.151 }
    }
    sphere { center 0.452 0.573 0.466  radius 0.076 }
    }
    sphere { center 0.489 0.536 0.507  radius 0.097 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 02 — Gold berry bunch  (`02_berry`)

```
isosurface {
    material gold
    smooth_union { k 0.021
        smooth_union { k 0.028
        smooth_union { k 0.026
        smooth_union { k 0.026
        smooth_union { k 0.028
        smooth_union { k 0.023
        smooth_union { k 0.022
        smooth_union { k 0.029
        smooth_union { k 0.026
        smooth_union { k 0.030
        smooth_union { k 0.021
        sphere { center 0.613 0.497 0.602  radius 0.071 }
        sphere { center 0.500 0.539 0.432  radius 0.089 }
    }
        sphere { center 0.485 0.487 0.598  radius 0.077 }
    }
        sphere { center 0.538 0.363 0.403  radius 0.081 }
    }
        sphere { center 0.395 0.534 0.494  radius 0.098 }
    }
        sphere { center 0.616 0.550 0.528  radius 0.078 }
    }
        sphere { center 0.571 0.457 0.435  radius 0.069 }
    }
        sphere { center 0.545 0.430 0.506  radius 0.075 }
    }
        sphere { center 0.566 0.393 0.583  radius 0.067 }
    }
        sphere { center 0.496 0.460 0.439  radius 0.069 }
    }
        sphere { center 0.441 0.521 0.583  radius 0.072 }
    }
        sphere { center 0.361 0.522 0.507  radius 0.088 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 03 — Silver ring-weave  (`03_rings`)

```
isosurface {
    material silver
    difference {
    difference {
    smooth_union { k 0.022
        smooth_union { k 0.023
        smooth_union { k 0.026
        smooth_union { k 0.040
        smooth_union { k 0.021
        torus { translate 0.536 0.549 0.494  rotate 172 73 150  major 0.178  minor 0.048 }
        torus { translate 0.532 0.545 0.519  rotate 167 18 144  major 0.208  minor 0.051 }
    }
        torus { translate 0.446 0.516 0.528  rotate 115 92 94  major 0.176  minor 0.045 }
    }
        sphere { center 0.500 0.500 0.500  radius 0.081 }
    }
        torus { translate 0.454 0.449 0.499  rotate 176 56 1  major 0.182  minor 0.049 }
    }
        torus { translate 0.477 0.462 0.454  rotate 31 79 166  major 0.188  minor 0.052 }
    }
    sphere { center 0.569 0.437 0.480  radius 0.059 }
    }
    sphere { center 0.505 0.501 0.523  radius 0.063 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 04 — Copper tentacle worm  (`04_worm`)

```
isosurface {
    material copper
    smooth_union { k 0.071
        smooth_union { k 0.061
        smooth_union { k 0.060
        smooth_union { k 0.053
        smooth_union { k 0.056
        smooth_union { k 0.070
        smooth_union { k 0.054
        smooth_union { k 0.070
        smooth_union { k 0.070
        sphere { center 0.549 0.572 0.491  radius 0.130 }
        sphere { center 0.496 0.486 0.432  radius 0.121 }
    }
        sphere { center 0.429 0.423 0.374  radius 0.112 }
    }
        sphere { center 0.331 0.409 0.354  radius 0.103 }
    }
        sphere { center 0.300 0.362 0.340  radius 0.094 }
    }
        sphere { center 0.300 0.343 0.340  radius 0.086 }
    }
        sphere { center 0.300 0.314 0.340  radius 0.077 }
    }
        sphere { center 0.300 0.300 0.340  radius 0.068 }
    }
        sphere { center 0.300 0.300 0.340  radius 0.059 }
    }
        sphere { center 0.300 0.300 0.340  radius 0.050 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 05 — Chrome caltrop  (`05_caltrop`)

```
isosurface {
    material chrome
    smooth_union { k 0.040
        smooth_union { k 0.038
        smooth_union { k 0.030
        smooth_union { k 0.042
        smooth_union { k 0.047
        cylinder { translate 0.500 0.500 0.500  rotate 93 3 174  radius 0.047  height 0.566 }
        cylinder { translate 0.500 0.500 0.500  rotate 47 159 174  radius 0.049  height 0.528 }
    }
        sphere { center 0.500 0.500 0.500  radius 0.124 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 70 39 82  radius 0.047  height 0.527 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 131 102 109  radius 0.051  height 0.544 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 147 117 135  radius 0.039  height 0.576 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 06 — Steel cratered asteroid  (`06_asteroid`)

```
isosurface {
    material steel
    difference {
    difference {
    difference {
    difference {
    difference {
    smooth_union { k 0.038
        smooth_union { k 0.030
        smooth_union { k 0.042
        smooth_union { k 0.038
        smooth_union { k 0.032
        smooth_union { k 0.038
        smooth_union { k 0.035
        smooth_union { k 0.045
        sphere { center 0.332 0.644 0.513  radius 0.083 }
        sphere { center 0.335 0.629 0.573  radius 0.055 }
    }
        sphere { center 0.557 0.296 0.566  radius 0.074 }
    }
        sphere { center 0.559 0.289 0.469  radius 0.060 }
    }
        sphere { center 0.559 0.647 0.655  radius 0.052 }
    }
        sphere { center 0.472 0.532 0.718  radius 0.078 }
    }
        sphere { center 0.392 0.505 0.694  radius 0.066 }
    }
        sphere { center 0.500 0.500 0.500  radius 0.246 }
    }
        sphere { center 0.522 0.285 0.549  radius 0.089 }
    }
    sphere { center 0.327 0.666 0.442  radius 0.089 }
    }
    sphere { center 0.560 0.484 0.262  radius 0.082 }
    }
    sphere { center 0.735 0.481 0.570  radius 0.095 }
    }
    sphere { center 0.649 0.441 0.313  radius 0.072 }
    }
    sphere { center 0.553 0.273 0.581  radius 0.070 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 07 — Gold pebble pile  (`07_pebbles`)

```
isosurface {
    material gold
    smooth_union { k 0.032
        smooth_union { k 0.039
        smooth_union { k 0.031
        smooth_union { k 0.038
        smooth_union { k 0.033
        smooth_union { k 0.037
        smooth_union { k 0.047
        box { translate 0.486 0.408 0.537  rotate 70 106 121  size 0.145 0.114 0.106  round 0.088 }
        sphere { center 0.505 0.482 0.490  radius 0.105 }
    }
        box { translate 0.519 0.477 0.404  rotate 120 157 56  size 0.183 0.179 0.168  round 0.058 }
    }
        box { translate 0.536 0.378 0.439  rotate 172 54 2  size 0.141 0.127 0.121  round 0.058 }
    }
        sphere { center 0.583 0.385 0.505  radius 0.100 }
    }
        box { translate 0.576 0.543 0.451  rotate 50 77 155  size 0.187 0.144 0.140  round 0.078 }
    }
        sphere { center 0.494 0.387 0.489  radius 0.108 }
    }
        box { translate 0.482 0.477 0.599  rotate 162 172 16  size 0.200 0.142 0.190  round 0.088 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 08 — Copper coral  (`08_coral`)

```
isosurface {
    material copper
    smooth_union { k 0.045
        smooth_union { k 0.034
        smooth_union { k 0.038
        smooth_union { k 0.050
        smooth_union { k 0.032
        smooth_union { k 0.042
        smooth_union { k 0.038
        smooth_union { k 0.039
        ellipsoid { translate 0.482 0.639 0.512  rotate 70 125 1  radius 0.061 0.119 0.061 }
        ellipsoid { translate 0.500 0.500 0.500  rotate 0 0 0  radius 0.156 0.130 0.156 }
    }
        ellipsoid { translate 0.591 0.615 0.498  rotate 17 175 20  radius 0.065 0.149 0.063 }
    }
        ellipsoid { translate 0.344 0.526 0.479  rotate 47 48 92  radius 0.065 0.137 0.060 }
    }
        ellipsoid { translate 0.585 0.512 0.626  rotate 113 148 167  radius 0.058 0.116 0.062 }
    }
        ellipsoid { translate 0.396 0.530 0.391  rotate 142 14 132  radius 0.059 0.124 0.067 }
    }
        ellipsoid { translate 0.530 0.462 0.642  rotate 123 144 48  radius 0.061 0.145 0.054 }
    }
        ellipsoid { translate 0.610 0.594 0.543  rotate 160 87 88  radius 0.050 0.136 0.066 }
    }
        ellipsoid { translate 0.434 0.585 0.602  rotate 88 146 74  radius 0.060 0.101 0.059 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 09 — Brass dumbbell-ring  (`09_dumbbell`)

```
isosurface {
    material brass
    difference {
    smooth_union { k 0.046
        smooth_union { k 0.033
        smooth_union { k 0.032
        sphere { center 0.340 0.500 0.500  radius 0.150 }
        sphere { center 0.660 0.500 0.500  radius 0.150 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 0 0 90  radius 0.065  height 0.340 }
    }
        torus { translate 0.500 0.500 0.500  rotate 0 0 90  major 0.150  minor 0.045 }
    }
    sphere { center 0.369 0.528 0.501  radius 0.067 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 10 — Silver fused knuckles  (`10_knuckle`)

```
isosurface {
    material silver
    difference {
    difference {
    smooth_union { k 0.046
        smooth_union { k 0.060
        smooth_union { k 0.048
        sphere { center 0.350 0.445 0.548  radius 0.140 }
        sphere { center 0.450 0.465 0.482  radius 0.136 }
    }
        sphere { center 0.550 0.477 0.498  radius 0.139 }
    }
        sphere { center 0.650 0.465 0.522  radius 0.133 }
    }
    sphere { center 0.426 0.566 0.540  radius 0.072 }
    }
    sphere { center 0.528 0.628 0.493  radius 0.058 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 11 — Gold lobed knot  (`11_knot`)

```
isosurface {
    material gold
    smooth_union { k 0.037
        smooth_union { k 0.033
        smooth_union { k 0.042
        sphere { center 0.500 0.500 0.500  radius 0.100 }
        torus { translate 0.524 0.476 0.503  rotate 0 12 -7  major 0.192  minor 0.053 }
    }
        torus { translate 0.537 0.529 0.481  rotate 85 -10 4  major 0.178  minor 0.047 }
    }
        torus { translate 0.480 0.514 0.523  rotate 11 9 98  major 0.183  minor 0.048 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 12 — Chrome pill heap  (`12_pills`)

```
isosurface {
    material chrome
    smooth_union { k 0.056
        smooth_union { k 0.049
        smooth_union { k 0.053
        smooth_union { k 0.052
        smooth_union { k 0.040
        smooth_union { k 0.041
        smooth_union { k 0.057
        cylinder { translate 0.423 0.528 0.572  rotate 68 28 152  radius 0.072  height 0.157 }
        sphere { center 0.597 0.470 0.530  radius 0.074 }
    }
        sphere { center 0.513 0.493 0.434  radius 0.093 }
    }
        cylinder { translate 0.387 0.410 0.480  rotate 71 139 141  radius 0.075  height 0.150 }
    }
        cylinder { translate 0.517 0.467 0.576  rotate 132 75 115  radius 0.063  height 0.187 }
    }
        cylinder { translate 0.478 0.484 0.561  rotate 140 164 159  radius 0.074  height 0.155 }
    }
        sphere { center 0.439 0.457 0.468  radius 0.086 }
    }
        cylinder { translate 0.616 0.466 0.427  rotate 2 14 66  radius 0.070  height 0.148 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 13 — Morpho bulb cluster  (`13_morpho`)

```
isosurface {
    material morpho
    difference {
    difference {
    smooth_union { k 0.027
        smooth_union { k 0.041
        smooth_union { k 0.031
        smooth_union { k 0.038
        smooth_union { k 0.043
        smooth_union { k 0.048
        sphere { center 0.551 0.661 0.371  radius 0.127 }
        torus { translate 0.478 0.518 0.470  rotate 75 112 142  major 0.214  minor 0.066 }
    }
        sphere { center 0.586 0.570 0.374  radius 0.141 }
    }
        torus { translate 0.431 0.459 0.567  rotate 95 172 103  major 0.200  minor 0.060 }
    }
        sphere { center 0.606 0.614 0.560  radius 0.113 }
    }
        sphere { center 0.619 0.404 0.574  radius 0.103 }
    }
        sphere { center 0.353 0.506 0.510  radius 0.132 }
    }
    sphere { center 0.468 0.434 0.429  radius 0.083 }
    }
    sphere { center 0.549 0.488 0.515  radius 0.083 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 14 — Oil-slick ring-weave  (`14_oil`)

```
isosurface {
    material oil
    difference {
    difference {
    smooth_union { k 0.034
        smooth_union { k 0.021
        smooth_union { k 0.037
        smooth_union { k 0.031
        smooth_union { k 0.040
        torus { translate 0.558 0.520 0.455  rotate 81 70 174  major 0.199  minor 0.044 }
        sphere { center 0.500 0.500 0.500  radius 0.096 }
    }
        torus { translate 0.501 0.518 0.454  rotate 75 48 35  major 0.209  minor 0.053 }
    }
        torus { translate 0.554 0.440 0.520  rotate 132 175 19  major 0.193  minor 0.056 }
    }
        torus { translate 0.512 0.555 0.470  rotate 10 153 8  major 0.212  minor 0.051 }
    }
        torus { translate 0.478 0.466 0.466  rotate 114 36 80  major 0.232  minor 0.045 }
    }
    sphere { center 0.464 0.519 0.471  radius 0.060 }
    }
    sphere { center 0.507 0.453 0.557  radius 0.070 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 15 — Copper sea-urchin  (`15_urchin`)

```
isosurface {
    material copper
    smooth_union { k 0.033
        smooth_union { k 0.034
        smooth_union { k 0.039
        smooth_union { k 0.037
        smooth_union { k 0.036
        smooth_union { k 0.025
        smooth_union { k 0.026
        smooth_union { k 0.026
        smooth_union { k 0.038
        cylinder { translate 0.500 0.500 0.500  rotate 94 147 78  radius 0.031  height 0.604 }
        cylinder { translate 0.500 0.500 0.500  rotate 91 27 40  radius 0.027  height 0.596 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 155 39 40  radius 0.034  height 0.614 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 86 168 154  radius 0.027  height 0.613 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 102 170 143  radius 0.024  height 0.583 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 91 14 90  radius 0.028  height 0.579 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 172 119 101  radius 0.024  height 0.572 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 51 163 12  radius 0.023  height 0.572 }
    }
        sphere { center 0.500 0.500 0.500  radius 0.101 }
    }
        cylinder { translate 0.500 0.500 0.500  rotate 152 22 152  radius 0.023  height 0.634 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

## 16 — Silver grape bunch  (`16_grapes`)

```
isosurface {
    material silver
    smooth_union { k 0.021
        smooth_union { k 0.033
        smooth_union { k 0.026
        smooth_union { k 0.028
        smooth_union { k 0.028
        smooth_union { k 0.026
        smooth_union { k 0.025
        smooth_union { k 0.020
        smooth_union { k 0.022
        smooth_union { k 0.030
        smooth_union { k 0.024
        smooth_union { k 0.023
        ellipsoid { translate 0.473 0.407 0.503  rotate 64 109 132  radius 0.063 0.083 0.082 }
        ellipsoid { translate 0.530 0.584 0.579  rotate 49 176 106  radius 0.066 0.074 0.061 }
    }
        ellipsoid { translate 0.390 0.526 0.525  rotate 66 152 118  radius 0.077 0.071 0.079 }
    }
        ellipsoid { translate 0.545 0.336 0.492  rotate 62 104 153  radius 0.077 0.070 0.084 }
    }
        ellipsoid { translate 0.471 0.382 0.543  rotate 20 62 137  radius 0.074 0.077 0.081 }
    }
        ellipsoid { translate 0.422 0.571 0.493  rotate 14 157 156  radius 0.068 0.085 0.083 }
    }
        ellipsoid { translate 0.641 0.636 0.360  rotate 101 178 18  radius 0.063 0.085 0.075 }
    }
        ellipsoid { translate 0.403 0.560 0.598  rotate 125 87 131  radius 0.082 0.075 0.075 }
    }
        ellipsoid { translate 0.504 0.539 0.583  rotate 109 108 63  radius 0.083 0.081 0.084 }
    }
        ellipsoid { translate 0.540 0.554 0.369  rotate 115 10 95  radius 0.075 0.087 0.066 }
    }
        ellipsoid { translate 0.511 0.547 0.554  rotate 153 55 104  radius 0.080 0.094 0.077 }
    }
        ellipsoid { translate 0.438 0.441 0.477  rotate 120 122 25  radius 0.076 0.086 0.060 }
    }
        ellipsoid { translate 0.476 0.455 0.450  rotate 76 120 156  radius 0.060 0.071 0.071 }
    }
    contained_by { min 0.08 0.08 0.12   max 0.92 0.92 0.88 }
}
```

