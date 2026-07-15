# Isosurface functions for `iso_gallery_grid_v4_dumbbell.png`

Each grid tile mapped to its exact isosurface definition (material + `function`/CSG body), as rendered.

## 01 — Brass dumbbell-ring  (`01_classic`)

```
isosurface {
    material brass
    union { translate 0.5 0.5 0.5  rotate 0 0 0
    difference {
    smooth_union { k 0.042
        smooth_union { k 0.031
        smooth_union { k 0.032
        sphere { center -0.160 0.000 0.000  radius 0.150 }
        sphere { center 0.160 0.000 0.000  radius 0.150 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.065  height 0.314 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.150  minor 0.045 }
    }
    sphere { center 0.039 0.142 -0.017  radius 0.066 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 02 — Gold uneven weights  (`02_asym`)

```
isosurface {
    material gold
    union { translate 0.5 0.5 0.5  rotate 12 22 -8
    difference {
    smooth_union { k 0.033
        smooth_union { k 0.031
        smooth_union { k 0.035
        sphere { center -0.170 0.000 0.000  radius 0.165 }
        sphere { center 0.170 0.000 0.000  radius 0.110 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.060  height 0.333 }
    }
        torus { translate 0.030 0.000 0.000  rotate 0 0 90  major 0.150  minor 0.045 }
    }
    sphere { center 0.011 0.157 0.021  radius 0.053 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 03 — Silver triple collar  (`03_triple`)

```
isosurface {
    material silver
    union { translate 0.5 0.5 0.5  rotate 16 0 -14
    difference {
    smooth_union { k 0.035
        smooth_union { k 0.044
        smooth_union { k 0.042
        smooth_union { k 0.042
        smooth_union { k 0.047
        sphere { center -0.170 0.000 0.000  radius 0.140 }
        sphere { center 0.170 0.000 0.000  radius 0.140 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.060  height 0.333 }
    }
        torus { translate -0.080 0.000 0.000  rotate 0 0 90  major 0.130  minor 0.040 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.150  minor 0.045 }
    }
        torus { translate 0.080 0.000 0.000  rotate 0 0 90  major 0.130  minor 0.040 }
    }
    sphere { center -0.007 0.133 -0.010  radius 0.056 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 04 — Copper tilted bell  (`04_tilted`)

```
isosurface {
    material copper
    union { translate 0.5 0.5 0.5  rotate 34 18 -24
    difference {
    smooth_union { k 0.034
        smooth_union { k 0.050
        smooth_union { k 0.044
        sphere { center -0.160 0.000 0.000  radius 0.150 }
        sphere { center 0.160 0.000 0.000  radius 0.150 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.065  height 0.314 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.155  minor 0.050 }
    }
    sphere { center 0.003 0.142 0.028  radius 0.053 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 05 — Steel plate barbell  (`05_plates`)

```
isosurface {
    material steel
    union { translate 0.5 0.5 0.5  rotate 10 16 -8
    smooth_union { k 0.033
        smooth_union { k 0.022
        smooth_union { k 0.033
        smooth_union { k 0.022
        smooth_union { k 0.020
        smooth_union { k 0.034
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.050  height 0.399 }
        cylinder { translate -0.118 0.000 0.000  rotate 0 0 90  radius 0.135  height 0.028 }
    }
        cylinder { translate -0.148 0.000 0.000  rotate 0 0 90  radius 0.095  height 0.028 }
    }
        sphere { center -0.194 0.000 0.000  radius 0.055 }
    }
        cylinder { translate 0.118 0.000 0.000  rotate 0 0 90  radius 0.135  height 0.028 }
    }
        cylinder { translate 0.148 0.000 0.000  rotate 0 0 90  radius 0.095  height 0.028 }
    }
        sphere { center 0.194 0.000 0.000  radius 0.055 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 06 — Chrome cross-jack  (`06_jack`)

```
isosurface {
    material chrome
    union { translate 0.5 0.5 0.5  rotate 18 12 -10
    smooth_union { k 0.042
        smooth_union { k 0.032
        smooth_union { k 0.037
        smooth_union { k 0.032
        smooth_union { k 0.046
        smooth_union { k 0.031
        smooth_union { k 0.040
        sphere { center 0.000 0.000 0.000  radius 0.098 }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.050  height 0.333 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 90 0 0  radius 0.050  height 0.333 }
    }
        sphere { center -0.170 0.000 0.000  radius 0.115 }
    }
        sphere { center 0.170 0.000 0.000  radius 0.115 }
    }
        sphere { center 0.000 0.000 -0.170  radius 0.115 }
    }
        sphere { center 0.000 0.000 0.170  radius 0.115 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 0  major 0.150  minor 0.040 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 07 — Gold beaded bar  (`07_beaded`)

```
isosurface {
    material gold
    union { translate 0.5 0.5 0.5  rotate 14 24 -12
    smooth_union { k 0.031
        smooth_union { k 0.028
        smooth_union { k 0.028
        smooth_union { k 0.025
        smooth_union { k 0.026
        smooth_union { k 0.021
        sphere { center -0.190 0.000 0.000  radius 0.120 }
        sphere { center 0.190 0.000 0.000  radius 0.120 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.045  height 0.380 }
    }
        torus { translate -0.118 0.000 0.000  rotate 0 0 90  major 0.103  minor 0.034 }
    }
        torus { translate -0.039 0.000 0.000  rotate 0 0 90  major 0.099  minor 0.033 }
    }
        torus { translate 0.039 0.000 0.000  rotate 0 0 90  major 0.086  minor 0.036 }
    }
        torus { translate 0.118 0.000 0.000  rotate 0 0 90  major 0.098  minor 0.041 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 08 — Copper capsule bell  (`08_capsule`)

```
isosurface {
    material copper
    union { translate 0.5 0.5 0.5  rotate 20 10 -16
    smooth_union { k 0.048
        smooth_union { k 0.042
        smooth_union { k 0.044
        ellipsoid { translate -0.170 0.000 0.000  rotate 0 0 0  radius 0.100 0.085 0.085 }
        ellipsoid { translate 0.170 0.000 0.000  rotate 0 0 0  radius 0.100 0.085 0.085 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.055  height 0.333 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.160  minor 0.055 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 09 — Silver linked rings  (`09_chain`)

```
isosurface {
    material silver
    union { translate 0.5 0.5 0.5  rotate 16 20 -8
    smooth_union { k 0.040
        smooth_union { k 0.039
        smooth_union { k 0.035
        torus { translate -0.085 0.000 0.000  rotate 90 0 0  major 0.150  minor 0.050 }
        torus { translate 0.085 0.000 0.000  rotate 0 0 0  major 0.150  minor 0.050 }
    }
        sphere { center -0.240 0.000 0.000  radius 0.085 }
    }
        sphere { center 0.240 0.000 0.000  radius 0.085 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 10 — Brass twin barbell  (`10_double`)

```
isosurface {
    material brass
    union { translate 0.5 0.5 0.5  rotate 14 10 -10
    smooth_union { k 0.034
        smooth_union { k 0.032
        smooth_union { k 0.040
        smooth_union { k 0.048
        smooth_union { k 0.047
        smooth_union { k 0.033
        smooth_union { k 0.043
        sphere { center -0.160 -0.110 0.000  radius 0.100 }
        sphere { center 0.160 -0.110 0.000  radius 0.100 }
    }
        cylinder { translate 0.000 -0.110 0.000  rotate 0 0 90  radius 0.050  height 0.300 }
    }
        sphere { center -0.160 0.110 0.000  radius 0.100 }
    }
        sphere { center 0.160 0.110 0.000  radius 0.100 }
    }
        cylinder { translate 0.000 0.110 0.000  rotate 0 0 90  radius 0.050  height 0.300 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 0  radius 0.045  height 0.240 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 0  major 0.120  minor 0.040 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 11 — Chrome knurled grip  (`11_knurled`)

```
isosurface {
    material chrome
    union { translate 0.5 0.5 0.5  rotate 18 16 -14
    smooth_union { k 0.013
        smooth_union { k 0.020
        smooth_union { k 0.014
        smooth_union { k 0.014
        smooth_union { k 0.012
        smooth_union { k 0.013
        smooth_union { k 0.012
        smooth_union { k 0.018
        smooth_union { k 0.018
        sphere { center -0.170 0.000 0.000  radius 0.130 }
        sphere { center 0.170 0.000 0.000  radius 0.130 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.050  height 0.333 }
    }
        torus { translate -0.075 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate -0.050 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate -0.025 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate 0.025 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate 0.050 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
        torus { translate 0.075 0.000 0.000  rotate 0 0 90  major 0.068  minor 0.012 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 12 — Gold fat short bell  (`12_fatshort`)

```
isosurface {
    material gold
    union { translate 0.5 0.5 0.5  rotate 22 14 -18
    difference {
    smooth_union { k 0.044
        smooth_union { k 0.043
        smooth_union { k 0.042
        sphere { center -0.115 0.000 0.000  radius 0.170 }
        sphere { center 0.115 0.000 0.000  radius 0.170 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.090  height 0.225 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.150  minor 0.055 }
    }
    sphere { center -0.004 0.162 0.006  radius 0.057 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 13 — Silver long thin bar  (`13_longthin`)

```
isosurface {
    material silver
    union { translate 0.5 0.5 0.5  rotate 12 26 -10
    difference {
    smooth_union { k 0.033
        smooth_union { k 0.044
        smooth_union { k 0.041
        smooth_union { k 0.038
        sphere { center -0.210 0.000 0.000  radius 0.100 }
        sphere { center 0.210 0.000 0.000  radius 0.100 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.035  height 0.412 }
    }
        torus { translate -0.050 0.000 0.000  rotate 0 0 90  major 0.075  minor 0.028 }
    }
        torus { translate 0.050 0.000 0.000  rotate 0 0 90  major 0.075  minor 0.028 }
    }
    sphere { center 0.017 0.095 -0.012  radius 0.068 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 14 — Copper big collar  (`14_bigcollar`)

```
isosurface {
    material copper
    union { translate 0.5 0.5 0.5  rotate 20 12 -16
    difference {
    smooth_union { k 0.037
        smooth_union { k 0.041
        smooth_union { k 0.035
        sphere { center -0.150 0.000 0.000  radius 0.130 }
        sphere { center 0.150 0.000 0.000  radius 0.130 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.050  height 0.294 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.200  minor 0.060 }
    }
    sphere { center 0.038 0.123 -0.013  radius 0.059 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 15 — Morpho dumbbell-ring  (`15_morpho`)

```
isosurface {
    material morpho
    union { translate 0.5 0.5 0.5  rotate 24 18 -18
    difference {
    smooth_union { k 0.034
        smooth_union { k 0.042
        smooth_union { k 0.041
        sphere { center -0.160 0.000 0.000  radius 0.150 }
        sphere { center 0.160 0.000 0.000  radius 0.150 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.065  height 0.314 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 90  major 0.155  minor 0.050 }
    }
    sphere { center -0.013 0.142 -0.015  radius 0.060 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

## 16 — Oil-slick cross-jack  (`16_oiljack`)

```
isosurface {
    material oil
    union { translate 0.5 0.5 0.5  rotate 20 14 -12
    smooth_union { k 0.035
        smooth_union { k 0.048
        smooth_union { k 0.048
        smooth_union { k 0.049
        smooth_union { k 0.035
        smooth_union { k 0.036
        smooth_union { k 0.046
        sphere { center 0.000 0.000 0.000  radius 0.098 }
        cylinder { translate 0.000 0.000 0.000  rotate 0 0 90  radius 0.050  height 0.333 }
    }
        cylinder { translate 0.000 0.000 0.000  rotate 90 0 0  radius 0.050  height 0.333 }
    }
        sphere { center -0.170 0.000 0.000  radius 0.115 }
    }
        sphere { center 0.170 0.000 0.000  radius 0.115 }
    }
        sphere { center 0.000 0.000 -0.170  radius 0.115 }
    }
        sphere { center 0.000 0.000 0.170  radius 0.115 }
    }
        torus { translate 0.000 0.000 0.000  rotate 0 0 0  major 0.150  minor 0.040 }
    }
    }
    contained_by { min 0.06 0.06 0.1   max 0.94 0.94 0.9 }
}
```

