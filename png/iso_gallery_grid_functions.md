# Isosurface functions for `iso_gallery_grid.png`

Each grid tile mapped to its exact isosurface definition (material + `function`/CSG body), as rendered.

## 01 — Warped gyroid shell  (`01_gyroid`)

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5
        expr "abs(sin(13.16*x+0.42)*cos(15.59*y) + sin(15.59*y+4.27)*cos(14.53*z) + sin(14.53*z+1.57)*cos(13.16*x) + 0.25*noise(7.85*x, 7.85*y, 7.85*z)) - (0.55 + 0.18*sin(3.01*x+4.27)*cos(3.33*z+1.18))"
    }
    contained_by { min 0.15 0.15 0.15   max 0.85 0.85 0.85 }
    max_gradient 75.9
}
```

## 02 — Chrome lumpy blob  (`02_blobA`)

```
isosurface {
    material chrome
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.3 - (0.072*sin(16.47*x+2.2)*sin(14.37*y+3.53)*sin(14.50*z+4.12) + 0.072*sin(13.63*x+4.73)*sin(9.06*y+5.78)*sin(12.15*z+4.73) + 0.054*sin(10.08*x+3.14)*sin(16.16*y+2.07)*sin(19.15*z+5.13) + 0.064*sin(18.67*x+4.93)*sin(15.22*y+1.22)*sin(15.17*z+2.22) + 0.070*noise(6.59*x+2.875, 6.59*y+2.383, 6.59*z+1.868))"
    }
    contained_by { min 0.08 0.08 0.08   max 0.92 0.92 0.92 }
    max_gradient 22.0
}
```

## 03 — Copper knurled blob  (`03_blobB`)

```
isosurface {
    material copper
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.29 - (0.032*sin(14.11*x+0.21)*sin(11.34*y+1.99)*sin(16.05*z+3.99) + 0.059*sin(17.32*x+4.68)*sin(12.87*y+0.79)*sin(10.18*z+5.96) + 0.056*sin(11.09*x+1.84)*sin(11.71*y+2.63)*sin(18.26*z+3.6) + 0.054*sin(10.60*x+0.35)*sin(18.32*y+1.82)*sin(19.26*z+0.29) + 0.033*sin(21.47*x+5.0)*sin(19.65*y+2.87)*sin(12.23*z+0.09) + 0.060*noise(5.96*x+3.53, 5.96*y+3.942, 5.96*z+1.214))"
    }
    contained_by { min 0.08 0.08 0.08   max 0.92 0.92 0.92 }
    max_gradient 19.8
}
```

## 04 — Warped Schwarz-P  (`04_schwarzP`)

```
isosurface {
    material silver
    function {
        translate 0.5 0.5 0.5
        expr "abs(cos(15.86*x+4.66) + cos(15.33*y+3.3) + cos(13.66*z+0.45) + 0.25*noise(7.13*x, 7.13*y, 7.13*z)) - (0.40 + 0.16*sin(2.69*x+3.11)*cos(3.87*z+1.27))"
    }
    contained_by { min 0.16 0.16 0.16   max 0.84 0.84 0.84 }
    max_gradient 77.2
}
```

## 05 — Brass CSG cluster  (`05_cluster`)

```
isosurface {
    material brass
    difference {
    difference {
    smooth_union { k 0.038
        smooth_union { k 0.050
        smooth_union { k 0.039
        smooth_union { k 0.035
        smooth_union { k 0.032
        smooth_union { k 0.042
        sphere { center 0.642 0.391 0.564  radius 0.137 }
        sphere { center 0.602 0.392 0.536  radius 0.133 }
    }
        torus { translate 0.369 0.629 0.403  rotate 155 167 28  major 0.201  minor 0.041 }
    }
        sphere { center 0.671 0.427 0.395  radius 0.092 }
    }
        sphere { center 0.566 0.601 0.542  radius 0.097 }
    }
        sphere { center 0.365 0.665 0.501  radius 0.128 }
    }
        torus { translate 0.390 0.561 0.494  rotate 152 95 79  major 0.150  minor 0.060 }
    }
    sphere { center 0.435 0.587 0.480  radius 0.070 }
    }
    sphere { center 0.520 0.500 0.495  radius 0.071 }
    }
    contained_by { min 0.12 0.12 0.22   max 0.88 0.88 0.78 }
}
```

## 06 — Iridescent lumpy torus  (`06_torusA`)

```
isosurface {
    material morpho
    function {
        translate 0.5 0.5 0.5
        expr "(sqrt(x^2+y^2) - 0.72)^2 + (z + 0.10*sin(2*atan2(y,x)+0.21))^2 - (0.26 + 0.11*sin(3*atan2(y,x)+1.9) + 0.05*cos(5*atan2(y,x)+4.34))^2"
    }
    contained_by { min 0.12 0.12 0.3   max 0.88 0.88 0.7 }
    max_gradient 9.0
}
```

## 07 — Tilted algebraic heart  (`07_heart`)

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.24   rotate 22 36 -12
        expr "(x^2 + (9/4)*z^2 + y^2 - 1)^3 - x^2*y^3 - (9/80)*z^2*y^3"
    }
    contained_by { min 0.14 0.14 0.2   max 0.86 0.86 0.8 }
}
```

## 08 — Skewed genus-2 torus  (`08_genus2`)

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.42
        expr "2*y*(y^2-3*x^2)*(1-z^2) + (x^2+y^2)^2 - (9*z^2-1)*(1-z^2) + 0.35*x*(x^2+y^2) + 0.20*y*z"
    }
    contained_by { min 0.08 0.16 0.28   max 0.92 0.84 0.72 }
}
```

## 09 — Chrome spiky star  (`09_star`)

```
isosurface {
    material chrome
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.28 - (0.107*sin(9.25*x+2.77)*sin(11.84*y+3.12)*sin(11.22*z+4.78) + 0.061*sin(11.03*x+1.62)*sin(8.83*y+5.59)*sin(8.95*z+3.46) + 0.093*sin(9.06*x+0.85)*sin(10.99*y+2.12)*sin(10.32*z+0.83) + 0.040*noise(5.98*x+2.369, 5.98*y+1.93, 5.98*z+0.683) + 0.05*sin(15.36*x+3.44)*sin(15.29*z+1.48))"
    }
    contained_by { min 0.08 0.08 0.08   max 0.92 0.92 0.92 }
    max_gradient 15.7
}
```

## 10 — Silver metaball drip  (`10_metaballs`)

```
isosurface {
    material silver
    difference {
    smooth_union { k 0.043
        smooth_union { k 0.039
        smooth_union { k 0.030
        smooth_union { k 0.025
        smooth_union { k 0.046
        smooth_union { k 0.028
        sphere { center 0.607 0.552 0.383  radius 0.091 }
        sphere { center 0.485 0.528 0.434  radius 0.117 }
    }
        sphere { center 0.639 0.512 0.550  radius 0.094 }
    }
        sphere { center 0.334 0.404 0.530  radius 0.108 }
    }
        sphere { center 0.356 0.444 0.456  radius 0.108 }
    }
        sphere { center 0.600 0.521 0.593  radius 0.157 }
    }
        torus { translate 0.517 0.486 0.465  rotate 145 83 116  major 0.212  minor 0.043 }
    }
    sphere { center 0.555 0.499 0.482  radius 0.083 }
    }
    contained_by { min 0.1 0.1 0.2   max 0.9 0.9 0.8 }
}
```

## 11 — Steel drilled block  (`11_mech`)

```
isosurface {
    material steel
    difference {
    difference {
    difference {
    difference {
    box { translate 0.5 0.5 0.5  rotate 25 32 13  size 0.42 0.43 0.46  round 0.09 }
    cylinder { translate 0.519 0.479 0.427  rotate 0 0 0  radius 0.093  height 1.2 }
    }
    cylinder { translate 0.536 0.561 0.471  rotate 0 0 90  radius 0.090  height 1.2 }
    }
    cylinder { translate 0.536 0.493 0.554  rotate 0 0 90  radius 0.098  height 1.2 }
    }
    cylinder { translate 0.505 0.427 0.440  rotate 0 0 90  radius 0.087  height 1.2 }
    }
    contained_by { min 0.16 0.16 0.16   max 0.84 0.84 0.84 }
}
```

## 12 — Oil-slick pinched ring  (`12_torusB`)

```
isosurface {
    material oil
    function {
        translate 0.5 0.5 0.5
        expr "(sqrt(x^2+z^2) - 0.70)^2 + (y + 0.12*cos(2*atan2(z,x)+0.58))^2 - (0.24 + 0.10*sin(4*atan2(z,x)+1.16) + 0.06*sin(7*atan2(z,x)+0.63))^2"
    }
    contained_by { min 0.12 0.28 0.12   max 0.88 0.72 0.88 }
    max_gradient 10.0
}
```

## 13 — Warped diamond TPMS  (`13_diamond`)

```
isosurface {
    material beetle
    function {
        translate 0.5 0.5 0.5
        expr "abs(sin(14.32*x+4.59)*sin(13.54*y)*sin(12.37*z) + cos(14.32*x)*cos(13.54*y)*cos(12.37*z+5.8) + 0.25*noise(7.45*x, 7.45*y, 7.45*z)) - (0.55 + 0.15*sin(2.57*x+4.06)*cos(2.42*z+2.16))"
    }
    contained_by { min 0.16 0.16 0.16   max 0.84 0.84 0.84 }
    max_gradient 130.9
}
```

## 14 — Nacre eroded boulder  (`14_noise`)

```
isosurface {
    material nacre
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.28 - 0.16*noise(5*x+0.39, 5*y+0.416, 5*z+0.585) - 0.08*noise(11*x, 11*y+2.035, 11*z) - 0.05*sin(7.99*x)*cos(5.91*y)"
    }
    contained_by { min 0.08 0.08 0.08   max 0.92 0.92 0.92 }
    max_gradient 12.0
}
```

## 15 — Skewed Goursat quartic  (`15_goursat`)

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.22
        expr "x^4+y^4+z^4 - 1.5*(x^2+y^2+z^2) + 0.568*x*y + 0.493*y*z + 0.423*x*z + 0.332*x - 0.329*y + 0.85"
    }
    contained_by { min 0.12 0.12 0.12   max 0.88 0.88 0.88 }
}
```

