# Isosurface functions for `iso_gallery_grid_v1_diffuse.png`

The **original** gallery grid (matte diffuse materials: clay / jade / coral / gold-diffuse).
Each grid tile mapped to its exact isosurface definition. Superseded by the metal gallery
(`iso_gallery_grid.png` + `iso_gallery_grid_functions.md`), preserved here for reference.

Shared studio (all tiles): unit Cornell box (white/red/green walls), ceiling area light
`origin 0.30 0.999 0.30  u 0.40  v 0.40  spd preset:bb6500`, camera `eye 0.5 0.52 2.75
look_at 0.5 0.48 0.5  fov_y 38  mode R`, film 320x320.

## 01 — Gyroid shell (TPMS)  (`01_gyroid`)  — material: gold

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5
        expr "abs(sin(15*x)*cos(15*y) + sin(15*y)*cos(15*z) + sin(15*z)*cos(15*x)) - 0.7"
    }
    contained_by { min 0.16 0.16 0.16   max 0.84 0.84 0.84 }
}
```

## 02 — Schwarz P (TPMS)  (`02_schwarzP`)  — material: jade

```
isosurface {
    material jade
    function {
        translate 0.5 0.5 0.5
        expr "abs(cos(14*x) + cos(14*y) + cos(14*z)) - 0.4"
    }
    contained_by { min 0.17 0.17 0.17   max 0.83 0.83 0.83 }
}
```

## 03 — Diamond D (TPMS)  (`03_diamond`)  — material: coral

```
isosurface {
    material coral
    function {
        translate 0.5 0.5 0.5
        expr "abs(sin(13*x)*sin(13*y)*sin(13*z) + sin(13*x)*cos(13*y)*cos(13*z) + cos(13*x)*sin(13*y)*cos(13*z) + cos(13*x)*cos(13*y)*sin(13*z)) - 0.5"
    }
    contained_by { min 0.17 0.17 0.17   max 0.83 0.83 0.83 }
}
```

## 04 — Neovius (TPMS)  (`04_neovius`)  — material: clay

```
isosurface {
    material clay
    function {
        translate 0.5 0.5 0.5
        expr "abs(3*(cos(12*x)+cos(12*y)+cos(12*z)) + 4*cos(12*x)*cos(12*y)*cos(12*z)) - 1.5"
    }
    contained_by { min 0.18 0.18 0.18   max 0.82 0.82 0.82 }
}
```

## 05 — Chmutov (Chebyshev)  (`05_chmutov`)  — material: gold

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.26
        expr "(8*x^4-8*x^2+1) + (8*y^4-8*y^2+1) + (8*z^4-8*z^2+1) + 0.6"
    }
    contained_by { min 0.15 0.15 0.15   max 0.85 0.85 0.85 }
}
```

## 06 — Tanglecube (quartic)  (`06_tanglecube`)  — material: jade

```
isosurface {
    material jade
    function {
        translate 0.5 0.5 0.5   scale 0.20
        expr "x^4 - 5*x^2 + y^4 - 5*y^2 + z^4 - 5*z^2 + 11.8"
    }
    contained_by { min 0.12 0.12 0.12   max 0.88 0.88 0.88 }
}
```

## 07 — Heart (algebraic)  (`07_heart`)  — material: coral

```
isosurface {
    material coral
    function {
        translate 0.5 0.42 0.5   scale 0.34   rotate 0 0 0
        expr "(x^2 + (9/4)*z^2 + y^2 - 1)^3 - x^2*y^3 - (9/80)*z^2*y^3"
    }
    contained_by { min 0.12 0.06 0.20   max 0.88 0.80 0.80 }
}
```

## 08 — Genus-2 double torus  (`08_genus2`)  — material: gold

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.42
        expr "2*y*(y^2-3*x^2)*(1-z^2) + (x^2+y^2)^2 - (9*z^2-1)*(1-z^2)"
    }
    contained_by { min 0.10 0.18 0.30   max 0.90 0.82 0.70 }
}
```

## 09 — Noise-warped blob  (`09_noiseblob`)  — material: clay

```
isosurface {
    material clay
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.30 - 0.11*noise(6*x, 6*y, 6*z) - 0.06*noise(13*x+4, 13*y, 13*z-2)"
    }
    contained_by { min 0.10 0.10 0.10   max 0.90 0.90 0.90 }
    max_gradient 3
}
```

## 10 — Metaball cluster (CSG)  (`10_metaballs`)  — material: jade

```
isosurface {
    material jade
    smooth_union { k 0.09
        smooth_union { k 0.10
            sphere { center 0.40 0.42 0.52  radius 0.15 }
            sphere { center 0.58 0.50 0.46  radius 0.12 }
        }
        smooth_union { k 0.08
            sphere { center 0.50 0.62 0.55  radius 0.10 }
            box    { translate 0.52 0.34 0.44  size 0.10 0.07 0.10  round 0.03 }
        }
    }
    contained_by { min 0.10 0.10 0.20   max 0.90 0.90 0.80 }
}
```

## 11 — CSG mechanical (boolean)  (`11_csg_mech`)  — material: clay

```
isosurface {
    material clay
    difference {
        difference {
            box { translate 0.5 0.5 0.5  size 0.22 0.22 0.22  round 0.05 }
            cylinder { translate 0.5 0.5 0.5  radius 0.10  height 0.9 }
        }
        cylinder { translate 0.5 0.5 0.5  rotate 0 0 90  radius 0.08  height 0.9 }
    }
    contained_by { min 0.20 0.20 0.20   max 0.80 0.80 0.80 }
}
```

## 12 — Lumpy torus  (`12_twist_torus`)  — material: coral

```
isosurface {
    material coral
    function {
        translate 0.5 0.5 0.5   scale 0.30
        expr "(sqrt(x^2+y^2) - 0.78)^2 + z^2 - (0.30 + 0.14*sin(3*atan2(y,x)))^2"
    }
    contained_by { min 0.14 0.14 0.34   max 0.86 0.86 0.66 }
    max_gradient 6
}
```

## 13 — Scherk saddle tower  (`13_scherk`)  — material: clay

```
isosurface {
    material clay
    function {
        translate 0.5 0.5 0.5   scale 0.17
        expr "exp(z)*cos(y) - cos(x)"
    }
    contained_by { min 0.30 0.24 0.30   max 0.70 0.76 0.70 }
    max_gradient 8
}
```

## 14 — Knobbly star sphere  (`14_knobbly`)  — material: jade

```
isosurface {
    material jade
    function {
        translate 0.5 0.5 0.5
        expr "r - 0.30 - 0.10*sin(9*x)*sin(9*y)*sin(9*z)"
    }
    contained_by { min 0.12 0.12 0.12   max 0.88 0.88 0.88 }
    max_gradient 4
}
```

## 15 — Goursat quartic  (`15_goursat`)  — material: gold

```
isosurface {
    material gold
    function {
        translate 0.5 0.5 0.5   scale 0.22
        expr "x^4+y^4+z^4 - 1.5*(x^2+y^2+z^2) + 0.6*(x*y + y*z) + 0.9"
    }
    contained_by { min 0.13 0.13 0.13   max 0.87 0.87 0.87 }
}
```
