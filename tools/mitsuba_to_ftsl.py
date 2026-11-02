#!/usr/bin/env python3
"""Mitsuba (0.6 / 2 / 3) XML scene -> FTSL converter.

Mitsuba and this renderer are both spectral, physically-based path tracers, so most
concepts map almost 1:1: perspective/thinlens sensor -> FTSL camera, area/constant/
envmap emitters -> FTSL lights, and diffuse/conductor/dielectric/plastic BSDFs ->
FTSL materials. That makes Mitsuba XML the natural import target (and Blender can
export straight to it via the mitsuba-blender addon, so this doubles as a Blender
path). RGB reflectances ride FTSL's Jakob-Hanika `rgb` upsampling and measured spectra
pass through; emitters keep Mitsuba's magnitudes (see Converter.emitter_spd).

Usage:
    python tools/mitsuba_to_ftsl.py scene.xml [out.ftsl]

Unsupported constructs are skipped with a `# WARN:` comment in the output and a note
on stderr, so the result always parses. Coverage is the common subset an exported
scene uses; exotic BSDFs/shapes degrade to a documented approximation.
"""
import sys, os, math, re
import xml.etree.ElementTree as ET

# --------------------------------------------------------------------------- utils
warnings = []
def warn(msg):
    warnings.append(msg)

def fnum(x):
    """Format a float compactly (trim trailing zeros), keeping enough precision."""
    s = f"{x:.6g}"
    return s

# ------------------------------------------------------------------- 4x4 matrices
# Row-major 4x4 as a flat list of 16. Points are column vectors: p' = M @ p.
def mat_identity():
    return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]

def mat_mul(a, b):
    r = [0.0]*16
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i*4+k]*b[k*4+j]
            r[i*4+j] = s
    return r

def mat_translate(x, y, z):
    return [1,0,0,x, 0,1,0,y, 0,0,1,z, 0,0,0,1]

def mat_scale(x, y, z):
    return [x,0,0,0, 0,y,0,0, 0,0,z,0, 0,0,0,1]

def mat_rotate(ax, ay, az, deg):
    # Rotate about axis (ax,ay,az) by `deg` degrees (Rodrigues).
    n = math.sqrt(ax*ax+ay*ay+az*az)
    if n == 0: return mat_identity()
    ax, ay, az = ax/n, ay/n, az/n
    a = math.radians(deg); c = math.cos(a); s = math.sin(a); t = 1-c
    return [t*ax*ax+c,     t*ax*ay-s*az, t*ax*az+s*ay, 0,
            t*ax*ay+s*az,  t*ay*ay+c,    t*ay*az-s*ax, 0,
            t*ax*az-s*ay,  t*ay*az+s*ax, t*az*az+c,    0,
            0,0,0,1]

def mat_lookat(origin, target, up):
    # Mitsuba lookat: camera at `origin`, looking toward `target`, local +z = forward.
    fwd = vnorm(vsub(target, origin))
    right = vnorm(vcross(up, fwd))
    newup = vcross(fwd, right)
    # columns = right, newup, fwd, origin
    return [right[0], newup[0], fwd[0], origin[0],
            right[1], newup[1], fwd[1], origin[1],
            right[2], newup[2], fwd[2], origin[2],
            0,0,0,1]

def xform_point(m, p):
    return [m[0]*p[0]+m[1]*p[1]+m[2]*p[2]+m[3],
            m[4]*p[0]+m[5]*p[1]+m[6]*p[2]+m[7],
            m[8]*p[0]+m[9]*p[1]+m[10]*p[2]+m[11]]

def xform_vec(m, v):
    return [m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
            m[4]*v[0]+m[5]*v[1]+m[6]*v[2],
            m[8]*v[0]+m[9]*v[1]+m[10]*v[2]]

# ------------------------------------------------------------------- small vec ops
def vsub(a,b): return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]
def vdot(a,b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
def vcross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def vlen(a): return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def vnorm(a):
    l = vlen(a)
    return [a[0]/l, a[1]/l, a[2]/l] if l > 0 else [0,0,0]

def cube_faces(M):
    """Mitsuba's cube, [-1,1]^3 placed by `M`, as six (origin, u, v) quads with u x v pointing
    out of the cube (for an `M` that does not mirror), the side a `light area` faces when it
    is given no `normal`."""
    faces = [
        ([-1,-1, 1],[ 1,-1, 1],[-1, 1, 1]),  # +z
        ([ 1,-1,-1],[-1,-1,-1],[ 1, 1,-1]),  # -z
        ([ 1,-1, 1],[ 1,-1,-1],[ 1, 1, 1]),  # +x
        ([-1,-1,-1],[-1,-1, 1],[-1, 1,-1]),  # -x
        ([-1, 1, 1],[ 1, 1, 1],[-1, 1,-1]),  # +y
        ([-1,-1,-1],[ 1,-1,-1],[-1,-1, 1]),  # -y
    ]
    out = []
    for a, b, c in faces:
        o = xform_point(M, a)
        out.append((o, vsub(xform_point(M, b), o), vsub(xform_point(M, c), o)))
    return out

# ------------------------------------------------------------------- XML helpers
def parse_floats(s):
    return [float(x) for x in s.replace(',', ' ').split()]

def parse_transform(node, defaults):
    """Compose a <transform> node's children into a single object->world matrix.
    Mitsuba applies children in document order with each new one on the LEFT, so the
    first-listed transform is applied first to the object point (M = child_n * ... * child_1)."""
    m = mat_identity()
    for c in node:
        tag = c.tag
        g = lambda k, d='0': float(sub(c.get(k, d), defaults))
        def xyz(d):   # `value="x, y, z"` (or one number for all three), else x/y/z attributes
            if c.get('value') is not None:
                v = parse_floats(sub(c.get('value'), defaults))
                return (v * 3)[:3] if len(v) == 1 else v[:3]
            return [g('x', d), g('y', d), g('z', d)]
        if tag == 'translate':
            cm = mat_translate(*xyz('0'))
        elif tag == 'scale':
            cm = mat_scale(*xyz('1'))
        elif tag == 'rotate':
            cm = mat_rotate(*xyz('0'), g('angle'))
        elif tag == 'matrix':
            vals = parse_floats(sub(c.get('value'), defaults))
            cm = vals if len(vals) == 16 else mat_identity()
        elif tag == 'lookat' or tag == 'lookAt':
            o = [float(v) for v in parse_floats(sub(c.get('origin'), defaults))]
            t = [float(v) for v in parse_floats(sub(c.get('target'), defaults))]
            up = c.get('up')
            up = [float(v) for v in parse_floats(sub(up, defaults))] if up else [0,1,0]
            cm = mat_lookat(o, t, up)
        else:
            continue
        m = mat_mul(cm, m)
    return m

def sub(s, defaults):
    """Substitute $var / ${var} using <default> values."""
    if s is None: return s
    if '$' in s:
        for k, v in defaults.items():
            s = s.replace('${'+k+'}', v).replace('$'+k, v)
    return s

# ------------------------------------------------------------------- spectra
# Every named IOR Mitsuba accepts (include/mitsuba/render/ior.h, the same table in 0.6 and 3).
MITSUBA_IOR = {
    'vacuum': 1.0,          'helium': 1.000036,     'hydrogen': 1.000132,
    'air': 1.000277,        'carbon dioxide': 1.00045,
    'water': 1.3330,        'acetone': 1.36,        'ethanol': 1.361,
    'carbon tetrachloride': 1.461,  'glycerol': 1.4729,     'benzene': 1.501,
    'silicone oil': 1.52045,        'bromine': 1.661,       'water ice': 1.31,
    'fused quartz': 1.458,  'pyrex': 1.470,         'acrylic glass': 1.49,
    'polypropylene': 1.49,  'bk7': 1.5046,          'sodium chloride': 1.544,
    'amber': 1.55,          'pet': 1.5750,          'diamond': 2.419,
}
# The names the glass library holds as the same material import as that DISPERSIVE glass, not
# as Mitsuba's constant: a deliberate upgrade, so a Mitsuba diamond gains the fire a constant
# index cannot give it (and bk7 moves from Mitsuba's 1.5046 to the catalogue's n_d 1.5168).
# Pyrex is a borosilicate but not BK7 (1.47 against 1.517), so it keeps its constant.
LIBRARY_GLASS = {
    'water': 'water', 'water ice': 'ice', 'fused quartz': 'silica',
    'acrylic glass': 'acrylic', 'bk7': 'BK7', 'diamond': 'diamond',
}

def get_xyz(elem, tag, name, defaults):
    """The <`tag` name="`name`"> child of `elem` (a point or vector) as [x, y, z], from its
    `value="x, y, z"` or its x/y/z attributes, or None when there is no such child."""
    for c in elem:
        if c.tag == tag and c.get('name') == name:
            if c.get('value') is not None:
                v = parse_floats(sub(c.get('value'), defaults))
                return (v * 3)[:3] if len(v) == 1 else v[:3]
            return [float(sub(c.get(k, '0'), defaults)) for k in 'xyz']
    return None

def linear_rgb(node, defaults):
    """An <rgb> or <srgb> node's colour as linear [r, g, b]. Mitsuba 0.6's srgb is
    display-encoded, so it is decoded by the sRGB curve; a '#rrggbb' value reads as its
    bytes / 255, encoded or not by the same rule, as 0.6 itself reads it."""
    val = sub(node.get('value'), defaults).strip()
    if val.startswith('#'):
        c = [int(val[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
    else:
        c = parse_floats(val)
        if len(c) == 1: c = c*3
    if node.tag == 'srgb':
        c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return c[:3]

def planck_table(T, scale=1.0, lo=360, hi=830, step=5):
    """A blackbody at `T` kelvin, times `scale`, as an FTSL `table` in Mitsuba's unit,
    W / (m^2 sr nm), sampled every `step` nm over the converted scene's spectral range."""
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    pairs = []
    for nm in range(lo, hi + 1, step):
        lam = nm * 1e-9
        b = 2 * h * c * c / lam ** 5 / math.expm1(h * c / (lam * k * T)) * 1e-9 * scale
        pairs.append(f"{nm}:{b:.6g}")
    return "table { " + " ".join(pairs) + " }"

def blackbody_table(node, defaults):
    """A Mitsuba blackbody -- 0.6's <blackbody temperature="5000K" scale=".."> or 3's
    <spectrum type="blackbody"> with <float> temperature and scale -- as planck_table."""
    def prop(k, d):
        v = node.get(k)
        if v is None:
            return get_float(node, k, d, defaults)
        return float(sub(v, defaults).strip().rstrip('Kk'))   # 0.6 writes "5000K"
    return planck_table(prop('temperature', 6500.0), prop('scale', 1.0))

def spectrum_from_node(node, defaults):
    """Turn a <rgb>/<spectrum>/<blackbody> child into an FTSL spectrum expression."""
    if node is None: return None
    tag = node.tag
    val = sub(node.get('value'), defaults)
    if tag == 'rgb' or tag == 'srgb':
        c = linear_rgb(node, defaults)
        return f"rgb {fnum(c[0])} {fnum(c[1])} {fnum(c[2])}"
    if tag == 'blackbody' or node.get('type') == 'blackbody':
        return blackbody_table(node, defaults)
    if tag == 'spectrum' and node.get('type') in ('irregular', 'regular'):
        # Mitsuba 3's sampled spectra: values at listed wavelengths, or spread evenly
        # from wavelength_min to wavelength_max.
        vals = parse_floats(get_string(node, 'values', defaults) or '')
        if node.get('type') == 'irregular':
            wls = parse_floats(get_string(node, 'wavelengths', defaults) or '')
        else:
            lo = get_float(node, 'wavelength_min', None, defaults)
            hi = get_float(node, 'wavelength_max', None, defaults)
            n = len(vals)
            wls = [lo + (hi - lo) * i / (n - 1) for i in range(n)] \
                if lo is not None and hi is not None and n > 1 else []
        if vals and len(wls) == len(vals):
            return "table { " + " ".join(f"{fnum(w)}:{fnum(v)}" for w, v in zip(wls, vals)) + " }"
        return None
    if tag == 'spectrum':
        fn = node.get('filename')
        if fn:
            return f"file:{fn}"
        if val is None: return None
        if ':' in val:
            # "wl:v, wl:v, ..." measured pairs -> FTSL table
            pairs = []
            for tok in val.replace(',', ' ').split():
                if ':' in tok:
                    w, v = tok.split(':'); pairs.append(f"{fnum(float(w))}:{fnum(float(v))}")
            if pairs:
                return "table { " + " ".join(pairs) + " }"
        f = parse_floats(val)
        return fnum(f[0]) if f else None
    return None

def first_child_spectrum(elem, name, defaults):
    """Find a <rgb|spectrum|blackbody name="..."> child and convert it."""
    for c in elem:
        if c.tag in ('rgb','srgb','spectrum','blackbody') and c.get('name') == name:
            return spectrum_from_node(c, defaults)
    # fall back to a nameless spectrum child
    for c in elem:
        if c.tag in ('rgb','srgb','spectrum','blackbody') and c.get('name') is None:
            return spectrum_from_node(c, defaults)
    return None

def get_float(elem, name, default, defaults):
    for c in elem:
        if c.tag in ('float','integer') and c.get('name') == name:
            return float(sub(c.get('value'), defaults))
    return default

def get_string(elem, name, defaults):
    for c in elem:
        if c.tag == 'string' and c.get('name') == name:
            return sub(c.get('value'), defaults)
    return None

def get_ior(elem, name, default, defaults):
    """The BSDF property `name` (int_ior or ext_ior) as a pair: the FTSL `ior ...` expression
    for it, and the index Mitsuba itself uses. Mitsuba takes a number or a MITSUBA_IOR name;
    `default`, a MITSUBA_IOR name, stands in when the property is absent, and when it names
    nothing Mitsuba knows (with a warning, so the output still parses). Constants are written
    with every digit Mitsuba's table has: a gas's n - 1 lives in the sixth."""
    nm = default
    for c in elem:
        if c.get('name') == name:
            if c.tag in ('float', 'integer'):
                x = float(sub(c.get('value'), defaults))
                return f"ior {x!r}", x
            if c.tag == 'string':
                given = sub(c.get('value'), defaults).lower()
                if given in MITSUBA_IOR:
                    nm = given
                else:
                    warn(f"unknown named IOR '{given}' -> {default}")
            break
    g = LIBRARY_GLASS.get(nm)
    return (f"ior glass:{g}" if g else f"ior {MITSUBA_IOR[nm]!r}"), MITSUBA_IOR[nm]

PHONG_HW = math.sqrt(math.log(2))           # Phong lobe half-width per unit FTSL roughness
BECKMANN_HW = math.sqrt(math.log(2))        # half-vector half-width per unit alpha
GGX_HW = math.sqrt(math.sqrt(2) - 1)

def ftsl_roughness(bsdf, name, defaults, spread=2.0):
    """(FTSL `roughness`, Mitsuba alpha) for the microfacet BSDF `bsdf` (material `name`).

    Matched by lobe width. FTSL's glossy lobe, which a dielectric's frosting also puts about
    its refracted ray, is a Phong lobe of exponent 2/r^2 - 2: half-width PHONG_HW * r.
    Mitsuba's alpha sets the half-vector distribution, half-width BECKMANN_HW * alpha (its
    default, and what 0.6's Phong reduces to) or GGX_HW * alpha, and the outgoing ray turns
    `spread` times as fast as the half-vector: 2 for a reflection, 1 - 1/eta entering a
    refraction and eta - 1 leaving it. So r = spread * alpha for Beckmann (0.773 of that for
    GGX), capped at 1. An anisotropic alpha_u/alpha_v pair is averaged (a lone one, invalid
    in Mitsuba, against 0.1), and an alpha this cannot read (a texture) becomes Mitsuba's
    default 0.1; each with a warning."""
    a = get_float(bsdf, 'alpha', None, defaults)
    if a is None:
        au = get_float(bsdf, 'alpha_u', None, defaults)
        av = get_float(bsdf, 'alpha_v', None, defaults)
        if au is not None or av is not None:
            if au is None or av is None:
                warn(f"'{name}': only one of alpha_u/alpha_v given, which Mitsuba rejects -> "
                     f"the other taken as 0.1")
            au = 0.1 if au is None else au
            av = 0.1 if av is None else av
            a = 0.5 * (au + av)
            if abs(au - av) > 1e-9:
                warn(f"'{name}': anisotropic alpha_u {fnum(au)} / alpha_v {fnum(av)} -> "
                     f"isotropic alpha {fnum(a)}")
        else:
            a = 0.1
            if any(c.get('name') in ('alpha', 'alpha_u', 'alpha_v') for c in bsdf):
                warn(f"'{name}': a textured alpha -> Mitsuba's default 0.1")
    ggx = (get_string(bsdf, 'distribution', defaults) or '').lower() == 'ggx'
    r = spread * (GGX_HW if ggx else BECKMANN_HW) / PHONG_HW * a
    return min(max(r, 0.0), 1.0), a

# ------------------------------------------------------------------- BSDF -> material
CONDUCTOR_METAL = {  # Mitsuba conductor material -> FTSL metal:
    'au':'Au','ag':'Ag','cu':'Cu','al':'Al','cr':'Cr','none':None,
}

class Converter:
    def __init__(self):
        self.materials = {}     # ftsl_name -> body string (for `material "n" { ... }`)
        self.mat_order = []
        self.named_bsdf = {}    # xml id -> ftsl material name
        self.textures = {}      # ftsl_name -> body
        self.tex_order = []
        self.geometry = []      # list of FTSL geometry/light lines
        self.cameras = []       # list of FTSL camera blocks
        self.anon_count = 0
        self.defaults = {}
        self.legacy = False     # a Mitsuba 0.x scene (set by convert)
        self.has_light = False  # set by add_light and emissive
        self.thinlens = False   # a thinlens sensor became a mode A camera
        self.emissive_of = {}   # (material, spd) -> its emissive copy

    def add_light(self, line):
        self.geometry.append(line)
        self.has_light = True

    def uniq(self, base):
        base = base or 'mat'
        base = ''.join(ch if (ch.isalnum() or ch in '_-') else '_' for ch in base)
        if base not in self.materials and base not in [n for n in self.mat_order]:
            return base
        i = 1
        while f"{base}_{i}" in self.materials:
            i += 1
        return f"{base}_{i}"

    def add_material(self, name, body):
        if name not in self.materials:
            self.materials[name] = body
            self.mat_order.append(name)
        return name

    # --- textures -----------------------------------------------------------
    def texture_from_node(self, node, hint):
        fn = get_string(node, 'filename', self.defaults)
        if not fn: return None
        name = self.uniq((hint or 'tex') + '_tex')
        enc = 'srgb'
        body = f'file "{fn}"  encoding {enc}'
        self.textures[name] = body
        self.tex_order.append(name)
        return f"texture:{name}"

    def reflectance_expr(self, elem, name, default, hint):
        """A reflectance can be a spectrum/rgb OR a nested <texture>."""
        for c in elem:
            if c.tag == 'texture' and c.get('name') == name:
                t = self.texture_from_node(c, hint)
                if t: return t
        s = first_child_spectrum(elem, name, self.defaults)
        return s if s is not None else default

    def bsdf_to_material(self, bsdf, hint=None):
        """Return an FTSL material name for a <bsdf> node (registers it)."""
        btype = bsdf.get('type')
        bid = bsdf.get('id')
        hint = bid or hint

        # unwrap transparent wrappers
        if btype in ('twosided', 'bumpmap', 'normalmap'):
            inner = bsdf.find('bsdf')
            if inner is not None:
                if btype != 'twosided':
                    warn(f"{btype} BSDF: bump/normal perturbation dropped (using base BSDF)")
                return self.bsdf_to_material(inner, hint)
            return self.add_material(self.uniq(hint or 'mat'), 'type diffuse  reflect 0.5')
        if btype == 'mask':
            inner = bsdf.find('bsdf')
            if inner is not None:
                warn("mask BSDF: opacity ignored (using base BSDF opaque)")
                return self.bsdf_to_material(inner, hint)

        name = self.uniq(hint or btype or 'mat')

        if btype == 'diffuse':
            refl = self.reflectance_expr(bsdf, 'reflectance', '0.5', name)
            body = f"type diffuse  reflect {refl}"
        elif btype in ('conductor', 'roughconductor'):
            # Mitsuba 0.x defaulted a conductor to copper; 2 and later to a perfect mirror.
            mat = (get_string(bsdf, 'material', self.defaults)
                   or ('cu' if self.legacy else 'none')).lower()
            metal = CONDUCTOR_METAL.get(mat)
            refl = f"metal:{metal}" if metal else \
                   (first_child_spectrum(bsdf, 'specular_reflectance', self.defaults) or '0.9')
            if mat not in CONDUCTOR_METAL:
                warn(f"conductor material '{mat}' has no FTSL metal: data -> reflect {refl}")
            if any(c.get('name') in ('eta', 'k') for c in bsdf):
                warn(f"{btype} '{name}': eta/k dropped -> reflect {refl}")
            if btype == 'roughconductor':
                r, _ = ftsl_roughness(bsdf, name, self.defaults)
                body = f"type glossy  reflect {refl}  roughness {fnum(r)}"
            else:
                body = f"type mirror  reflect {refl}"
        elif btype in ('dielectric', 'roughdielectric', 'thindielectric'):
            ior, n_in = get_ior(bsdf, 'int_ior', 'bk7', self.defaults)
            _, ext = get_ior(bsdf, 'ext_ior', 'air', self.defaults)
            for tint in ('specular_reflectance', 'specular_transmittance'):
                if any(c.get('name') == tint for c in bsdf):
                    warn(f"{btype} '{name}': {tint} dropped - an FTSL dielectric's interface "
                         f"is untinted (its `absorb` tints by depth instead)")
            if btype == 'thindielectric':
                # A sheet with no thickness: light crosses undeviated, and what it reflects is
                # both faces' Fresnel with every inter-reflection, 2R/(1+R), taken here at
                # normal incidence -- a lossless beamsplitter, which is FTSL's halfmirror.
                eta = n_in / ext
                r0 = ((eta - 1) / (eta + 1)) ** 2
                refl = 2 * r0 / (1 + r0)
                body = f"type halfmirror  reflect {fnum(refl)}"
                warn(f"thindielectric '{name}' -> halfmirror reflecting {fnum(refl)}, the "
                     f"sheet's normal-incidence reflectance (no rise toward grazing)")
            else:
                body = f"type dielectric  {ior}"
                if abs(ext - 1.0) > 1e-3:   # Mitsuba's default is air
                    warn(f"{btype} '{name}': ext_ior {fnum(ext)} dropped - FTSL takes the "
                         f"index outside a dielectric from the dielectric enclosing it, so "
                         f"model the outside medium as a dielectric of its own")
            if btype == 'roughdielectric':
                # Matched to the transmitted lobe, which carries most of glass's light: the
                # mean of its widths entering and leaving. The reflected lobe comes out narrower.
                eta = n_in / ext
                r, a = ftsl_roughness(bsdf, name, self.defaults,
                                      spread=(abs(1 - 1 / eta) + abs(eta - 1)) / 2)
                body += f"  roughness {fnum(r)}"
                warn(f"roughdielectric '{name}': alpha {fnum(a)} -> roughness {fnum(r)}, as wide "
                     f"as its transmitted lobe; FTSL frosts with one glossy lobe about both rays, "
                     f"so its reflection is the narrower")
        elif btype in ('plastic', 'roughplastic'):
            diff = self.reflectance_expr(bsdf, 'diffuse_reflectance', '0.5', name)
            r = ftsl_roughness(bsdf, name, self.defaults)[0] if btype == 'roughplastic' else 0.05
            body = f"type glossy  reflect {diff}  roughness {fnum(r)}"
            warn(f"{btype} -> glossy approximation (diffuse+specular coat merged)")
        elif btype == 'blendbsdf':
            kids = bsdf.findall('bsdf')
            w = get_float(bsdf, 'weight', 0.5, self.defaults)
            if len(kids) == 2:
                a = self.bsdf_to_material(kids[0], name + '_a')
                b = self.bsdf_to_material(kids[1], name + '_b')
                # Mitsuba weight = fraction of the SECOND bsdf
                body = f'type mix\n    layer "{a}" {fnum(1-w)}\n    layer "{b}" {fnum(w)}'
            else:
                warn("blendbsdf without exactly 2 children -> diffuse fallback")
                body = "type diffuse  reflect 0.5"
        else:
            warn(f"unsupported BSDF type '{btype}' -> diffuse 0.5 fallback")
            body = "type diffuse  reflect 0.5"

        return self.add_material(name, body)

    def emitter_spd(self, emitter, name):
        """The FTSL `spd` for the emitter property `name` (radiance, irradiance, intensity), at
        Mitsuba's own magnitude, so lights keep their brightness relative to one another. An
        rgb value becomes `rgbillum`, FTSL's emission upsampling (`rgb` would read it as a 0..1
        reflectance). A bare number -- a <float>, a one-value <spectrum>, 3's <spectrum
        type="d65"> scale, or Mitsuba's default 1 when there is none -- is white, as Mitsuba
        makes it (D65-shaped, or an RGB build's grey), so it becomes `rgbillum v v v` too. A
        sampled spectrum passes through, and a blackbody becomes its Planck curve tabulated per
        nanometre, Mitsuba's unit (FTSL's own `blackbody` is per metre, 1e9 brighter)."""
        node = None
        for c in emitter:
            if c.tag in ('rgb','srgb','spectrum','blackbody','float','texture') and \
                    (c.get('name') == name or (c.get('name') is None and c.tag not in ('float', 'texture'))):
                node = c; break
        white = lambda v: f"rgbillum {fnum(v)} {fnum(v)} {fnum(v)}"
        kind = emitter.get('type')
        if node is None:
            warn(f"{kind} emitter has no {name} -> Mitsuba's default, 1")
            return white(1.0)
        if node.tag == 'texture':
            warn(f"{kind} emitter: its textured {name} dropped -> 1")
            return white(1.0)
        if node.tag in ('rgb','srgb'):
            return "rgbillum " + " ".join(fnum(x) for x in linear_rgb(node, self.defaults))
        if node.tag == 'float':
            return white(float(sub(node.get('value'), self.defaults)))
        if node.get('type') == 'd65':
            return white(get_float(node, 'scale', 1.0, self.defaults))
        val = sub(node.get('value'), self.defaults)
        if node.tag == 'spectrum' and val is not None and ':' not in val \
                and len(parse_floats(val)) == 1:
            return white(parse_floats(val)[0])
        s = spectrum_from_node(node, self.defaults)
        if s is None:
            warn(f"{kind} emitter: its {name} could not be read -> 1")
            return white(1.0)
        return s

    # --- shapes / emitters --------------------------------------------------
    def resolve_shape_material(self, shape):
        ref = shape.find('ref')
        if ref is not None and ref.get('id') in self.named_bsdf:
            return self.named_bsdf[ref.get('id')]
        b = shape.find('bsdf')
        if b is not None:
            return self.bsdf_to_material(b)
        return None

    def handle_shape(self, shape):
        stype = shape.get('type')
        tnode = None
        for c in shape:
            if c.tag == 'transform' and c.get('name') == 'to_world':
                tnode = c
        M = parse_transform(tnode, self.defaults) if tnode is not None else mat_identity()
        emitter = shape.find('emitter')
        is_light = emitter is not None and emitter.get('type') == 'area'
        spd = None
        if is_light:
            spd = self.emitter_spd(emitter, 'radiance')

        if stype == 'rectangle':
            # canonical rectangle: [-1,-1,0]..[1,1,0], normal +z
            o = xform_point(M, [-1,-1,0])
            pu = xform_point(M, [1,-1,0])
            pv = xform_point(M, [-1,1,0])
            u = vsub(pu, o); v = vsub(pv, o)
            if is_light:
                self.add_light(
                    f"light area {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  spd {spd} }}")
            else:
                mat = self.resolve_shape_material(shape) or self.default_mat()
                self.geometry.append(
                    f"quad {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  material {mat} }}")
        elif stype == 'cube':
            if is_light:   # six area lights, as a rectangle emitter becomes one
                for o, u, v in cube_faces(M):
                    self.add_light(
                        f"light area {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  spd {spd} }}")
            else:
                mat = self.resolve_shape_material(shape) or self.default_mat()
                for o, u, v in cube_faces(M):
                    self.geometry.append(
                        f"quad {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  material {mat} }}")
        elif stype == 'sphere':
            # center/radius from explicit props or from the transform of the unit sphere
            # Mitsuba places it by to_world . translate(center) . scale(radius).
            center = xform_point(M, get_xyz(shape, 'point', 'center', self.defaults) or [0,0,0])
            s = [vlen(xform_vec(M, e)) for e in ([1,0,0], [0,1,0], [0,0,1])]
            if max(s) - min(s) > 1e-6 * max(s):
                warn(f"sphere: to_world scales it unevenly ({v3(s)}), which a sphere cannot "
                     f"take -> their mean")
            radius = get_float(shape, 'radius', 1.0, self.defaults) * sum(s) / 3
            if is_light:
                self.add_light(
                    f"light sphere {{ center {v3(center)}  radius {fnum(radius)}  spd {spd} }}")
            else:
                mat = self.resolve_shape_material(shape) or self.default_mat()
                self.geometry.append(
                    f"sphere {{ center {v3(center)}  radius {fnum(radius)}  material {mat} }}")
        elif stype == 'serialized':
            fn = get_string(shape, 'filename', self.defaults) or '?'
            warn(f"shape '{fn}' is Mitsuba .serialized, which ftrace does not read -> skipped; "
                 f"convert it to .obj or .ply")
        elif stype in ('obj', 'ply'):
            fn = get_string(shape, 'filename', self.defaults) or '?'
            mat = self.resolve_shape_material(shape) or self.default_mat()
            if is_light:
                mat = self.emissive(mat, spd)
            self.emit_mesh(fn, mat, M)
        else:
            warn(f"unsupported shape type '{stype}' -> skipped")

    def emissive(self, mat, spd):
        """The name of a copy of material `mat` that also radiates `spd`, which makes the mesh
        bound to it an area light. A copy, so other shapes sharing `mat` stay dark; one per
        (material, spd), so meshes that glow alike share it."""
        key = (mat, spd)
        if key not in self.emissive_of:
            body = self.materials[mat]
            if '\n' in body:   # a mix; `emit` belongs on one plain material
                warn(f"area emitter on mix material '{mat}' -> black diffuse emitter")
                body = 'type diffuse  reflect 0'
            self.emissive_of[key] = self.add_material(self.uniq(mat + '_emit'),
                                                      f"{body}  emit {spd}")
        self.has_light = True
        return self.emissive_of[key]

    def emit_mesh(self, fn, mat, M):
        """Append a `mesh` line placing the file `fn` by Mitsuba's `to_world` M. FTSL places a
        mesh by a per-axis scale, then rotate (Euler degrees, R = Rz.Ry.Rx), then translate,
        which is exactly an M whose linear part has orthogonal columns and does not mirror. A
        shear or a mirror is flagged and placed approximately."""
        t = xform_point(M, [0,0,0])
        cols = [xform_vec(M, e) for e in ([1,0,0], [0,1,0], [0,0,1])]
        s = [vlen(c) for c in cols]
        R = [vnorm(c) for c in cols]               # R[j] is column j: m[row][j] = R[j][row]
        line = f'mesh {{ file "{fn}"  material {mat}  translate {v3(t)}'
        if any(abs(x - 1) > 1e-6 for x in s):
            line += f"  scale {fnum(s[0])}" if max(s) - min(s) < 1e-6 * max(s) else f"  scale {v3(s)}"
        if max(abs(vdot(R[0], R[1])), abs(vdot(R[0], R[2])), abs(vdot(R[1], R[2]))) > 1e-6:
            warn(f"mesh '{fn}' (material {mat}): to_world shears it, which a mesh's "
                 f"scale/rotate/translate cannot express -> placed approximately, without the shear")
        if vdot(vcross(R[0], R[1]), R[2]) < 0:
            warn(f"mesh '{fn}' (material {mat}): to_world mirrors it, which a mesh's "
                 f"scale/rotate/translate cannot express -> placed unmirrored")
            R[0] = [-x for x in R[0]]
        # R = Rz(g) Ry(b) Rx(a): m20 = -sin b, (m21, m22) = cos b (sin a, cos a). Near cos b = 0
        # only a combination of a and g is defined, and entries of size cos b are mostly the
        # matrix's own rounding; so b is taken from its whole column, and g, given a, from
        # entries of size 1 -- the second column of R Rx(-a), which is (-sin g, cos g, 0).
        b = math.atan2(-R[0][2], math.hypot(R[0][0], R[0][1]))
        a = math.atan2(R[1][2], R[2][2]) if math.hypot(R[1][2], R[2][2]) > 1e-9 else 0.0
        sa, ca = math.sin(a), math.cos(a)
        g = math.atan2(sa * R[2][0] - ca * R[1][0], ca * R[1][1] - sa * R[2][1])
        rot = [math.degrees(x) for x in (a, b, g)]
        rot = [0.0 if abs(x) < 1e-9 else x for x in rot]
        if any(rot):
            line += f"  rotate {v3(rot)}"
        line += " }"
        self.geometry.append(line)

    def default_mat(self):
        if '_default' not in self.materials:
            self.add_material('_default', 'type diffuse  reflect 0.7')
        return '_default'

    # --- sensor -------------------------------------------------------------
    def handle_sensor(self, sensor):
        stype = sensor.get('type')
        tnode = None
        film = sensor.find('film')
        w = int(get_float(film, 'width', 768, self.defaults)) if film is not None else 768
        h = int(get_float(film, 'height', 576, self.defaults)) if film is not None else 576
        for c in sensor:
            if c.tag == 'transform' and c.get('name') == 'to_world':
                tnode = c
        M = parse_transform(tnode, self.defaults) if tnode is not None else mat_identity()
        eye = xform_point(M, [0,0,0])
        look = xform_point(M, [0,0,1])   # Mitsuba camera forward = +z
        up = xform_vec(M, [0,1,0])

        fov = get_float(sensor, 'fov', 45.0, self.defaults)
        fov_axis = (get_string(sensor, 'fov_axis', self.defaults) or 'x').lower()
        aspect = w / h if h else 1.0
        fov_y = self.fov_to_y(fov, fov_axis, aspect)

        lines = [f'camera "cam" {{']
        lines.append(f"    eye {v3(eye)}  look_at {v3(look)}  up {v3(up)}")
        lines.append(f"    fov_y {fnum(fov_y)}")
        if stype == 'thinlens':
            ap = get_float(sensor, 'aperture_radius', 0.0, self.defaults)
            focus = get_float(sensor, 'focus_distance', 0.0, self.defaults)
            lines.append(f"    aperture {fnum(ap)}  focus {fnum(focus)}  mode A")
            self.thinlens = True
        else:   # backward path tracing, as Mitsuba renders; B would leave glass seen directly black
            lines.append(f"    mode R")
        lines.append(f"    film {{ res {w} {h} }}")
        lines.append("}")
        self.cameras.append("\n".join(lines))

    def fov_to_y(self, fov, axis, aspect):
        fr = math.radians(fov)
        if axis in ('y',):
            return fov
        if axis in ('x',):
            return math.degrees(2*math.atan(math.tan(fr/2)/aspect))
        if axis == 'smaller':
            return fov if aspect <= 1 else math.degrees(2*math.atan(math.tan(fr/2)/aspect))
        if axis == 'larger':
            return fov if aspect >= 1 else math.degrees(2*math.atan(math.tan(fr/2)/aspect))
        if axis == 'diagonal':
            warn("fov_axis=diagonal approximated as horizontal")
            return math.degrees(2*math.atan(math.tan(fr/2)/aspect))
        return fov

    # --- top-level emitters (env / constant) --------------------------------
    def handle_scene_emitter(self, emitter):
        etype = emitter.get('type')
        tnode = next((c for c in emitter if c.tag == 'transform' and c.get('name') == 'to_world'), None)
        M = parse_transform(tnode, self.defaults) if tnode is not None else mat_identity()
        if etype == 'constant':
            spd = self.emitter_spd(emitter, 'radiance')
            self.add_light(f"light env {{ spd {spd} }}")
        elif etype == 'envmap':
            fn = get_string(emitter, 'filename', self.defaults) or 'env.hdr'
            scale = get_float(emitter, 'scale', 1.0, self.defaults)
            if any(abs(x - y) > 1e-9 for x, y in zip(M, mat_identity())):
                warn(f"envmap '{fn}': to_world dropped - an FTSL env turns only about +y, by "
                     f"its `rotate`; set that by hand")
            self.add_light(f'light env {{ file "{fn}"  intensity {fnum(scale)} }}')
        elif etype in ('point', 'spot'):
            # A spot whose cone is the whole sphere is an exact point light, and a spot's
            # spd is, like Mitsuba's intensity, per steradian. Both aim along to_world's +z.
            pos = get_xyz(emitter, 'point', 'position', self.defaults) or xform_point(M, [0,0,0])
            spd = self.emitter_spd(emitter, 'intensity')
            if etype == 'point':
                inner = outer = 180.0
                d = [0.0, -1.0, 0.0]
            else:
                outer = get_float(emitter, 'cutoff_angle', 20.0, self.defaults)
                inner = get_float(emitter, 'beam_width', 0.75 * outer, self.defaults)
                d = vnorm(xform_vec(M, [0,0,1]))
                if vlen(d) == 0:
                    warn("spot emitter whose to_world leaves it no direction -> straight down")
                    d = [0.0, -1.0, 0.0]
                warn(f"spot emitter -> light spot {fnum(inner)}/{fnum(outer)} deg, whose edge "
                     f"falls off smoothly in cosine where Mitsuba's falls linearly in angle")
            self.add_light(f"light spot {{ origin {v3(pos)}  dir {v3(d)}  inner_angle "
                           f"{fnum(inner)}  outer_angle {fnum(outer)}  spd {spd} }}")
        elif etype == 'directional':
            # Mitsuba's direction is the one light travels, and its irradiance falls on a
            # surface facing it, as a sun's `spd` does; a sun's `dir` points back at the sun.
            d = vnorm(get_xyz(emitter, 'vector', 'direction', self.defaults)
                      or xform_vec(M, [0,0,1]))
            if vlen(d) == 0:
                warn("directional emitter with a zero direction -> straight down")
                d = [0.0, -1.0, 0.0]
            spd = self.emitter_spd(emitter, 'irradiance')
            warn("directional emitter -> light sun 0.53 deg across (FTSL's sun needs a size; "
                 "Mitsuba's has none, so its shadows were hard)")
            self.add_light(f"light sun {{ dir {v3([0.0 - x for x in d])}  spd {spd} }}")   # no -0
        else:
            warn(f"unsupported scene emitter '{etype}' -> skipped")

    # --- driver -------------------------------------------------------------
    @staticmethod
    def upgrade_from_v1(root):
        """Rename a Mitsuba 0.x scene's properties in place, as Mitsuba's own parser does
        (src/core/parser.cpp upgrade_from_v1), so the rest of the converter reads one naming:
        a camelCase name becomes snake_case (toWorld -> to_world, intIOR -> int_ior), and a
        diffuse BSDF's diffuse_reflectance becomes reflectance. <default> names are left
        alone: they are the scene's own `$name` parameters, not properties. Returns whether
        the scene is 0.x, whose remaining differences are defaults (a conductor's, say)."""
        ver = root.get('version', '').split('.')[0]
        if not ver.isdigit() or int(ver) >= 2:
            return False
        snake = lambda s: re.sub(r'(?<=[a-z])[A-Z]+', lambda m: '_' + m.group(0).lower(), s)
        for e in root.iter():
            if e.tag != 'default' and e.get('name') is not None:
                e.set('name', snake(e.get('name')))
        for b in root.iter('bsdf'):
            if b.get('type') == 'diffuse':
                for c in b:
                    if c.get('name') == 'diffuse_reflectance':
                        c.set('name', 'reflectance')
        return True

    def convert(self, root):
        self.legacy = self.upgrade_from_v1(root)
        # <default name=.. value=..> substitution table
        for c in root:
            if c.tag == 'default':
                self.defaults[c.get('name')] = c.get('value', '')
        # top-level named BSDFs (id) first, so <ref> resolves
        for c in root:
            if c.tag == 'bsdf' and c.get('id'):
                nm = self.bsdf_to_material(c)
                self.named_bsdf[c.get('id')] = nm
        for c in root:
            if c.tag == 'shape':
                self.handle_shape(c)
            elif c.tag == 'sensor':
                self.handle_sensor(c)
            elif c.tag == 'emitter':
                self.handle_scene_emitter(c)
        if self.thinlens and any(self.materials[m].startswith(('type dielectric', 'type mirror',
                                                               'type halfmirror'))
                                 for m in self.used_materials()):
            warn("thinlens sensor -> mode A, whose depth of field only the forward modes "
                 "render, and which shows glass and mirrors seen directly as black; pass "
                 "`-mode R` to see them, in focus everywhere")

    def used_materials(self):
        """The materials some geometry is bound to, and the layers of any mix among them."""
        todo = {m for g in self.geometry for m in re.findall(r'\bmaterial (\S+)', g)}
        used = set()
        while todo:
            m = todo.pop()
            if m in used or m not in self.materials:
                continue
            used.add(m)
            todo |= set(re.findall(r'layer "([^"]+)"', self.materials[m]))
        return used

    def emit(self):
        out = []
        out.append("# Auto-generated from a Mitsuba XML scene by tools/mitsuba_to_ftsl.py")
        out.append("# Review WARN comments below; approximations are documented inline.\n")
        out.append("scene {\n    units    meters\n    spectral 360 830 1\n}\n")
        for name in self.tex_order:
            out.append(f'texture "{name}" {{ {self.textures[name]} }}')
        if self.tex_order: out.append("")
        for name in self.mat_order:
            body = self.materials[name]
            if "\n" in body:
                out.append(f'material "{name}" {{\n    {body}\n}}')
            else:
                out.append(f'material "{name}" {{ {body} }}')
        out.append("")
        for g in self.geometry:
            out.append(g)
        out.append("")
        if not self.has_light:
            out.append("# WARN: no lights found in source scene; adding a fallback so it renders")
            out.append("light area { origin -0.5 2 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:d65 }")
            out.append("")
        for cam in self.cameras:
            out.append(cam)
        if not self.cameras:
            out.append('# WARN: no sensor found; adding a default camera')
            out.append('camera "cam" { eye 0 1 4  look_at 0 1 0  up 0 1 0  fov_y 45  mode R  film { res 512 512 } }')
        return "\n".join(out) + "\n"

def v3(a):
    return f"{fnum(a[0])} {fnum(a[1])} {fnum(a[2])}"

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(inp)[0] + '.ftsl'
    tree = ET.parse(inp)
    root = tree.getroot()
    conv = Converter()
    conv.convert(root)
    text = conv.emit()
    if warnings:
        # dedupe, keep order
        seen = set(); uniqw = []
        for wmsg in warnings:
            if wmsg not in seen: seen.add(wmsg); uniqw.append(wmsg)
        # inject the warning list near the top of the file too
        header, rest = text.split('\n', 1)
        wblock = "\n".join(f"# WARN: {w}" for w in uniqw)
        text = header + "\n" + wblock + "\n" + rest
        sys.stderr.write(f"[mitsuba_to_ftsl] {len(uniqw)} warning(s):\n")
        for w in uniqw:
            sys.stderr.write(f"  - {w}\n")
    with open(outp, 'w', encoding='utf-8', newline='') as f:   # ftrace reads UTF-8
        f.write(text)
    sys.stderr.write(f"[mitsuba_to_ftsl] wrote {outp}  "
                     f"({len(conv.mat_order)} materials, {len(conv.geometry)} geo/light, "
                     f"{len(conv.cameras)} camera)\n")

if __name__ == '__main__':
    main()
