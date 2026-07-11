#!/usr/bin/env python3
"""Mitsuba (0.6 / 2 / 3) XML scene -> FTSL converter.

Mitsuba and this renderer are both spectral, physically-based path tracers, so most
concepts map almost 1:1: perspective/thinlens sensor -> FTSL camera, area/constant/
envmap emitters -> FTSL lights, and diffuse/conductor/dielectric/plastic BSDFs ->
FTSL materials. That makes Mitsuba XML the natural import target (and Blender can
export straight to it via the mitsuba-blender addon, so this doubles as a Blender
path). RGB reflectances ride FTSL's Jakob-Hanika `rgb` upsampling; measured/blackbody
spectra pass through losslessly.

Usage:
    python tools/mitsuba_to_ftsl.py scene.xml [out.ftsl]

Unsupported constructs are skipped with a `# WARN:` comment in the output and a note
on stderr, so the result always parses. Coverage is the common subset an exported
scene uses; exotic BSDFs/shapes degrade to a documented approximation.
"""
import sys, os, math
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
def vcross(a,b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def vlen(a): return math.sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2])
def vnorm(a):
    l = vlen(a)
    return [a[0]/l, a[1]/l, a[2]/l] if l > 0 else [0,0,0]

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
        if tag == 'translate':
            cm = mat_translate(g('x'), g('y'), g('z'))
        elif tag == 'scale':
            if c.get('value') is not None:
                v = g('value','1'); cm = mat_scale(v, v, v)
            else:
                cm = mat_scale(g('x','1'), g('y','1'), g('z','1'))
        elif tag == 'rotate':
            cm = mat_rotate(g('x'), g('y'), g('z'), g('angle'))
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
NAMED_IOR = {  # Mitsuba named IOR -> FTSL glass: dispersion (closest available)
    'bk7':'BK7','water':'water','diamond':'diamond','acrylic glass':'acrylic',
    'polycarbonate':'polycarbonate','fused quartz':'silica','silica':'silica',
    'sapphire':'sapphire','pyrex':'BK7','air':None,'vacuum':None,
}

def spectrum_from_node(node, defaults):
    """Turn a <rgb>/<spectrum>/<blackbody> child into an FTSL spectrum expression."""
    if node is None: return None
    tag = node.tag
    val = sub(node.get('value'), defaults)
    if tag == 'rgb' or tag == 'srgb':
        c = parse_floats(val)
        if len(c) == 1: c = c*3
        return f"rgb {fnum(c[0])} {fnum(c[1])} {fnum(c[2])}"
    if tag == 'blackbody':
        t = sub(node.get('temperature','6500'), defaults)
        return f"blackbody {fnum(float(t))}"
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
    """int_ior/ext_ior can be a named material or a float -> FTSL glass:/ior expr."""
    for c in elem:
        if c.get('name') == name:
            if c.tag == 'string':
                nm = sub(c.get('value'), defaults).lower()
                g = NAMED_IOR.get(nm)
                return f"glass:{g}" if g else None
            if c.tag in ('float','integer'):
                return f"ior {fnum(float(sub(c.get('value'), defaults)))}"
    return default

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
            mat = (get_string(bsdf, 'material', self.defaults) or 'none').lower()
            metal = CONDUCTOR_METAL.get(mat)
            refl = f"metal:{metal}" if metal else \
                   (first_child_spectrum(bsdf, 'specular_reflectance', self.defaults) or '0.9')
            if btype == 'roughconductor':
                a = get_float(bsdf, 'alpha', 0.1, self.defaults)
                body = f"type glossy  reflect {refl}  roughness {fnum(min(max(a,0),1))}"
            else:
                body = f"type mirror  reflect {refl}"
        elif btype in ('dielectric', 'roughdielectric', 'thindielectric'):
            ior = get_ior(bsdf, 'int_ior', 'ior glass:BK7', self.defaults)
            body = f"type dielectric  {ior}"
            if btype == 'roughdielectric':
                warn("roughdielectric -> smooth dielectric (no rough transmission in FTSL)")
        elif btype in ('plastic', 'roughplastic'):
            diff = self.reflectance_expr(bsdf, 'diffuse_reflectance', '0.5', name)
            a = get_float(bsdf, 'alpha', 0.1, self.defaults) if btype == 'roughplastic' else 0.05
            body = f"type glossy  reflect {diff}  roughness {fnum(min(max(a,0),1))}"
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
        elif btype in ('conductor',):
            body = "type mirror  reflect 0.9"
        else:
            warn(f"unsupported BSDF type '{btype}' -> diffuse 0.5 fallback")
            body = "type diffuse  reflect 0.5"

        return self.add_material(name, body)

    def emitter_spd(self, emitter, name):
        """Emitter radiance -> FTSL spd. rgb radiance is unbounded in Mitsuba, but FTSL's
        `rgb` upsamples a [0,1] *reflectance*, so normalize an rgb emitter to max=1 to keep
        its hue (absolute brightness is handled by auto-exposure in relative mode)."""
        # find the radiance node to inspect its raw form
        node = None
        for c in emitter:
            if c.tag in ('rgb','srgb','spectrum','blackbody') and (c.get('name') == name or c.get('name') is None):
                node = c; break
        if node is not None and node.tag in ('rgb','srgb'):
            c = parse_floats(sub(node.get('value'), self.defaults))
            if len(c) == 1: c = c*3
            m = max(c)
            if m > 1.0:
                c = [x/m for x in c]
                warn("rgb emitter radiance normalized to max=1 (brightness set by auto-exposure)")
            return f"rgb {fnum(c[0])} {fnum(c[1])} {fnum(c[2])}"
        s = spectrum_from_node(node, self.defaults) if node is not None else None
        return s if s is not None else 'blackbody 6500'

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
                self.geometry.append(
                    f"light area {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  spd {spd} }}")
            else:
                mat = self.resolve_shape_material(shape) or self.default_mat()
                self.geometry.append(
                    f"quad {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  material {mat} }}")
        elif stype == 'cube':
            mat = self.resolve_shape_material(shape) or self.default_mat()
            if is_light:
                warn("cube with area emitter -> emitting quads not supported; geometry only")
            self.emit_cube(M, mat)
        elif stype == 'sphere':
            # center/radius from explicit props or from the transform of the unit sphere
            center = None; radius = None
            for c in shape:
                if c.tag == 'point' and c.get('name') == 'center':
                    center = [float(c.get('x',0)), float(c.get('y',0)), float(c.get('z',0))]
                if c.tag == 'float' and c.get('name') == 'radius':
                    radius = float(sub(c.get('value'), self.defaults))
            if center is None: center = xform_point(M, [0,0,0])
            if radius is None:
                radius = vlen(vsub(xform_point(M, [1,0,0]), xform_point(M, [0,0,0])))
            if is_light:
                self.geometry.append(
                    f"light sphere {{ center {v3(center)}  radius {fnum(radius)}  spd {spd} }}")
            else:
                mat = self.resolve_shape_material(shape) or self.default_mat()
                self.geometry.append(
                    f"sphere {{ center {v3(center)}  radius {fnum(radius)}  material {mat} }}")
        elif stype in ('obj', 'ply', 'serialized'):
            fn = get_string(shape, 'filename', self.defaults) or '?'
            mat = self.resolve_shape_material(shape) or self.default_mat()
            if stype != 'obj':
                warn(f"shape '{fn}' is {stype}; ftrace's mesh loader reads OBJ only - convert to .obj")
            if is_light:
                warn(f"mesh '{fn}' has an area emitter; FTSL has no emissive mesh -> emitting as lit geometry (light dropped)")
            self.emit_mesh(fn, mat, M)
        else:
            warn(f"unsupported shape type '{stype}' -> skipped")

    def emit_cube(self, M, mat):
        # 6 faces of [-1,1]^3 as quads, each transformed by M.
        faces = [
            ([-1,-1, 1],[ 1,-1, 1],[-1, 1, 1]),  # +z
            ([ 1,-1,-1],[-1,-1,-1],[ 1, 1,-1]),  # -z
            ([ 1,-1, 1],[ 1,-1,-1],[ 1, 1, 1]),  # +x
            ([-1,-1,-1],[-1,-1, 1],[-1, 1,-1]),  # -x
            ([-1, 1, 1],[ 1, 1, 1],[-1, 1,-1]),  # +y
            ([-1,-1,-1],[ 1,-1,-1],[-1,-1, 1]),  # -y
        ]
        for a,b,c in faces:
            o = xform_point(M, a); u = vsub(xform_point(M, b), o); v = vsub(xform_point(M, c), o)
            self.geometry.append(f"quad {{ origin {v3(o)}  u {v3(u)}  v {v3(v)}  material {mat} }}")

    def emit_mesh(self, fn, mat, M):
        # Decompose M into translate + (approx) uniform scale + note if rotated/sheared.
        t = xform_point(M, [0,0,0])
        sx = vlen(xform_vec(M, [1,0,0])); sy = vlen(xform_vec(M, [0,1,0])); sz = vlen(xform_vec(M, [0,0,1]))
        line = f'mesh {{ file "{fn}"  material {mat}  translate {v3(t)}'
        if abs(sx-1)>1e-6 or abs(sy-1)>1e-6 or abs(sz-1)>1e-6:
            if abs(sx-sy)<1e-4 and abs(sy-sz)<1e-4:
                line += f"  scale {fnum(sx)}"
            else:
                line += f"  scale {fnum(sx)} {fnum(sy)} {fnum(sz)}"
        # rotation/shear can't be expressed by mesh translate/scale alone
        if self.has_rotation(M):
            warn(f"mesh '{fn}' has a rotation/shear in to_world; ftrace mesh supports only "
                 f"translate+scale (+euler rotate) - rotation approximated as dropped")
        line += " }"
        self.geometry.append(line)

    def has_rotation(self, M):
        # off-diagonal of the upper-left 3x3 (after removing scale) far from 0 => rotated
        cols = [vnorm(xform_vec(M,[1,0,0])), vnorm(xform_vec(M,[0,1,0])), vnorm(xform_vec(M,[0,0,1]))]
        return abs(cols[0][1])>1e-3 or abs(cols[0][2])>1e-3 or abs(cols[1][0])>1e-3

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
        else:
            lines.append(f"    mode B")
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
        if etype == 'constant':
            spd = self.emitter_spd(emitter, 'radiance')
            self.geometry.append(f"light env {{ spd {spd} }}")
        elif etype == 'envmap':
            fn = get_string(emitter, 'filename', self.defaults) or 'env.hdr'
            scale = get_float(emitter, 'scale', 1.0, self.defaults)
            self.geometry.append(f'light env {{ file "{fn}"  intensity {fnum(scale)} }}')
        elif etype == 'point':
            warn("point emitter -> FTSL has no point light; approximated as a small sphere light")
            pos = None
            for c in emitter:
                if c.tag == 'point' and c.get('name') == 'position':
                    pos = [float(c.get('x',0)), float(c.get('y',0)), float(c.get('z',0))]
            if pos is None: pos = [0,1,0]
            spd = first_child_spectrum(emitter, 'intensity', self.defaults) or 'blackbody 6500'
            self.geometry.append(f"light sphere {{ center {v3(pos)}  radius 0.02  spd {spd} }}")
        elif etype == 'directional':
            warn("directional emitter -> collimated light")
            d = None
            for c in emitter:
                if c.tag == 'vector' and c.get('name') == 'direction':
                    d = [float(c.get('x',0)), float(c.get('y',0)), float(c.get('z',0))]
            if d is None: d = [0,0,-1]
            spd = first_child_spectrum(emitter, 'irradiance', self.defaults) or 'blackbody 6500'
            self.geometry.append(f"light collimated {{ dir {v3(d)}  spd {spd} }}")
        else:
            warn(f"unsupported scene emitter '{etype}' -> skipped")

    # --- driver -------------------------------------------------------------
    def convert(self, root):
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
        if not any('light' in g for g in self.geometry):
            out.append("# WARN: no lights found in source scene; adding a fallback so it renders")
            out.append("light area { origin -0.5 2 -0.5  u 1 0 0  v 0 0 1  normal 0 -1 0  spd preset:d65 }")
            out.append("")
        for cam in self.cameras:
            out.append(cam)
        if not self.cameras:
            out.append('# WARN: no sensor found; adding a default camera')
            out.append('camera "cam" { eye 0 1 4  look_at 0 1 0  up 0 1 0  fov_y 45  mode B  film { res 512 512 } }')
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
    with open(outp, 'w') as f:
        f.write(text)
    sys.stderr.write(f"[mitsuba_to_ftsl] wrote {outp}  "
                     f"({len(conv.mat_order)} materials, {len(conv.geometry)} geo/light, "
                     f"{len(conv.cameras)} camera)\n")

if __name__ == '__main__':
    main()
