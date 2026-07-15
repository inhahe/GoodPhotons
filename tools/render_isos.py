"""CLI launcher for tools/iso_render.html — the headless WebGL isosurface raymarcher.

Renders random isosurfaces from one of three generators and writes PNGs (plus a
functions.md mapping each image to the exact field function, and a params.txt) into
a per-approach directory under png/iso_gen/.

Usage:
    python tools/render_isos.py <approach> [--count N] [--seed S] [--size PX]
                                           [--steps K] [--cpu]
    <approach> = trig | sdf | super | all

Renders on the real GPU (ANGLE/D3D11) by default; pass --cpu to force the CPU
SwiftShader fallback (much slower, only needed if the GPU path fails).

Examples:
    python tools/render_isos.py all                 # 12 of each, default size
    python tools/render_isos.py trig --count 16 --size 560
    python tools/render_isos.py sdf --seed 100 --cpu

It launches headless Chrome on iso_render.html?batch=1&..., waits for the page to
finish (title -> DONE), reads the JSON blob it dumps into #out, and decodes the
base64 PNGs to disk. Chrome is found automatically (Windows install paths)."""
import argparse, base64, html, json, os, re, subprocess, sys, tempfile, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PAGE = os.path.join(ROOT, 'tools', 'iso_render.html')

APPROACHES = {
    'trig':  ('01_symmetric_trig', 'Symmetric trigonometric / minimal-surface / Chmutov family'),
    'sdf':   ('02_sdf_assembly',   'Signed-distance CSG assembly (smooth-min of primitives)'),
    'super': ('03_superformula',   'Superformula (Gielis) implicit radial field'),
}

def find_chrome():
    for p in [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
              r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
              os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe'),
              r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']:
        if os.path.exists(p):
            return p
    sys.exit('Chrome/Edge not found — install Chrome or edit find_chrome().')

CHROME = find_chrome()

def file_url(path, query):
    u = 'file:///' + os.path.abspath(path).replace('\\', '/')
    return u + '?' + urllib.parse.urlencode(query)

def run_headless(query, gpu, timeout=600):
    url = file_url(PAGE, query)
    gpu_flags = (['--use-angle=d3d11', '--enable-unsafe-swiftshader']
                 if gpu else
                 ['--disable-gpu', '--enable-unsafe-swiftshader',
                  '--use-gl=angle', '--use-angle=swiftshader'])
    with tempfile.TemporaryDirectory() as prof:   # fresh profile avoids default-profile lock flakiness
        cmd = [CHROME, '--headless=new', f'--user-data-dir={prof}', *gpu_flags,
               '--virtual-time-budget=120000', '--dump-dom', url]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout

def extract_out(dom):
    m = re.search(r'<pre id="out"[^>]*>(.*?)</pre>', dom, re.S)
    if not m:
        raise SystemExit('could not find #out in DOM (render failed to complete)')
    txt = html.unescape(m.group(1))
    return json.loads(txt)

def render_approach(key, count, seed, size, steps, gpu):
    dirname, desc = APPROACHES[key]
    outdir = os.path.join('png', 'iso_gen', dirname)
    os.makedirs(outdir, exist_ok=True)
    print(f'[{key}] rendering {count} shapes (seed {seed}, {size}px) -> {outdir}')
    dom = run_headless({'batch': 1, 'approach': key, 'count': count, 'seed': seed,
                        'w': size, 'h': size, 'steps': steps}, gpu)
    data = extract_out(dom)
    if data.get('error'):
        raise SystemExit(f'renderer error: {data["error"]}')
    shapes = data['shapes']
    md = [f'# {dirname}', '', f'_{desc}_', '',
          f'{len(shapes)} shapes, base seed {seed}, {size}px. Field is `F(x,y,z)`; '
          f'surface at F=0. `expr` is ready for an ftrace isosurface `function {{ expr "..." }}`.',
          '', '| view | seed | name | ftrace expr |', '|---|---|---|---|']
    kept = 0
    for sh in shapes:
        if 'png' not in sh:
            print(f'   seed {sh["seed"]}: ERROR {sh.get("error","")[:80]}')
            continue
        b64 = sh['png'].split(',', 1)[1]
        fn = f'shape_{sh["seed"]:03d}.png'
        with open(os.path.join(outdir, fn), 'wb') as f:
            f.write(base64.b64decode(b64))
        kept += 1
        expr = sh['expr'].replace('|', r'\|')
        md.append(f'| ![]({fn}) | {sh["seed"]} | {sh["name"]} | `{expr}` |')
    with open(os.path.join(outdir, 'functions.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(outdir, 'params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'approach   : {key}  ({dirname})\n{desc}\n\n')
        f.write(f'count      : {count}\nbase seed  : {seed}\nsize       : {size}px\n'
                f'raymarch   : {steps} steps\nbackend    : {"gpu(angle-d3d11)" if gpu else "swiftshader(cpu)"}\n')
    print(f'[{key}] wrote {kept} PNGs + functions.md + params.txt')
    return outdir, [sh for sh in shapes if 'png' in sh]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('approach', choices=list(APPROACHES) + ['all'])
    ap.add_argument('--count', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--size', type=int, default=480)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--cpu', action='store_true',
                    help='force CPU SwiftShader (default is real GPU via ANGLE/D3D11, ~3-10x faster)')
    a = ap.parse_args()
    keys = list(APPROACHES) if a.approach == 'all' else [a.approach]
    gpu = not a.cpu
    for k in keys:
        render_approach(k, a.count, a.seed, a.size, a.steps, gpu)

if __name__ == '__main__':
    main()
