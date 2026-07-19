// Procedural patterns: math-driven scalar fields for material properties.
//
// A Pattern maps a per-hit context (world point x,y,z; field value f; surface
// normal nx,ny,nz; radius r = |p|) to a single scalar. That scalar then drives a
// material property at the shading point — roughness, thin-film thickness, an
// emission/absorption scale, or the selection weight between two whole materials
// (so ANY property, including colour and material TYPE, can vary across a surface).
//
// Two authoring front-ends compile to the SAME representation:
//   (A) built-in GENERATORS — named patterns (axis, radial, bands, checker, noise,
//       field) with a few parameters; the parser expands them to postfix.
//   (B) an EXPRESSION — an infix math formula over the variables x y z f nx ny nz r
//       (e.g. "0.5 + 0.5*sin(10*x)") compiled by shunting-yard to postfix.
//
// Like the implicit field, a pattern is a FLAT ARRAY of POD nodes in POSTFIX order,
// evaluated by a tiny scalar stack — no pointers, no recursion, trivially GPU-
// portable (the CUDA backend iterates the same array).
#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include "linalg.h"
#include "pov_functions.h"   // exact POV-Ray internal isosurface functions (f_torus, ...)

// ---------------------------------------------------------------------------
// Postfix opcodes.
// ---------------------------------------------------------------------------
enum class PatOp : int {
    // nullary: push a value
    Const = 0,                       // push a (the node's literal)
    VarX, VarY, VarZ,                // world hit point
    VarF,                            // implicit field value at the hit (~0 on a surface)
    VarNx, VarNy, VarNz,             // surface normal (oriented against the ray)
    VarR,                            // radius = sqrt(x*x+y*y+z*z)
    VarU, VarV,                      // surface UV coordinates (mesh or native-primitive wrap)
    VarT,                            // flyby timeline t in [0,1] — ONLY in scope inside a
                                     // camera_curve record track (fov_from/roll_from/...);
                                     // deliberately placed AFTER VarV so patternHasFreeVars'
                                     // "surface intrinsic" range (VarX..VarV) excludes it.
    // unary: pop 1, push 1
    Neg, Abs, Sqrt, Sin, Cos, Tan, Exp, Log, Floor, Fract, Sign, Saturate,
    // binary: pop 2 (a below b), push 1
    Add, Sub, Mul, Div, Mod, Pow, Min, Max, Atan2, Step,
    // ternary: pop 3 (a,b,c bottom->top), push 1
    Clamp,        // clamp(x, lo, hi)
    Mix,          // lerp(a, b, t)
    Smoothstep,   // smoothstep(edge0, edge1, x)
    Noise,        // value noise of (x, y, z) in [0,1]
    // variadic built-in: an exact POV-Ray internal function. The node's `a` holds
    // the POV internal id (0..75); the evaluator pops povFnArity(id) args (the
    // first three are the coordinates) and pushes the returned scalar.
    PovFn,
};

// One postfix node. POD (no std:: members) so it uploads to the GPU verbatim.
struct PatNode {
    PatOp  op = PatOp::Const;
    double a  = 0.0;   // literal for Const (unused otherwise)
};

// Per-hit evaluation context (the pattern variables).
struct PatCtx {
    double x = 0, y = 0, z = 0;   // world point
    double f = 0;                 // implicit field value (0 for non-implicit hits)
    double nx = 0, ny = 0, nz = 0;// surface normal
    double r = 0;                 // radius |p|
    double u = 0, v = 0;          // surface UV (mesh interpolated or native-primitive wrap)
    double t = 0;                 // flyby timeline in [0,1] (camera_curve record tracks only)
};

inline PatCtx makePatCtx(const Vec3& p, double f, const Vec3& n, double u = 0, double v = 0) {
    PatCtx c;
    c.x = p.x; c.y = p.y; c.z = p.z;
    c.f = f;
    c.nx = n.x; c.ny = n.y; c.nz = n.z;
    c.r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    c.u = u; c.v = v;
    return c;
}

// ---- 3-D value noise (hash lattice + trilinear smoothstep fade) -------------
// Deterministic integer hash so CPU and GPU agree bit-for-bit. Output in [0,1].
inline double patHash3(int ix, int iy, int iz) {
    uint32_t h = (uint32_t)ix * 374761393u + (uint32_t)iy * 668265263u
               + (uint32_t)iz * 2147483647u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h ^= (h >> 16);
    return (double)h / 4294967295.0;   // [0,1]
}
inline double patValueNoise(double x, double y, double z) {
    double fx = std::floor(x), fy = std::floor(y), fz = std::floor(z);
    int ix = (int)fx, iy = (int)fy, iz = (int)fz;
    double tx = x - fx, ty = y - fy, tz = z - fz;
    // smoothstep fade
    double ux = tx * tx * (3.0 - 2.0 * tx);
    double uy = ty * ty * (3.0 - 2.0 * ty);
    double uz = tz * tz * (3.0 - 2.0 * tz);
    double c000 = patHash3(ix,     iy,     iz);
    double c100 = patHash3(ix + 1, iy,     iz);
    double c010 = patHash3(ix,     iy + 1, iz);
    double c110 = patHash3(ix + 1, iy + 1, iz);
    double c001 = patHash3(ix,     iy,     iz + 1);
    double c101 = patHash3(ix + 1, iy,     iz + 1);
    double c011 = patHash3(ix,     iy + 1, iz + 1);
    double c111 = patHash3(ix + 1, iy + 1, iz + 1);
    double x00 = c000 + (c100 - c000) * ux;
    double x10 = c010 + (c110 - c010) * ux;
    double x01 = c001 + (c101 - c001) * ux;
    double x11 = c011 + (c111 - c011) * ux;
    double y0  = x00 + (x10 - x00) * uy;
    double y1  = x01 + (x11 - x01) * uy;
    return y0 + (y1 - y0) * uz;
}

// ---- postfix evaluator ------------------------------------------------------
inline double patternEval(const PatNode* nodes, int n, const PatCtx& c) {
    double st[64];
    int sp = 0;
    for (int i = 0; i < n; ++i) {
        const PatNode& nd = nodes[i];
        switch (nd.op) {
            case PatOp::Const:    st[sp++] = nd.a; break;
            case PatOp::VarX:     st[sp++] = c.x;  break;
            case PatOp::VarY:     st[sp++] = c.y;  break;
            case PatOp::VarZ:     st[sp++] = c.z;  break;
            case PatOp::VarF:     st[sp++] = c.f;  break;
            case PatOp::VarNx:    st[sp++] = c.nx; break;
            case PatOp::VarNy:    st[sp++] = c.ny; break;
            case PatOp::VarNz:    st[sp++] = c.nz; break;
            case PatOp::VarR:     st[sp++] = c.r;  break;
            case PatOp::VarU:     st[sp++] = c.u;  break;
            case PatOp::VarV:     st[sp++] = c.v;  break;
            case PatOp::VarT:     st[sp++] = c.t;  break;
            case PatOp::Neg:      st[sp-1] = -st[sp-1]; break;
            case PatOp::Abs:      st[sp-1] = std::fabs(st[sp-1]); break;
            case PatOp::Sqrt:     st[sp-1] = std::sqrt(std::fmax(0.0, st[sp-1])); break;
            case PatOp::Sin:      st[sp-1] = std::sin(st[sp-1]); break;
            case PatOp::Cos:      st[sp-1] = std::cos(st[sp-1]); break;
            case PatOp::Tan:      st[sp-1] = std::tan(st[sp-1]); break;
            case PatOp::Exp:      st[sp-1] = std::exp(st[sp-1]); break;
            case PatOp::Log:      st[sp-1] = std::log(std::fmax(1e-300, st[sp-1])); break;
            case PatOp::Floor:    st[sp-1] = std::floor(st[sp-1]); break;
            case PatOp::Fract:    st[sp-1] = st[sp-1] - std::floor(st[sp-1]); break;
            case PatOp::Sign:     st[sp-1] = (st[sp-1] > 0.0) - (st[sp-1] < 0.0); break;
            case PatOp::Saturate: st[sp-1] = std::fmin(1.0, std::fmax(0.0, st[sp-1])); break;
            case PatOp::Add:      { double b = st[--sp]; st[sp-1] += b; break; }
            case PatOp::Sub:      { double b = st[--sp]; st[sp-1] -= b; break; }
            case PatOp::Mul:      { double b = st[--sp]; st[sp-1] *= b; break; }
            case PatOp::Div:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] / b : 0.0; break; }
            case PatOp::Mod:      { double b = st[--sp]; st[sp-1] = (b != 0.0) ? st[sp-1] - b * std::floor(st[sp-1] / b) : 0.0; break; }
            case PatOp::Pow:      { double b = st[--sp]; st[sp-1] = std::pow(st[sp-1], b); break; }
            case PatOp::Min:      { double b = st[--sp]; st[sp-1] = std::fmin(st[sp-1], b); break; }
            case PatOp::Max:      { double b = st[--sp]; st[sp-1] = std::fmax(st[sp-1], b); break; }
            case PatOp::Atan2:    { double b = st[--sp]; st[sp-1] = std::atan2(st[sp-1], b); break; }
            case PatOp::Step:     { double b = st[--sp]; st[sp-1] = (b >= st[sp-1]) ? 1.0 : 0.0; break; }
            case PatOp::Clamp:    { double hi = st[--sp], lo = st[--sp]; st[sp-1] = std::fmin(hi, std::fmax(lo, st[sp-1])); break; }
            case PatOp::Mix:      { double t = st[--sp], b = st[--sp]; st[sp-1] = st[sp-1] + (b - st[sp-1]) * t; break; }
            case PatOp::Smoothstep: {
                double xx = st[--sp], e1 = st[--sp], e0 = st[sp-1];
                double tt = (e1 != e0) ? (xx - e0) / (e1 - e0) : 0.0;
                tt = std::fmin(1.0, std::fmax(0.0, tt));
                st[sp-1] = tt * tt * (3.0 - 2.0 * tt);
                break;
            }
            case PatOp::Noise:    { double zz = st[--sp], yy = st[--sp]; st[sp-1] = patValueNoise(st[sp-1], yy, zz); break; }
            case PatOp::PovFn: {
                int id = (int)nd.a;
                int na = povFnArity(id);
                double args[POV_FN_MAX_ARGS];
                for (int k = na - 1; k >= 0; --k) args[k] = st[--sp];
                st[sp++] = povFnEval(id, args);
                break;
            }
        }
    }
    return sp > 0 ? st[0] : 0.0;
}

// A named, storable pattern (owns its postfix program). Scene::patterns holds these.
struct Pattern {
    std::vector<PatNode> nodes;
    double eval(const PatCtx& c) const { return patternEval(nodes.data(), (int)nodes.size(), c); }
};

// ---------------------------------------------------------------------------
// (B) Expression compiler: infix math -> postfix, over the pattern variables.
// Supports: literals, constant `pi`; variables x y z f nx ny nz r u v; unary + -;
// binary + - * / % ^ (^ = pow, right-assoc); functions abs sqrt sin cos tan exp
// log floor fract sign saturate min max pow atan2 step clamp mix smoothstep noise.
// Returns false + fills `err` on a parse error. Shunting-yard with an operator and
// an output (postfix) queue; function arity is checked at the closing paren.
// ---------------------------------------------------------------------------
namespace pattern_detail {

struct Tok {
    enum Kind { Num, Var, Func, Op, LParen, RParen, Comma } kind;
    double num = 0;
    PatOp  var = PatOp::VarX;   // for Var
    std::string name;          // for Func
    char op = 0;               // for Op ('+','-','*','/','%','^','u' unary minus)
};

inline bool isIdentStart(char c) { return std::isalpha((unsigned char)c) || c == '_'; }
inline bool isIdentCh(char c)    { return std::isalnum((unsigned char)c) || c == '_'; }

// Map an identifier to a variable opcode; returns false if it isn't a variable.
inline bool varOp(const std::string& s, PatOp& out) {
    if (s == "x")  { out = PatOp::VarX;  return true; }
    if (s == "y")  { out = PatOp::VarY;  return true; }
    if (s == "z")  { out = PatOp::VarZ;  return true; }
    if (s == "f")  { out = PatOp::VarF;  return true; }
    if (s == "nx") { out = PatOp::VarNx; return true; }
    if (s == "ny") { out = PatOp::VarNy; return true; }
    if (s == "nz") { out = PatOp::VarNz; return true; }
    if (s == "r")  { out = PatOp::VarR;  return true; }
    if (s == "u")  { out = PatOp::VarU;  return true; }
    if (s == "v")  { out = PatOp::VarV;  return true; }
    return false;
}

// Function name -> (opcode, arity[, povId]). Returns false if not a known function.
// For exact POV-Ray internal functions (f_torus, f_heart, ...) out=PatOp::PovFn and
// povId is the POV internal id; for built-ins povId is left as -1.
inline bool funcOp(const std::string& s, PatOp& out, int& arity, int& povId) {
    povId = -1;
    struct F { const char* n; PatOp op; int ar; };
    static const F fs[] = {
        {"abs",PatOp::Abs,1},{"sqrt",PatOp::Sqrt,1},{"sin",PatOp::Sin,1},
        {"cos",PatOp::Cos,1},{"tan",PatOp::Tan,1},{"exp",PatOp::Exp,1},
        {"log",PatOp::Log,1},{"floor",PatOp::Floor,1},{"fract",PatOp::Fract,1},
        {"sign",PatOp::Sign,1},{"saturate",PatOp::Saturate,1},
        {"min",PatOp::Min,2},{"max",PatOp::Max,2},{"pow",PatOp::Pow,2},
        {"atan2",PatOp::Atan2,2},{"step",PatOp::Step,2},
        {"clamp",PatOp::Clamp,3},{"mix",PatOp::Mix,3},
        {"smoothstep",PatOp::Smoothstep,3},{"noise",PatOp::Noise,3},
    };
    for (const F& g : fs) if (s == g.n) { out = g.op; arity = g.ar; return true; }
    int id, ar;
    if (povFnLookup(s.c_str(), id, ar)) { out = PatOp::PovFn; arity = ar; povId = id; return true; }
    return false;
}
inline bool funcOp(const std::string& s, PatOp& out, int& arity) {
    int dummy; return funcOp(s, out, arity, dummy);
}

inline int opPrec(char c) {
    switch (c) {
        case 'u': return 5;              // unary minus
        case '^': return 4;
        case '*': case '/': case '%': return 3;
        case '+': case '-': return 2;
        default:  return 0;
    }
}
inline bool opRightAssoc(char c) { return c == '^' || c == 'u'; }
inline PatOp binOp(char c) {
    switch (c) {
        case '+': return PatOp::Add; case '-': return PatOp::Sub;
        case '*': return PatOp::Mul; case '/': return PatOp::Div;
        case '%': return PatOp::Mod; case '^': return PatOp::Pow;
        default:  return PatOp::Add;
    }
}

inline bool tokenize(const std::string& s, std::vector<Tok>& out, std::string& err, bool allowT = false) {
    size_t i = 0, n = s.size();
    bool prevValue = false;   // was the previous token a value/RParen (for unary minus)
    while (i < n) {
        char c = s[i];
        if (std::isspace((unsigned char)c)) { ++i; continue; }
        if (std::isdigit((unsigned char)c) || (c == '.' && i + 1 < n && std::isdigit((unsigned char)s[i+1]))) {
            size_t j = i;
            while (j < n && (std::isdigit((unsigned char)s[j]) || s[j] == '.' ||
                             s[j] == 'e' || s[j] == 'E' ||
                             ((s[j] == '+' || s[j] == '-') && j > i && (s[j-1] == 'e' || s[j-1] == 'E'))))
                ++j;
            Tok t; t.kind = Tok::Num; t.num = std::atof(s.substr(i, j - i).c_str());
            out.push_back(t); i = j; prevValue = true; continue;
        }
        if (isIdentStart(c)) {
            size_t j = i;
            while (j < n && isIdentCh(s[j])) ++j;
            std::string id = s.substr(i, j - i);
            i = j;
            // skip spaces to see if a '(' follows -> function call
            size_t k = i; while (k < n && std::isspace((unsigned char)s[k])) ++k;
            bool isCall = (k < n && s[k] == '(');
            PatOp vop; PatOp fop; int ar;
            if (isCall && funcOp(id, fop, ar)) {
                Tok t; t.kind = Tok::Func; t.name = id; out.push_back(t);
                prevValue = false; continue;
            }
            if (id == "pi") {
                Tok t; t.kind = Tok::Num; t.num = 3.14159265358979323846;
                out.push_back(t); prevValue = true; continue;
            }
            if (id == "t") {
                // The flyby timeline is in scope ONLY inside a camera_curve record track.
                // Everywhere else `t` is out of scope; report that specifically rather than
                // as a bare "unknown identifier" so the scope rule is legible.
                if (allowT) { Tok t; t.kind = Tok::Var; t.var = PatOp::VarT; out.push_back(t); prevValue = true; continue; }
                err = "variable 't' (flyby timeline) is only in scope inside a camera_curve "
                      "record track (fov_from/roll_from/zoom_from/fstop_from/focus_from)"; return false;
            }
            if (varOp(id, vop)) {
                Tok t; t.kind = Tok::Var; t.var = vop; out.push_back(t);
                prevValue = true; continue;
            }
            err = "unknown identifier '" + id + "'"; return false;
        }
        if (c == '(') { Tok t; t.kind = Tok::LParen; out.push_back(t); ++i; prevValue = false; continue; }
        if (c == ')') { Tok t; t.kind = Tok::RParen; out.push_back(t); ++i; prevValue = true;  continue; }
        if (c == ',') { Tok t; t.kind = Tok::Comma;  out.push_back(t); ++i; prevValue = false; continue; }
        if (c == '+' || c == '-' || c == '*' || c == '/' || c == '%' || c == '^') {
            Tok t; t.kind = Tok::Op;
            t.op = (c == '-' && !prevValue) ? 'u' : c;   // leading '-' => unary
            if (c == '+' && !prevValue) { ++i; prevValue = false; continue; }  // unary + is a no-op
            out.push_back(t); ++i; prevValue = false; continue;
        }
        err = std::string("unexpected character '") + c + "'"; return false;
    }
    return true;
}

}  // namespace pattern_detail

// Compile an infix expression string into a postfix pattern program.
// `allowT` publishes the flyby-timeline variable `t` as in-scope (camera_curve record
// tracks only). Default false: every other call site keeps `t` an out-of-scope error,
// so a surface/constant driver can never silently read the timeline.
inline bool compilePatternExpr(const std::string& expr, std::vector<PatNode>& out, std::string& err, bool allowT = false) {
    using namespace pattern_detail;
    std::vector<Tok> toks;
    if (!tokenize(expr, toks, err, allowT)) return false;

    std::vector<PatNode> queue;             // output (postfix)
    std::vector<Tok>     ops;               // operator stack (Op / Func / LParen)
    std::vector<int>     argCount;          // arg counter per open function
    std::vector<bool>    wasFunc;           // parallel to LParen: did a func precede it

    auto emitOp = [&](const Tok& t) {
        if (t.kind == Tok::Op) {
            if (t.op == 'u') { PatNode nd; nd.op = PatOp::Neg; queue.push_back(nd); }
            else             { PatNode nd; nd.op = binOp(t.op); queue.push_back(nd); }
        } else if (t.kind == Tok::Func) {
            PatOp op; int ar; int povId; funcOp(t.name, op, ar, povId);
            PatNode nd; nd.op = op; if (op == PatOp::PovFn) nd.a = (double)povId; queue.push_back(nd);
        }
    };

    for (size_t ti = 0; ti < toks.size(); ++ti) {
        const Tok& t = toks[ti];
        switch (t.kind) {
            case Tok::Num:  { PatNode nd; nd.op = PatOp::Const; nd.a = t.num; queue.push_back(nd); break; }
            case Tok::Var:  { PatNode nd; nd.op = t.var; queue.push_back(nd); break; }
            case Tok::Func: ops.push_back(t); break;
            case Tok::Comma: {
                while (!ops.empty() && ops.back().kind != Tok::LParen) { emitOp(ops.back()); ops.pop_back(); }
                if (ops.empty()) { err = "misplaced comma"; return false; }
                if (!argCount.empty()) argCount.back()++;
                break;
            }
            case Tok::Op: {
                while (!ops.empty() && ops.back().kind == Tok::Op) {
                    char o2 = ops.back().op;
                    if ((opRightAssoc(t.op) && opPrec(t.op) < opPrec(o2)) ||
                        (!opRightAssoc(t.op) && opPrec(t.op) <= opPrec(o2))) {
                        emitOp(ops.back()); ops.pop_back();
                    } else break;
                }
                ops.push_back(t);
                break;
            }
            case Tok::LParen: {
                ops.push_back(t);
                bool fn = (!ops.empty() && ops.size() >= 2 && ops[ops.size()-2].kind == Tok::Func);
                wasFunc.push_back(fn);
                argCount.push_back(1);
                break;
            }
            case Tok::RParen: {
                while (!ops.empty() && ops.back().kind != Tok::LParen) { emitOp(ops.back()); ops.pop_back(); }
                if (ops.empty()) { err = "mismatched ')'"; return false; }
                ops.pop_back();   // discard LParen
                int args = argCount.empty() ? 0 : argCount.back();
                bool fn = wasFunc.empty() ? false : wasFunc.back();
                if (!argCount.empty()) argCount.pop_back();
                if (!wasFunc.empty())  wasFunc.pop_back();
                if (!ops.empty() && ops.back().kind == Tok::Func) {
                    PatOp op; int ar; funcOp(ops.back().name, op, ar);
                    if (fn && args != ar) {
                        err = "function '" + ops.back().name + "' expects " + std::to_string(ar) +
                              " arg(s), got " + std::to_string(args); return false;
                    }
                    emitOp(ops.back()); ops.pop_back();
                }
                break;
            }
        }
    }
    while (!ops.empty()) {
        if (ops.back().kind == Tok::LParen) { err = "mismatched '('"; return false; }
        emitOp(ops.back()); ops.pop_back();
    }
    if (queue.empty()) { err = "empty expression"; return false; }
    out = std::move(queue);
    return true;
}

// True if a compiled pattern program references any per-hit surface intrinsic
// (x y z f nx ny nz r u v). Used by value sites that must be load-time constant
// (records stage 5a scope check): a constant site has no per-hit context, so it
// admits only var-free (constant) drivers — a `R.chan(u)` there is a scope error.
inline bool patternHasFreeVars(const std::vector<PatNode>& prog) {
    for (const PatNode& nd : prog)
        if (nd.op >= PatOp::VarX && nd.op <= PatOp::VarV) return true;
    return false;
}

// ---------------------------------------------------------------------------
// (A) Built-in generators -> postfix. Each returns a ready pattern program.
// These are thin sugar over the expression VM so both front-ends share evaluation.
// ---------------------------------------------------------------------------
namespace pattern_gen {

// helper builders
inline void pushConst(std::vector<PatNode>& o, double v) { PatNode n; n.op = PatOp::Const; n.a = v; o.push_back(n); }
inline void pushOp(std::vector<PatNode>& o, PatOp op)     { PatNode n; n.op = op; o.push_back(n); }

// axis: a coordinate (x|y|z) remapped by scale/offset -> value = coord*scale+offset
inline std::vector<PatNode> axis(PatOp coord, double scale, double offset) {
    std::vector<PatNode> o;
    pushOp(o, coord); pushConst(o, scale); pushOp(o, PatOp::Mul);
    pushConst(o, offset); pushOp(o, PatOp::Add);
    return o;
}

// radial: |p - center| * scale (a spherical gradient about `center`)
inline std::vector<PatNode> radial(const Vec3& c, double scale) {
    std::vector<PatNode> o;
    // sqrt((x-cx)^2 + (y-cy)^2 + (z-cz)^2) * scale
    pushOp(o, PatOp::VarX); pushConst(o, c.x); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow);
    pushOp(o, PatOp::VarY); pushConst(o, c.y); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::VarZ); pushConst(o, c.z); pushOp(o, PatOp::Sub);
    pushConst(o, 2.0); pushOp(o, PatOp::Pow); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::Sqrt);
    pushConst(o, scale); pushOp(o, PatOp::Mul);
    return o;
}

// bands: 0.5 + 0.5*sin(2*pi*freq * (coord) + phase) -> smooth stripes in [0,1]
inline std::vector<PatNode> bands(PatOp coord, double freq, double phase) {
    std::vector<PatNode> o;
    pushOp(o, coord); pushConst(o, 2.0 * 3.14159265358979323846 * freq); pushOp(o, PatOp::Mul);
    pushConst(o, phase); pushOp(o, PatOp::Add); pushOp(o, PatOp::Sin);
    pushConst(o, 0.5); pushOp(o, PatOp::Mul); pushConst(o, 0.5); pushOp(o, PatOp::Add);
    return o;
}

// checker: 3-D checkerboard parity in {0,1} at the given cell size.
// value = mod(floor(x/s)+floor(y/s)+floor(z/s), 2)
inline std::vector<PatNode> checker(double size) {
    std::vector<PatNode> o;
    double inv = (size != 0.0) ? 1.0 / size : 1.0;
    pushOp(o, PatOp::VarX); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor);
    pushOp(o, PatOp::VarY); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor); pushOp(o, PatOp::Add);
    pushOp(o, PatOp::VarZ); pushConst(o, inv); pushOp(o, PatOp::Mul); pushOp(o, PatOp::Floor); pushOp(o, PatOp::Add);
    pushConst(o, 2.0); pushOp(o, PatOp::Mod);
    return o;
}

// noise: value noise of (p*freq) in [0,1].
inline std::vector<PatNode> noise(double freq) {
    std::vector<PatNode> o;
    pushOp(o, PatOp::VarX); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::VarY); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::VarZ); pushConst(o, freq); pushOp(o, PatOp::Mul);
    pushOp(o, PatOp::Noise);
    return o;
}

// field: the raw implicit field value f (near 0 on a surface; useful volumetrically).
inline std::vector<PatNode> field(double scale) {
    std::vector<PatNode> o;
    pushOp(o, PatOp::VarF); pushConst(o, scale); pushOp(o, PatOp::Mul);
    return o;
}

}  // namespace pattern_gen
