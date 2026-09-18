// groom.h -- the hair-authoring tool's MODEL of a scene's `curve` blocks (0.330.0).
//
// The loader flattens a curve tree into strands and forgets the authored points. The tool needs
// the tree AS WRITTEN -- every `point` with its own tokens, every child in order, every other
// statement verbatim -- so that a file it saves reads back into exactly the strands it showed,
// and so that a point nobody touched is rewritten with the very characters it was authored with.
// Plain C++ on ftsl::Block; nothing here knows about a GUI. `ftrace -groom-rewrite <in> <out>`
// is the headless round trip tools/groom_rig.py checks: rewrite a scene through the model, and
// `-dumpcurves` of the two must be byte-identical.
//
// What is written back: files whose every top-level block is a `curve` or a `group` of curves
// (with the group's own `translate` / `rotate` / `scale` kept). A file with anything else in it
// is not rewritten -- keep hair in its own included file, which is what `include` is for.
#pragma once

#include "ftsl.h"

#include "assetbytes.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace groom {

// ---- tokens ---------------------------------------------------------------------------------
// The shortest decimal that reads back to the same double (std::to_chars): an edited point is
// written exactly, without the 17-digit tails `%.17g` would leave.
inline std::string fmtNum(double v) {
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    return r.ec == std::errc() ? std::string(buf, r.ptr) : std::to_string(v);
}
inline bool isNumberTok(const std::string& w) {
    if (w.empty()) return false;
    char* e = nullptr;
    std::strtod(w.c_str(), &e);
    return e && *e == '\0';
}
// A token as the grammar will read it back: bare when it is a plain word or a number, quoted
// otherwise (or when forced -- names and paths are always quoted).
inline std::string tok(const std::string& w, bool forceQuote) {
    bool need = forceQuote || w.empty();
    if (!need)
        for (char c : w)
            if (!(std::isalnum((unsigned char)c) || c == '_' || c == '.' || c == '-' || c == '+' || c == ':')) { need = true; break; }
    if (!need) return w;
    std::string q = "\"";
    for (char c : w) { if (c == '"' || c == '\\') q += '\\'; q += c; }
    return q + "\"";
}
// Keys whose value is a name or a path and is written quoted, as scenes write them.
inline bool stringKey(const std::string& k) {
    return k == "curve" || k == "on" || k == "guides" || k == "file" || k == "object" || k == "mesh" || k == "include";
}

// ---- the general writer: any Block back to text ----------------------------------------------
inline bool writeStmt(std::string& out, const ftsl::Stmt& s, int indent, std::string& err);

inline bool writeBody(std::string& out, const ftsl::Block& b, int indent, std::string& err) {
    const std::string pad((size_t)indent * 4, ' ');
    if (b.stmts.empty() && !b.words.empty()) {          // a token list (table / palette)
        out += pad;
        for (size_t i = 0; i < b.words.size(); ++i) { if (i) out += ' '; out += tok(b.words[i], false); }
        out += '\n';
        return true;
    }
    for (const ftsl::Stmt& s : b.stmts) if (!writeStmt(out, s, indent, err)) return false;
    return true;
}
inline bool writeStmt(std::string& out, const ftsl::Stmt& s, int indent, std::string& err) {
    const std::string pad((size_t)indent * 4, ' ');
    if (s.val.array) {
        err = "`" + s.key + "` (line " + std::to_string(s.line) + "): an inline array literal, which the groom writer cannot re-emit";
        return false;
    }
    out += pad + s.key;
    const bool sk = stringKey(s.key);
    for (const std::string& w : s.val.words) out += " " + tok(w, sk && !isNumberTok(w));
    if (s.val.block) {
        if (!s.val.block->name.empty()) out += " " + tok(s.val.block->name, true);
        out += " {\n";
        if (!writeBody(out, *s.val.block, indent + 1, err)) return false;
        out += pad + "}\n";
    } else {
        out += '\n';
    }
    return true;
}
inline bool writeBlock(std::string& out, const ftsl::Block& b, int indent, std::string& err) {
    const std::string pad((size_t)indent * 4, ' ');
    if (b.type == "include") { out += pad + "include " + tok(b.name, true) + "\n"; return true; }
    if (b.type == "prefer") {
        for (size_t i = 0; i < b.branches.size(); ++i) {
            out += (i == 0) ? pad + "prefer {\n" : pad + "} else {\n";
            for (const ftsl::Block& c : b.branches[i]) if (!writeBlock(out, c, indent + 1, err)) return false;
        }
        out += pad + "}\n";
        return true;
    }
    out += pad + b.type;
    if (!b.subtype.empty()) out += " " + b.subtype;
    if (!b.name.empty()) out += " " + tok(b.name, true);
    out += " {\n";
    if (!writeBody(out, b, indent + 1, err)) return false;
    out += pad + "}\n";
    return true;
}

// ---- the model ------------------------------------------------------------------------------
struct Pt {
    Vec3   p{0, 0, 0};
    double r = 0.0;                     // the point's own `r=` radius, when it had one
    bool   haveR = false;
    std::vector<std::string> extra;     // any other `key=value` tokens the point carried
    std::vector<std::string> raw;       // the tokens as written; re-emitted verbatim while untouched
    bool   edited = false;
};

// One `curve` node: a strand (points), a curve of curves (children), or a reference by name.
struct Node {
    int  id = 0;                        // stable for the session; the GUI selects by it
    std::string name;                   // the block's name, or the referenced name when `ref`
    bool ref = false;                   // `curve "name"` with no block
    std::vector<Pt>   pts;              // a strand's control points, in order
    std::vector<Node> kids;             // inline children, in order
    std::vector<ftsl::Stmt> other;      // every other statement, verbatim: count, closed, spline, material ...

    const ftsl::Stmt* find(const char* key) const { for (const auto& s : other) if (s.key == key) return &s; return nullptr; }
    ftsl::Stmt*       find(const char* key)       { for (auto& s : other) if (s.key == key) return &s; return nullptr; }
    bool placed()   const { return find("count") || find("density") || find("density_at"); }
    bool rendered() const { return find("material") != nullptr; }
    bool closed() const {
        const ftsl::Stmt* c = find("closed");
        if (!c) return false;
        if (c->val.words.empty()) return true;
        const std::string& w = c->val.words[0];
        return !(w == "off" || w == "false" || w == "0" || w == "no");
    }
    int count() const {
        const ftsl::Stmt* c = find("count");
        return (c && !c->val.words.empty()) ? std::atoi(c->val.words[0].c_str()) : 0;
    }
    double alpha() const {                       // `spline`, as the loader reads it
        const ftsl::Stmt* s = find("spline");
        if (!s || s->val.words.empty()) return 0.0;
        const std::string& w = s->val.words[0];
        if (w == "uniform") return 0.0;
        if (w == "centripetal") return 0.5;
        if (w == "chordal") return 1.0;
        return isNumberTok(w) ? std::atof(w.c_str()) : 0.0;
    }
};

// A top-level block of a file: a curve, a group of them (the group's own statements kept), or a
// `fur` block -- kept whole and edited statement by statement; in a file the tool does not
// rewrite it is patched into its own text span on save (Phase 4).
struct Entry {
    int  id = 0;
    bool group = false;
    bool fur = false;
    std::string name;                   // the group's or fur block's name
    std::vector<ftsl::Stmt> other;      // the group's statements: translate / rotate / scale ...
    std::vector<Entry> items;           // the group's members
    Node curve;                         // when neither group nor fur
    ftsl::Block furBlock;               // when fur
    bool furDirty = false;              // edited since the last save
};

struct FileModel {
    std::string path;
    std::vector<Entry> entries;         // in block order
    bool writable = true;               // every block is a curve or a group of curves
    std::string why;                    // when not
    std::string leading;                // the file's opening comment, kept on rewrite
    bool dirty = false;
};

struct Model {
    std::vector<FileModel> files;
    int nextId = 1;
    FileModel* fileFor(const std::string& path) {
        for (auto& f : files) if (f.path == path) return &f;
        files.push_back(FileModel());
        files.back().path = path;
        return &files.back();
    }
};

inline Node nodeFromBlock(const ftsl::Block& b, Model& m) {
    Node n;
    n.id = m.nextId++;
    n.name = b.name;
    for (const ftsl::Stmt& s : b.stmts) {
        if (s.key == "point") {
            // `point x y z [r=<radius>] [key=value ...]`, as curveLeaf reads it
            Pt p;
            p.raw = s.val.words;
            double v[3] = { 0, 0, 0 };
            for (size_t i = 0; i < s.val.words.size(); ++i) {
                const std::string& w = s.val.words[i];
                if (i < 3) { if (isNumberTok(w)) v[i] = std::atof(w.c_str()); continue; }
                const size_t eq = w.find('=');
                const std::string key = (eq == std::string::npos) ? std::string() : w.substr(0, eq);
                if (key == "r" || key == "radius") { p.r = std::atof(w.c_str() + eq + 1); p.haveR = true; }
                else p.extra.push_back(w);
            }
            p.p = Vec3{ v[0], v[1], v[2] };
            n.pts.push_back(std::move(p));
        } else if (s.key == "curve") {
            if (s.val.block) {
                n.kids.push_back(nodeFromBlock(*s.val.block, m));
            } else {
                Node r;
                r.id = m.nextId++;
                r.ref = true;
                if (!s.val.words.empty()) r.name = s.val.words[0];
                n.kids.push_back(std::move(r));
            }
        } else {
            n.other.push_back(s);
        }
    }
    return n;
}

inline void groupFromBlock(const ftsl::Block& g, Model& m, FileModel& fm, Entry& e) {
    e.group = true;
    e.name = g.name;
    for (const ftsl::Stmt& s : g.stmts) {
        if (s.key == "curve" && s.val.block) {
            Entry c; c.id = m.nextId++; c.curve = nodeFromBlock(*s.val.block, m);
            e.items.push_back(std::move(c));
        } else if (s.key == "group" && s.val.block) {
            Entry sub; sub.id = m.nextId++;
            groupFromBlock(*s.val.block, m, fm, sub);
            e.items.push_back(std::move(sub));
        } else if (s.key == "fur" && s.val.block) {
            Entry f; f.id = m.nextId++; f.fur = true; f.furBlock = *s.val.block; f.furBlock.type = "fur"; f.name = f.furBlock.name;
            e.items.push_back(std::move(f));
        } else if (s.val.block) {
            fm.writable = false;
            if (fm.why.empty()) fm.why = "group \"" + g.name + "\" has a `" + s.key + "` block";
        } else {
            e.other.push_back(s);
        }
    }
}

// The file's opening run of comment lines (the header a generator wrote), minus the marker this
// writer adds, so a rewrite keeps the human's notes at the top.
inline std::string leadingComment(const std::string& path) {
    std::ifstream f(path);
    std::string line, out;
    while (std::getline(f, line)) {
        while (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t i = line.find_first_not_of(" \t");
        if (i == std::string::npos) { out += '\n'; continue; }
        if (line[i] != '#') break;
        if (line.find("rewritten by ftrace -groom") != std::string::npos) continue;
        out += line + '\n';
    }
    while (out.size() >= 2 && out[out.size() - 1] == '\n' && out[out.size() - 2] == '\n') out.pop_back();
    return out;
}

// Blocks (with their `file` tags) -> the model. Files with any block that is not a curve or a
// group of curves are recorded as not writable; their curves are still modelled, for viewing.
inline Model modelFromBlocks(const std::vector<ftsl::Block>& blocks) {
    Model m;
    for (const ftsl::Block& b : blocks) {
        FileModel* fm = m.fileFor(b.file);
        if (b.type == "curve") {
            Entry e; e.id = m.nextId++; e.curve = nodeFromBlock(b, m);
            fm->entries.push_back(std::move(e));
        } else if (b.type == "group") {
            Entry e; e.id = m.nextId++;
            groupFromBlock(b, m, *fm, e);
            fm->entries.push_back(std::move(e));
        } else if (b.type == "fur") {
            Entry e; e.id = m.nextId++; e.fur = true; e.furBlock = b; e.name = b.name;
            fm->entries.push_back(std::move(e));
        } else {
            fm->writable = false;
            if (fm->why.empty()) fm->why = "has a `" + b.type + "` block";
        }
    }
    for (FileModel& f : m.files) f.leading = leadingComment(f.path);
    return m;
}

// ---- queries ------------------------------------------------------------------------------
inline Node* findNodeIn(Node& n, int id) {
    if (n.id == id) return &n;
    for (Node& k : n.kids) if (Node* r = findNodeIn(k, id)) return r;
    return nullptr;
}
inline Node* findNodeIn(Entry& e, int id) {
    if (e.fur) return nullptr;
    if (!e.group) return findNodeIn(e.curve, id);
    for (Entry& it : e.items) if (Node* r = findNodeIn(it, id)) return r;
    return nullptr;
}
inline Node* findNode(Model& m, int id) {
    for (FileModel& f : m.files) for (Entry& e : f.entries) if (Node* r = findNodeIn(e, id)) return r;
    return nullptr;
}
inline Node* findDefIn(Node& n, const std::string& name) {
    if (!n.ref && !n.name.empty() && n.name == name) return &n;
    for (Node& k : n.kids) if (Node* r = findDefIn(k, name)) return r;
    return nullptr;
}
inline Node* findDefIn(Entry& e, const std::string& name) {
    if (e.fur) return nullptr;
    if (!e.group) return findDefIn(e.curve, name);
    for (Entry& it : e.items) if (Node* r = findDefIn(it, name)) return r;
    return nullptr;
}
// The definition a reference names: any named non-reference node (the loader registers every
// named node, nested or not).
inline Node* findDef(Model& m, const std::string& name) {
    for (FileModel& f : m.files) for (Entry& e : f.entries) if (Node* r = findDefIn(e, name)) return r;
    return nullptr;
}
// Where a node lives: its file, its top-level entry, and the group entry that contains that
// entry (null at file level).
struct Where { FileModel* file = nullptr; Entry* entry = nullptr; Entry* container = nullptr; };
inline bool whereIn(Entry& e, Entry* container, int id, Where& w) {
    if (e.fur) return false;
    if (!e.group) { if (findNodeIn(e.curve, id)) { w.entry = &e; w.container = container; return true; } return false; }
    for (Entry& it : e.items) if (whereIn(it, &e, id, w)) return true;
    return false;
}
inline Where whereIs(Model& m, int id) {
    Where w;
    for (FileModel& f : m.files)
        for (Entry& e : f.entries)
            if (whereIn(e, nullptr, id, w)) { w.file = &f; return w; }
    return w;
}
// 0 for a strand, 1 + the deepest child for a curve of curves; a reference has its target's level.
// ---- what a leaf ACTUALLY renders as -----------------------------------------------------
// The editor used to draw a leaf curve by joining its `point`s with straight segments -- its
// control POLYGON, which is the one shape the renderer never produces: `curveLeaf` stores the
// authored points verbatim and `addCurve` then runs them through `tessellateCurve` at the
// default basis, catmull_rom with 4 cones per span. A curve of curves, meanwhile, drew its path
// through `catmullRomAt` and so looked right, which is why the straightness read as a property
// of the LEVEL. It was a property of the drawing code.
//
// These two go through the loader's own tessellator rather than a second evaluator, so the
// preview cannot drift from the render: whatever `tessellateCurve` does, the pane shows.
struct DrawBasis {
    CurveBasis basis = CurveBasis::CatmullRom;
    int        subdiv = 4;
    double     alpha = 0.0;
};
// `basis`, `segments` and `spline` are taken from the OUTERMOST node (ftsl.h
// parseCurveNodeParams says so in as many words -- a strand is flattened once, at the top), so
// this is resolved at the top of a tree and handed down, not read per node.
inline DrawBasis drawBasisOf(const Node& top) {
    DrawBasis d;
    if (const ftsl::Stmt* bs = top.find("basis")) {
        const std::string v = bs->val.words.empty() ? std::string() : bs->val.words[0];
        if      (v == "linear")                                                 d.basis = CurveBasis::Linear;
        else if (v == "bezier")                                                 d.basis = CurveBasis::Bezier;
        else if (v == "bspline" || v == "b-spline")                             d.basis = CurveBasis::BSpline;
        else if (v == "catmull_rom" || v == "catmull-rom" || v == "catmullrom") d.basis = CurveBasis::CatmullRom;
    }
    if (const ftsl::Stmt* sg = top.find("segments")) {
        if (!sg->val.words.empty()) {
            const int k = std::atoi(sg->val.words[0].c_str());
            d.subdiv = (k < 1) ? 1 : (k > 256 ? 256 : k);
        }
    }
    d.alpha = top.alpha();
    return d;
}
// The polyline the renderer will produce for this leaf, in the node's own coordinates. Empty
// when the control points cannot form a valid curve in this basis (bezier wants 3k+1, bspline
// at least 4) -- the caller falls back to the control polygon and the point count is visibly
// the reason, which is a better diagnosis than a silently missing strand.
inline void leafPolyline(const Node& n, const DrawBasis& db, std::vector<Vec3>& out) {
    out.clear();
    const int np = (int)n.pts.size();
    if (np < 2) return;
    std::vector<Vec3> cp; cp.reserve((size_t)np);
    for (const Pt& p : n.pts) cp.push_back(p.p);
    if (curveSpanCount(db.basis, np) <= 0) return;
    std::vector<double> radii((size_t)np, 1e-4);      // positions do not depend on radius
    std::vector<CurveSeg> segs;
    const int ns = tessellateCurve(cp, radii, db.basis, db.subdiv, 0, 0, segs, db.alpha);
    if (ns <= 0) return;
    out.reserve((size_t)ns + 1);
    out.push_back(segs[0].p0);
    for (int k = 0; k < ns; ++k) out.push_back(segs[(size_t)k].p1);
}

// The same thing for an ALREADY-FLATTENED strand -- what `flattenCurveForTool` hands back for a
// placed node. Those are the blended CONTROL points, not geometry: the loader tessellates them at
// the top node's basis exactly like a leaf's, so drawing them straight understated the instances
// the same way, and on a placed node that is most of the strands in the groom.
inline void strandPolyline(const ftsl::CurveStrand& s, const DrawBasis& db, std::vector<Vec3>& out) {
    out.clear();
    const int np = (int)s.pts.size();
    if (np < 2) return;
    if (curveSpanCount(db.basis, np) <= 0) { out = s.pts; return; }
    std::vector<double> radii = s.radii;
    if ((int)radii.size() != np) radii.assign((size_t)np, 1e-4);
    std::vector<CurveSeg> segs;
    const int ns = tessellateCurve(s.pts, radii, db.basis, db.subdiv, 0, 0, segs, db.alpha);
    if (ns <= 0) { out = s.pts; return; }
    out.reserve((size_t)ns + 1);
    out.push_back(segs[0].p0);
    for (int k = 0; k < ns; ++k) out.push_back(segs[(size_t)k].p1);
}

inline int levelOf(Model& m, const Node& n, int depth = 0) {
    if (depth > 32) return 0;
    if (n.ref) { Node* d = findDef(m, n.name); return d ? levelOf(m, *d, depth + 1) : 0; }
    if (n.kids.empty()) return 0;
    int mx = 0;
    for (const Node& k : n.kids) mx = std::max(mx, levelOf(m, k, depth + 1));
    return mx + 1;
}
// The node's ROOT as placement sees it: its first strand's first point -- a strand's first
// point, or the first child's root, recursively. In the node's authored coordinates.
inline bool rootOf(Model& m, const Node& n, Vec3& out, int depth = 0) {
    if (depth > 32) return false;
    if (n.ref) { Node* d = findDef(m, n.name); return d && rootOf(m, *d, out, depth + 1); }
    if (!n.pts.empty()) { out = n.pts[0].p; return true; }
    if (!n.kids.empty()) return rootOf(m, n.kids[0], out, depth + 1);
    return false;
}
inline void collectNodes(Node& n, std::vector<Node*>& out) { out.push_back(&n); for (Node& k : n.kids) collectNodes(k, out); }
inline void collectNodes(Entry& e, std::vector<Node*>& out) {
    if (e.fur) return;
    if (!e.group) { collectNodes(e.curve, out); return; }
    for (Entry& it : e.items) collectNodes(it, out);
}
inline void collectNodes(Model& m, std::vector<Node*>& out) {
    for (FileModel& f : m.files) for (Entry& e : f.entries) collectNodes(e, out);
}
// A name no node has yet: `prefix_N`.
inline std::string freshName(Model& m, const std::string& prefix) {
    std::vector<Node*> all; collectNodes(m, all);
    for (int k = 1;; ++k) {
        const std::string cand = prefix + "_" + std::to_string(k);
        bool taken = false;
        for (const Node* n : all) if (n->name == cand) { taken = true; break; }
        if (!taken) return cand;
    }
}
// Remove a node: an entry-level curve from its list, an inline child from its parent.
inline bool eraseNodeIn(Node& parent, int id) {
    for (size_t i = 0; i < parent.kids.size(); ++i) {
        if (parent.kids[i].id == id) { parent.kids.erase(parent.kids.begin() + (std::ptrdiff_t)i); return true; }
        if (eraseNodeIn(parent.kids[i], id)) return true;
    }
    return false;
}
inline bool eraseNodeIn(std::vector<Entry>& list, int id) {
    for (size_t i = 0; i < list.size(); ++i) {
        Entry& e = list[i];
        if (e.fur) continue;
        if (e.group) { if (eraseNodeIn(e.items, id)) return true; continue; }
        if (e.curve.id == id) { list.erase(list.begin() + (std::ptrdiff_t)i); return true; }
        if (eraseNodeIn(e.curve, id)) return true;
    }
    return false;
}
inline bool eraseNode(Model& m, int id) {
    for (FileModel& f : m.files) if (eraseNodeIn(f.entries, id)) { f.dirty = true; return true; }
    return false;
}
// A new empty strand at the file's top level, or inside `container` when given.
inline Node& newCurve(Model& m, FileModel& fm, Entry* container, const std::string& name) {
    Entry e; e.id = m.nextId++; e.curve.id = m.nextId++; e.curve.name = name;
    std::vector<Entry>& list = container ? container->items : fm.entries;
    list.push_back(std::move(e));
    fm.dirty = true;
    return list.back().curve;
}

// ---- the model writer -----------------------------------------------------------------------
inline void writePt(std::string& out, const Pt& p, int indent) {
    out += std::string((size_t)indent * 4, ' ') + "point";
    if (!p.edited && !p.raw.empty()) {
        for (const std::string& w : p.raw) out += " " + w;
    } else {
        out += " " + fmtNum(p.p.x) + " " + fmtNum(p.p.y) + " " + fmtNum(p.p.z);
        if (p.haveR) out += " r=" + fmtNum(p.r);
        for (const std::string& w : p.extra) out += " " + w;
    }
    out += '\n';
}
inline bool writeNode(std::string& out, const Node& n, int indent, std::string& err) {
    const std::string pad((size_t)indent * 4, ' ');
    if (n.ref) { out += pad + "curve " + tok(n.name, true) + "\n"; return true; }
    if (n.pts.empty() && n.kids.empty()) return true;   // an empty strand: nothing to say, and the loader would refuse it
    out += pad + "curve";
    if (!n.name.empty()) out += " " + tok(n.name, true);
    out += " {\n";
    for (const ftsl::Stmt& s : n.other) if (!writeStmt(out, s, indent + 1, err)) return false;
    for (const Pt& p : n.pts) writePt(out, p, indent + 1);
    for (const Node& k : n.kids) if (!writeNode(out, k, indent + 1, err)) return false;
    out += pad + "}\n";
    return true;
}
inline bool writeEntry(std::string& out, const Entry& e, int indent, std::string& err) {
    if (e.fur) return writeBlock(out, e.furBlock, indent, err);
    if (!e.group) return writeNode(out, e.curve, indent, err);
    const std::string pad((size_t)indent * 4, ' ');
    out += pad + "group";
    if (!e.name.empty()) out += " " + tok(e.name, true);
    out += " {\n";
    for (const ftsl::Stmt& s : e.other) if (!writeStmt(out, s, indent + 1, err)) return false;
    for (const Entry& it : e.items) if (!writeEntry(out, it, indent + 1, err)) return false;
    out += pad + "}\n";
    return true;
}
inline const char* kMarker = "# rewritten by ftrace -groom -- the tool writes this file whole; comments inside blocks do not survive\n";
inline bool fileText(const FileModel& fm, std::string& out, std::string& err) {
    out = fm.leading;
    if (!out.empty() && out.back() != '\n') out += '\n';
    out += kMarker;
    out += '\n';
    for (const Entry& e : fm.entries) {
        if (!writeEntry(out, e, 0, err)) return false;
        if (e.group) out += '\n';
    }
    return true;
}
// Replace one `type "name" { ... }` block's text in a file, leaving everything else -- the
// comments around it included -- byte for byte. Braces inside strings and after `#` do not
// count; the block keeps the indentation of its header line.
inline bool patchBlockInFile(const std::string& path, const std::string& type, const std::string& name,
                             const std::string& text, std::string& err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    const std::string quoted = "\"" + name + "\"";
    size_t pos = 0, lineStart = std::string::npos, afterName = 0;
    while (pos < s.size()) {
        size_t le = s.find('\n', pos);
        if (le == std::string::npos) le = s.size();
        size_t i = pos;
        while (i < le && (s[i] == ' ' || s[i] == '\t')) ++i;
        if (s.compare(i, type.size(), type) == 0) {
            size_t j = i + type.size();
            if (j < le && (s[j] == ' ' || s[j] == '\t')) {
                while (j < le && (s[j] == ' ' || s[j] == '\t')) ++j;
                if (s.compare(j, quoted.size(), quoted) == 0) { lineStart = pos; afterName = j + quoted.size(); break; }
            }
        }
        pos = le + 1;
    }
    if (lineStart == std::string::npos) { err = type + " " + quoted + " not found in " + path; return false; }
    const size_t ob = s.find('{', afterName);
    if (ob == std::string::npos) { err = type + " " + quoted + " in " + path + " has no `{`"; return false; }
    int depth = 0;
    bool inStr = false;
    size_t end = std::string::npos;
    for (size_t k = ob; k < s.size(); ++k) {
        const char c = s[k];
        if (inStr) { if (c == '\\') { ++k; continue; } if (c == '"') inStr = false; continue; }
        if (c == '"') { inStr = true; continue; }
        if (c == '#') { while (k < s.size() && s[k] != '\n') ++k; continue; }
        if (c == '{') ++depth;
        else if (c == '}') { if (--depth == 0) { end = k + 1; break; } }
    }
    if (end == std::string::npos) { err = type + " " + quoted + " in " + path + ": unbalanced braces"; return false; }
    std::string indent;
    for (size_t k = lineStart; k < s.size() && (s[k] == ' ' || s[k] == '\t'); ++k) indent += s[k];
    std::string t = text;
    while (!t.empty() && (t.back() == '\n' || t.back() == '\r')) t.pop_back();
    std::string body;
    {
        std::istringstream ls(t);
        std::string line;
        bool first = true;
        while (std::getline(ls, line)) { if (!first) body += "\n" + indent; body += line; first = false; }
    }
    s = s.substr(0, lineStart + indent.size()) + body + s.substr(end);
    std::ofstream o(path, std::ios::binary);
    if (!o) { err = "cannot write " + path; return false; }
    o << s;
    return true;
}
inline void clearFurDirty(std::vector<Entry>& list) {
    for (Entry& e : list) { if (e.fur) e.furDirty = false; else if (e.group) clearFurDirty(e.items); }
}
inline bool patchDirtyFur(FileModel& fm, std::vector<Entry>& list, bool& any, std::string& err) {
    for (Entry& e : list) {
        if (e.group) { if (!patchDirtyFur(fm, e.items, any, err)) return false; continue; }
        if (!e.fur || !e.furDirty) continue;
        if (e.name.empty()) { err = fm.path + ": an unnamed `fur` block cannot be patched in place -- name it"; return false; }
        std::string text;
        if (!writeBlock(text, e.furBlock, 0, err)) return false;
        if (!patchBlockInFile(fm.path, "fur", e.name, text, err)) return false;
        e.furDirty = false;
        any = true;
    }
    return true;
}
// Every dirty file, whole; every edited fur block in a file kept as is, patched into its own
// span. Nothing is written unless every file with curve edits is writable.
inline bool saveModel(Model& m, std::string& err, std::vector<std::string>* written = nullptr) {
    for (const FileModel& fm : m.files)
        if (fm.dirty && !fm.writable) {
            err = fm.path + ": not writable by the groom tool (" + fm.why + ") -- keep hair in its own included file";
            return false;
        }
    for (FileModel& fm : m.files) {
        if (fm.dirty) {
            std::string text;
            if (!fileText(fm, text, err)) { err = fm.path + ": " + err; return false; }
            std::ofstream f(fm.path, std::ios::binary);
            if (!f) { err = "cannot write " + fm.path; return false; }
            f << text;
            fm.dirty = false;
            clearFurDirty(fm.entries);
            if (written) written->push_back(fm.path);
        } else {
            bool any = false;
            if (!patchDirtyFur(fm, fm.entries, any, err)) return false;
            if (any && written) written->push_back(fm.path + " (fur block patched)");
        }
    }
    return true;
}

// ---- editing a node's statements (Phase 3) ---------------------------------------------------
inline void setStmt(Node& n, const std::string& key, const std::vector<std::string>& words) {
    for (ftsl::Stmt& s : n.other) if (s.key == key) { s.val.words = words; return; }
    ftsl::Stmt s; s.key = key; s.val.words = words;
    n.other.push_back(std::move(s));
}
inline void removeStmts(Node& n, const std::string& key) {
    n.other.erase(std::remove_if(n.other.begin(), n.other.end(), [&](const ftsl::Stmt& s) { return s.key == key; }), n.other.end());
}
inline std::vector<std::pair<double, double>> densityKeys(const Node& n) {   // (t, rho), as written
    std::vector<std::pair<double, double>> out;
    for (const ftsl::Stmt& s : n.other)
        if (s.key == "density_at" && s.val.words.size() >= 2)
            out.push_back({ std::atof(s.val.words[0].c_str()), std::atof(s.val.words[1].c_str()) });
    return out;
}
inline void setDensityKeys(Node& n, const std::vector<std::pair<double, double>>& keys) {
    removeStmts(n, "density_at");
    for (const auto& k : keys) {
        ftsl::Stmt s; s.key = "density_at"; s.val.words = { fmtNum(k.first), fmtNum(k.second) };
        n.other.push_back(std::move(s));
    }
}
inline double constDensity(const Node& n) {
    const ftsl::Stmt* d = n.find("density");
    return (d && !d->val.words.empty()) ? std::atof(d->val.words[0].c_str()) : 0.0;
}
inline std::string splineWord(const Node& n) {
    const ftsl::Stmt* s = n.find("spline");
    return (s && !s->val.words.empty()) ? s->val.words[0] : std::string("uniform");
}
// A new curve of curves referencing `childNames` in order, appended to the file (or group) so
// every child is defined above it.
inline Node& newGroupOf(Model& m, FileModel& fm, Entry* container, const std::string& name, const std::vector<std::string>& childNames) {
    Node& n = newCurve(m, fm, container, name);
    for (const std::string& c : childNames) { Node r; r.id = m.nextId++; r.ref = true; r.name = c; n.kids.push_back(std::move(r)); }
    return n;
}
// Move the entry holding node `id` to the end of its list (so the names it references, all
// above it, resolve at load).
inline bool moveEntryToEnd(std::vector<Entry>& list, int id) {
    for (size_t i = 0; i < list.size(); ++i) {
        Entry& e = list[i];
        if (e.fur) continue;
        if (e.group) { if (moveEntryToEnd(e.items, id)) return true; continue; }
        if (e.curve.id == id) { std::rotate(list.begin() + (std::ptrdiff_t)i, list.begin() + (std::ptrdiff_t)i + 1, list.end()); return true; }
    }
    return false;
}
inline bool moveEntryToEnd(Model& m, int id) {
    for (FileModel& f : m.files) if (moveEntryToEnd(f.entries, id)) { f.dirty = true; return true; }
    return false;
}
// The index of the entry holding node `id` within its list, or -1 (nested nodes have none).
inline int entryIndexOf(const std::vector<Entry>& list, int id) {
    for (size_t i = 0; i < list.size(); ++i) if (!list[i].group && !list[i].fur && list[i].curve.id == id) return (int)i;
    return -1;
}
// Every fur entry, with its file.
template <class F> inline void forEachFurIn(FileModel& fm, std::vector<Entry>& list, F& f) {
    for (Entry& e : list) { if (e.fur) f(fm, e); else if (e.group) forEachFurIn(fm, e.items, f); }
}
template <class F> inline void forEachFur(Model& m, F f) { for (FileModel& fm : m.files) forEachFurIn(fm, fm.entries, f); }

// ---- the model -> blocks, and the live preview through the loader's own recursion ---------------
inline ftsl::Stmt stmtFromPt(const Pt& p) {
    ftsl::Stmt s; s.key = "point";
    if (!p.edited && !p.raw.empty()) s.val.words = p.raw;
    else {
        s.val.words = { fmtNum(p.p.x), fmtNum(p.p.y), fmtNum(p.p.z) };
        if (p.haveR) s.val.words.push_back("r=" + fmtNum(p.r));
        for (const std::string& w : p.extra) s.val.words.push_back(w);
    }
    return s;
}
inline ftsl::Block blockFromNode(const Node& n) {
    ftsl::Block b; b.type = "curve"; b.name = n.name;
    for (const ftsl::Stmt& s : n.other) b.stmts.push_back(s);
    for (const Pt& p : n.pts) b.stmts.push_back(stmtFromPt(p));
    for (const Node& k : n.kids) {
        ftsl::Stmt s; s.key = "curve";
        if (k.ref) s.val.words = { k.name };
        else s.val.block = std::make_shared<ftsl::Block>(blockFromNode(k));
        b.stmts.push_back(std::move(s));
    }
    return b;
}
template <class F> inline void forEachEntryCurveC(const std::vector<Entry>& list, F& f) {
    for (const Entry& e : list) { if (e.fur) continue; if (e.group) forEachEntryCurveC(e.items, f); else f(e.curve); }
}
// Every top-level curve of the model flattened through ftsl::Builder::flattenCurveForTool, in
// model order (so a reference finds its definition), keyed by node name: what each named node
// makes (its strands, its child roots) in the file's own coordinates. `err` names the first node
// the loader would refuse.
struct Preview { std::map<std::string, ftsl::Loaded::HairCurveInfo> byName; std::string err; };
inline void buildPreview(const Model& m, Preview& out) {
    out.byName.clear();
    out.err.clear();
    ftsl::Builder bld;
    auto one = [&](const Node& n) {
        if (n.pts.empty() && n.kids.empty()) return;            // an empty strand: not written either
        const ftsl::Block b = blockFromNode(n);
        std::vector<ftsl::Loaded::HairCurveInfo> recs;
        std::vector<ftsl::CurveStrand> strands;
        std::string why;
        if (!bld.flattenCurveForTool(b, recs, strands, why)) {
            if (out.err.empty()) out.err = (n.name.empty() ? std::string("(unnamed curve)") : n.name) + ": " + why;
            return;
        }
        for (auto& r : recs) out.byName[r.name] = std::move(r);
    };
    for (const FileModel& f : m.files) forEachEntryCurveC(f.entries, one);
}

// The scene's curve blocks as written: parse the file, tag, expand its includes (each block
// keeps the file it came from), and model them.
inline bool parseSceneModel(const std::string& scenePath, Model& m, std::string& err) {
    std::ifstream f(scenePath);
    if (!f) { err = "cannot open " + scenePath; return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::vector<ftsl::Block> blocks;
    if (!ftsl_gpda::parse(ss.str(), blocks, err)) return false;
    for (ftsl::Block& b : blocks) b.file = scenePath;
    std::string dir;
    { const std::filesystem::path p(scenePath); if (p.has_parent_path()) dir = p.parent_path().string(); }
    assetbytes::ScopedSceneDir sd(dir);
    std::vector<std::string> chain;
    if (!ftsl::expandIncludes(blocks, scenePath, chain, err)) return false;
    m = modelFromBlocks(blocks);
    return true;
}

// `ftrace -in <scene> -groom-check`: the tool's preview of every named curve against what the
// loader just built for the same scene (its records, world space, brought back to the file's
// frame through the inverse of the transform it applied). The two run the same recursion on
// the same tokens, so they must agree to rounding.
inline bool checkPreviewAgainstLoaded(const ftsl::Loaded& L, const std::string& scenePath, std::string& report) {
    Model m;
    std::string err;
    if (!parseSceneModel(scenePath, m, err)) { report = "model: " + err; return false; }
    Preview pv;
    buildPreview(m, pv);
    if (!pv.err.empty()) { report = "preview: " + pv.err; return false; }
    int checked = 0, missing = 0, bad = 0;
    double worst = 0.0;
    for (const auto& rec : L.hairCurves) {
        auto it = pv.byName.find(rec.name);
        if (it == pv.byName.end()) { ++missing; continue; }
        const Affine inv = rec.xf.inverse();
        const auto& a = it->second.strands;
        const auto& b = rec.strands;
        bool ok = (a.size() == b.size());
        for (size_t i = 0; ok && i < a.size(); ++i) {
            ok = (a[i].pts.size() == b[i].pts.size());
            for (size_t k = 0; ok && k < a[i].pts.size(); ++k) {
                const Vec3 q = inv.apply(b[i].pts[k]);
                const double d = length(q - a[i].pts[k]);
                worst = std::max(worst, d);
                if (d > 1e-7 * (1.0 + length(a[i].pts[k]))) ok = false;
            }
        }
        ++checked;
        if (!ok) ++bad;
    }
    char buf[256];
    std::snprintf(buf, sizeof buf, "%d named curve(s) compared, %d differ, %d not in the model, worst point distance %.3g m", checked, bad, missing, worst);
    report = buf;
    return bad == 0 && missing == 0 && checked > 0;
}

// ---- the headless round trip: `ftrace -groom-rewrite <in> <out>` -----------------------------
// Parses ONE file (includes are kept as `include` lines), runs its curve and group blocks through
// the model and everything else through the general writer, and writes the result. A scene
// rewritten this way must `-dumpcurves` byte-identically to the original (tools/groom_rig.py).
inline bool rewriteFile(const std::string& in, const std::string& outPath, std::string& err) {
    std::ifstream f(in);
    if (!f) { err = "cannot open " + in; return false; }
    std::stringstream ss; ss << f.rdbuf();
    std::vector<ftsl::Block> blocks;
    if (!ftsl_gpda::parse(ss.str(), blocks, err)) return false;
    for (ftsl::Block& b : blocks) b.file = in;
    Model m = modelFromBlocks(blocks);
    std::string out = leadingComment(in);
    if (!out.empty()) out += '\n';
    size_t ei = 0;
    int curves = 0, others = 0;
    for (const ftsl::Block& b : blocks) {
        if (b.type == "curve" || b.type == "group" || b.type == "fur") {
            if (m.files.empty() || ei >= m.files[0].entries.size()) { err = "internal: entry/block mismatch"; return false; }
            if (!writeEntry(out, m.files[0].entries[ei++], 0, err)) return false;
            ++curves;
        } else {
            if (!writeBlock(out, b, 0, err)) return false;
            ++others;
        }
    }
    std::ofstream o(outPath, std::ios::binary);
    if (!o) { err = "cannot write " + outPath; return false; }
    o << out;
    std::fprintf(stderr, "[groom] rewrote %s -> %s: %d curve/group block(s) through the model, %d other block(s)\n",
                 in.c_str(), outPath.c_str(), curves, others);
    return true;
}

}  // namespace groom
