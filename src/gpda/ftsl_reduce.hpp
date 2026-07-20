// reduce.hpp — GPDA ParseNode tree  ->  ftrace ftsl::Block tree, + a structural
// differ.  This is the heart of the J3c validation shim: it turns the shared
// grammar's parse tree into the exact std::vector<ftsl::Block> shape ftrace's
// hand-written parseTop() produces, so the two can be diffed block-for-block.
//
// The mapping faithfully mirrors ftrace's Parser (src/ftsl.h):
//   * value continuation / record-override `= rhs [i]` / `[i]` selector folding
//   * nested-block type/name derivation (bareword => type, single quoted => name)
//   * brace-body flat `words` dump (key, then post-pop value words)
//   * record `range` stmt + one stmt per channel line; prefer/else branches
// STRING tokens carry their quotes in the grammar but ftrace's tokenizer strips
// them, so every string word/name is unquoted here to match.
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tokenized.hpp"
// ftsl::Block / Stmt / Value come from ftrace's front-end.  In-tree this header
// is included by ftsl.h *after* those types are defined; the standalone de-risk
// harness (scraps/gpda_shim) defines FTSL_SHIM_STANDALONE to pull a copied slice.
#ifdef FTSL_SHIM_STANDALONE
#include "ftrace_parse_slice.hpp"
#endif

namespace shim {

using PN = gpda_tok::ParseNode;

// ---- ParseNode navigation helpers ----------------------------------------

inline const PN* child(const PN* n, const char* name) {
    for (const auto& c : n->children)
        if (c->name == name) return c.get();
    return nullptr;
}

inline std::vector<const PN*> children(const PN* n, const char* name) {
    std::vector<const PN*> out;
    for (const auto& c : n->children)
        if (c->name == name) out.push_back(c.get());
    return out;
}

// The single terminal leaf under a wrapper rule (val_head / cont / rhs / ...).
inline const PN* leaf_of(const PN* wrapper) {
    return wrapper->children.empty() ? wrapper : wrapper->children[0].get();
}

// ftrace's tokenizer strips the quotes off strings; the grammar keeps them.
inline std::string unquote(const std::string& type, const std::string& value) {
    if (type == "STRING" && value.size() >= 2)
        return value.substr(1, value.size() - 2);
    return value;
}

// Concatenate a selector's sel_word leaves with no separator (ftrace builds the
// index string by appending each Word token's text).
inline std::string selector_index(const PN* selector) {
    std::string idx;
    for (const PN* sw : children(selector, "sel_word")) {
        const PN* t = leaf_of(sw);
        idx += t->value;
    }
    return idx;
}

// ---- forward decls --------------------------------------------------------

void reduce_brace_body(const PN* brace_body, ftsl::Block& b);
ftsl::Block reduce_top_block(const PN* top_block);

// ---- value ----------------------------------------------------------------

inline ftsl::Value reduce_value(const PN* value_node, const std::string& key) {
    ftsl::Value v;
    if (value_node->children.empty()) return v;
    const PN* c = value_node->children[0].get();   // override_val | normal_val

    if (c->name == "override_val") {                // '=' rhs? selector?
        v.words.push_back("=");
        const PN* rhs = child(c, "rhs");
        const PN* sel = child(c, "selector");
        if (rhs) {
            const PN* t = leaf_of(rhs);
            std::string r = unquote(t->name, t->value);
            if (sel) r += "[" + selector_index(sel) + "]";
            v.words.push_back(r);
        } else if (sel && !v.words.empty()) {
            v.words.back() += "[" + selector_index(sel) + "]";
        }
        return v;
    }

    // normal_val = val_head? cont* selector? block?
    const PN* vh = child(c, "val_head");
    bool firstWasString = false;
    if (vh) {
        const PN* t = leaf_of(vh);
        firstWasString = (t->name == "STRING");
        v.words.push_back(unquote(t->name, t->value));
    }
    for (const PN* cont : children(c, "cont")) {
        const PN* t = leaf_of(cont);
        v.words.push_back(unquote(t->name, t->value));
    }
    const PN* sel = child(c, "selector");
    if (sel && !v.words.empty() &&
            v.words.back().find('.') != std::string::npos) {
        v.words.back() += "[" + selector_index(sel) + "]";
    }
    const PN* blk = child(c, "block");
    if (blk) {
        std::string btype = key, bname;
        if (!v.words.empty()) {
            if (v.words.size() == 1 && firstWasString) {
                bname = v.words.back(); v.words.pop_back();
            } else {
                btype = v.words.back(); v.words.pop_back();
            }
        }
        v.block = std::make_shared<ftsl::Block>();
        v.block->type = btype;
        v.block->name = bname;
        reduce_brace_body(blk, *v.block);
    }
    return v;
}

// ---- brace body -----------------------------------------------------------

inline void reduce_brace_body(const PN* brace_body, ftsl::Block& b) {
    for (const PN* bi : children(brace_body, "body_item")) {
        const PN* st = child(bi, "stmt");
        if (!st) continue;
        const PN* kt = child(st, "key_tok");
        std::string key = leaf_of(kt)->value;      // key_tok is never a STRING
        ftsl::Stmt s;
        s.key = key;
        b.words.push_back(key);
        const PN* vnode = child(st, "value");
        s.val = reduce_value(vnode, key);
        for (const auto& w : s.val.words) b.words.push_back(w);
        b.stmts.push_back(std::move(s));
    }
}

// ---- top-level block ------------------------------------------------------

inline ftsl::Block reduce_top_block(const PN* top_block) {
    const PN* alt = top_block->children[0].get();
    const std::string& k = alt->name;
    ftsl::Block b;

    if (k == "plain_header") {                      // WORD STRING? subtype? brace_body
        b.type = child(alt, "WORD")->value;
        if (const PN* s = child(alt, "STRING")) b.name = unquote("STRING", s->value);
        if (const PN* st = child(alt, "subtype")) b.subtype = leaf_of(st)->value;
        reduce_brace_body(child(alt, "brace_body"), b);
    } else if (k == "assign_header") {              // WORD '=' WORD subtype? brace_body
        auto ws = children(alt, "WORD");            // [name, '=', KIND]
        b.name = ws[0]->value;
        b.type = ws.size() > 2 ? ws[2]->value : "";
        if (const PN* st = child(alt, "subtype")) b.subtype = leaf_of(st)->value;
        reduce_brace_body(child(alt, "brace_body"), b);
    } else if (k == "spectrum_decl") {              // 'spectrum' STRING? subtype? '=' value
        b.type = "spectrum";
        if (const PN* s = child(alt, "STRING")) b.name = unquote("STRING", s->value);
        if (const PN* st = child(alt, "subtype")) b.subtype = leaf_of(st)->value;
        ftsl::Stmt s;
        s.key = "=";
        s.val = reduce_value(child(alt, "value"), "=");
        b.stmts.push_back(std::move(s));
    } else if (k == "record_decl") {                // WORD '=' 'range' range_word* record_body
        b.type = "record";
        b.name = children(alt, "WORD")[0]->value;   // first WORD is the binding NAME
        ftsl::Stmt dom;
        dom.key = "range";
        for (const PN* rw : children(alt, "range_word")) {
            const PN* t = leaf_of(rw);
            dom.val.words.push_back(unquote(t->name, t->value));
        }
        b.stmts.push_back(std::move(dom));
        const PN* rb = child(alt, "record_body");   // '[' rec_item* ']'
        for (const PN* ri : children(rb, "rec_item")) {
            const PN* rl = child(ri, "record_line"); // WORD stop_word*
            if (!rl) continue;
            ftsl::Stmt s;
            s.key = child(rl, "WORD")->value;
            for (const PN* sw : children(rl, "stop_word")) {
                const PN* t = leaf_of(sw);
                s.val.words.push_back(unquote(t->name, t->value));
            }
            b.stmts.push_back(std::move(s));
        }
    } else if (k == "prefer_block") {               // 'prefer' block_list ('else' block_list)*
        b.type = "prefer";
        for (const PN* bl : children(alt, "block_list")) {  // '{' item* '}'
            std::vector<ftsl::Block> branch;
            for (const PN* it : children(bl, "item")) {
                const PN* tb = child(it, "top_block");
                if (tb) branch.push_back(reduce_top_block(tb));
            }
            b.branches.push_back(std::move(branch));
        }
    }
    return b;
}

// A whole scene_file -> vector<Block>.
inline std::vector<ftsl::Block> reduce_scene(const PN* scene_file) {
    std::vector<ftsl::Block> blocks;
    for (const PN* it : children(scene_file, "item")) {
        const PN* tb = child(it, "top_block");
        if (tb) blocks.push_back(reduce_top_block(tb));
    }
    return blocks;
}

// ---- structural diff ------------------------------------------------------

struct Diff {
    std::vector<std::string> msgs;
    bool ok() const { return msgs.empty(); }
    void add(const std::string& where, const std::string& m) {
        msgs.push_back(where + ": " + m);
    }
};

inline void diff_value(const ftsl::Value& a, const ftsl::Value& b,
                       const std::string& where, Diff& d);

inline void diff_block(const ftsl::Block& a, const ftsl::Block& b,
                       const std::string& where, Diff& d) {
    if (a.type != b.type) d.add(where, "type '" + a.type + "' != '" + b.type + "'");
    if (a.subtype != b.subtype) d.add(where, "subtype '" + a.subtype + "' != '" + b.subtype + "'");
    if (a.name != b.name) d.add(where, "name '" + a.name + "' != '" + b.name + "'");
    if (a.words != b.words) {
        std::string as, bs;
        for (auto& w : a.words) as += "|" + w;
        for (auto& w : b.words) bs += "|" + w;
        d.add(where, "words [" + as + " ] != [" + bs + " ]");
    }
    if (a.stmts.size() != b.stmts.size()) {
        d.add(where, "stmt count " + std::to_string(a.stmts.size()) +
              " != " + std::to_string(b.stmts.size()));
    } else {
        for (size_t i = 0; i < a.stmts.size(); ++i) {
            const auto& sa = a.stmts[i];
            const auto& sb = b.stmts[i];
            std::string w = where + ".stmt[" + std::to_string(i) + "]";
            if (sa.key != sb.key) d.add(w, "key '" + sa.key + "' != '" + sb.key + "'");
            diff_value(sa.val, sb.val, w, d);
        }
    }
    if (a.branches.size() != b.branches.size()) {
        d.add(where, "branch count " + std::to_string(a.branches.size()) +
              " != " + std::to_string(b.branches.size()));
    } else {
        for (size_t i = 0; i < a.branches.size(); ++i) {
            const auto& ba = a.branches[i];
            const auto& bb = b.branches[i];
            std::string w = where + ".branch[" + std::to_string(i) + "]";
            if (ba.size() != bb.size()) {
                d.add(w, "block count " + std::to_string(ba.size()) +
                      " != " + std::to_string(bb.size()));
            } else {
                for (size_t j = 0; j < ba.size(); ++j)
                    diff_block(ba[j], bb[j], w + "[" + std::to_string(j) + "]", d);
            }
        }
    }
}

inline void diff_value(const ftsl::Value& a, const ftsl::Value& b,
                       const std::string& where, Diff& d) {
    if (a.words != b.words) {
        std::string as, bs;
        for (auto& w : a.words) as += "|" + w;
        for (auto& w : b.words) bs += "|" + w;
        d.add(where, "val.words [" + as + " ] != [" + bs + " ]");
    }
    if (static_cast<bool>(a.block) != static_cast<bool>(b.block)) {
        d.add(where, "one has a nested block, the other doesn't");
    } else if (a.block && b.block) {
        diff_block(*a.block, *b.block, where + ".block", d);
    }
}

inline Diff diff_scene(const std::vector<ftsl::Block>& a,
                       const std::vector<ftsl::Block>& b) {
    Diff d;
    if (a.size() != b.size()) {
        d.add("scene", "top-block count " + std::to_string(a.size()) +
              " != " + std::to_string(b.size()));
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
            diff_block(a[i], b[i], "block[" + std::to_string(i) + "]", d);
        return d;
    }
    for (size_t i = 0; i < a.size(); ++i)
        diff_block(a[i], b[i], "block[" + std::to_string(i) + "]", d);
    return d;
}

}  // namespace shim
