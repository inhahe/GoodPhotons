// ftsl_reduce.hpp — GPDA ParseNode tree  ->  ftrace ftsl::Block tree.  This is the
// back half of ftrace's .ftsl front end (see ftsl_frontend.hpp): it turns the shared
// grammar's parse tree into the exact std::vector<ftsl::Block> shape the rest of
// ftrace consumes.
//
// That shape was reverse-engineered from the hand-written recursive-descent parser
// this front end replaced (deleted in 0.79.0), so the conventions below are its
// conventions and the loader still depends on every one of them:
//   * value continuation / record-override `= rhs [i]` / `[i]` selector folding
//   * nested-block type/name derivation (bareword => type, single quoted => name)
//   * brace-body flat `words` dump (key, then post-pop value words)
//   * record `range` stmt + one stmt per channel line; prefer/else branches
// STRING tokens carry their quotes in the grammar but the old tokenizer stripped
// them, so every string word/name is unquoted here to match.
//
// (A structural differ lived here too, driving the 0.68 corpus comparison to
// MATCH 2595/2595; it went with `-validate-grammar` in 0.79.0 — with one front end
// there is nothing left to diff against.)
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "tokenized.hpp"
// ftsl::Block / Stmt / Value come from ftrace's front-end.  In-tree this header
// is included by ftsl.h *after* those types are defined; the standalone de-risk
// harness (scraps/gpda_shim) defines FTSL_GPDA_STANDALONE to pull a copied slice.
#ifdef FTSL_GPDA_STANDALONE
#include "ftrace_parse_slice.hpp"
#endif

namespace ftsl_gpda {

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

// Collect a `[ … ]` group's raw item tree (bare words, and nested groups for an N-D
// array literal's inner axes) WITHOUT interpreting it — ftsl::applyBracketGroup owns
// that decision for both front ends. `node` is a `selector` or a `sub_array`.
inline std::vector<ftsl::BrItem> bracket_items(const PN* node) {
    std::vector<ftsl::BrItem> out;
    for (const auto& c : node->children) {
        if (c->name == "sel_item") {
            for (const auto& g : c->children) {
                if (g->name == "sub_array") {
                    ftsl::BrItem it; it.isGroup = true; it.items = bracket_items(g.get());
                    out.push_back(std::move(it));
                } else if (g->name == "sel_word") {
                    ftsl::BrItem it; it.word = leaf_of(g.get())->value;
                    out.push_back(std::move(it));
                }
            }
        }
    }
    return out;
}

// Flatten a record channel line's stop tokens into the flat `Value::words` list the
// loader reads, turning each `stop_group` back into a `[` … `]` pair of MARKER words.
//
// Records are the one place the language keeps its structure in the *word stream*
// rather than in a side-channel like `BrItem` — a channel line's shape is defined by
// the generalized stop grammar's delimiter ladder, whose other two rungs (comma and
// whitespace) already survive lexing as ordinary word text.  Re-serialising brackets
// into the same stream keeps all three rungs in one place, so `recladder::parse` sees
// the author's delimiters exactly as written instead of half a tree plus half a list.
// The markers can't be confused with content: `[` and `]` are excluded from the
// character class of every terminal, so no real word is ever spelled either way.
inline void flatten_stop_words(const PN* node, std::vector<std::string>& out) {
    for (const auto& c : node->children) {
        if (c->name == "stop_item") {
            flatten_stop_words(c.get(), out);
        } else if (c->name == "stop_word") {
            const PN* g = c->children.empty() ? nullptr : c->children[0].get();
            if (g && g->name == "stop_group") {
                out.push_back("[");
                flatten_stop_words(g, out);
                out.push_back("]");
            } else {
                const PN* t = leaf_of(c.get());
                out.push_back(unquote(t->name, t->value));
            }
        }
    }
}

// The trailing sample call on a selector — `(u)`, `(u,v)` — or "" when absent.
inline std::string selector_call(const PN* selector) {
    const PN* at = child(selector, "axistuple");
    return at ? leaf_of(at)->value : std::string();
}

// Hand the group to the shared decision function (see ftsl::applyBracketGroup).
inline void apply_selector(ftsl::Value& v, const PN* sel, bool overrideForm) {
    ftsl::applyBracketGroup(v, bracket_items(sel), selector_call(sel),
                            (int)sel->first_pos().first, overrideForm);
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
            v.words.push_back(unquote(t->name, t->value));
        }
        if (sel) apply_selector(v, sel, /*overrideForm=*/true);
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
    if (sel) apply_selector(v, sel, /*overrideForm=*/false);
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
        const PN* kleaf = leaf_of(kt);
        std::string key = kleaf->value;            // key_tok is never a STRING
        ftsl::Stmt s;
        s.key = key;
        s.line = (int)kleaf->line;                 // ftrace stamps the key's line
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
        const PN* sv = child(alt, "value");
        s.line = (int)sv->first_pos().first;        // ftrace stamps the value's line
        s.val = reduce_value(sv, "=");
        b.stmts.push_back(std::move(s));
    } else if (k == "record_decl") {                // WORD '=' 'range' range_word* record_body
        b.type = "record";
        b.name = children(alt, "WORD")[0]->value;   // first WORD is the binding NAME
        ftsl::Stmt dom;
        dom.key = "range";
        {   // ftrace stamps the line of the first token AFTER `range`
            auto rws = children(alt, "range_word");
            dom.line = rws.empty() ? 0 : (int)leaf_of(rws[0])->line;
        }
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
            const PN* chan = child(rl, "WORD");
            s.key = chan->value;
            s.line = (int)chan->line;
            flatten_stop_words(rl, s.val.words);
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

}  // namespace ftsl_gpda
