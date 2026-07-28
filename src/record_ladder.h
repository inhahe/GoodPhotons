// record_ladder.h — the delimiter **precedence ladder** for FTSL record stops.
//
// A record channel line is a nested value written with three delimiters that form a
// fixed precedence ladder — not an interchangeable free-order set:
//
//   * **whitespace binds tightest — like `*`** (juxtaposition builds a vector),
//   * **comma binds looser — like `+`**       (opens a new outer level),
//   * **brackets `[ ]` are the parentheses**  (an explicit level, anywhere).
//
// so `1 1 1, 2 2 2` reads exactly like `(1*1*1) + (2*2*2)` — two groups of three.
// Because the precedence is fixed, **structure is recoverable from the delimiters
// alone**; a channel's declared arity only *validates* what the delimiters already
// said. All of these therefore denote the same tree:
//
//     1 1 1                  ==  [1 1 1]                    (bracketing once is idempotent)
//     1 1 1, 2 2 2, 3 3 3    ==  [1 1 1] [2 2 2] [3 3 3]    (sum-of-products == product-of-groups)
//
// Parens `( )` are deliberately NOT a rung: they are reserved for expression grouping
// and the named-input application surface (`sin(v)`, `gold.color(u=x)`). The tokenizer
// therefore treats a parenthesised run as an **opaque atom**, so `clamp(x,0,1)` stays a
// single leaf and its inner commas are not delimiters. That is also why the ladder can
// live entirely in the loader instead of the grammar: `,` is not one of ftrace's
// tokenizer delimiters, so a comma survives lexing glued to its word (`0,`) and gets
// re-split here, paren-aware. Only `[` / `]` genuinely delimit, and the front end hands
// those through as marker words (see ftsl_reduce.hpp's `flatten_stop_words`).
//
// This is the C++ twin of loom's `loom/ladder.py`; the two must agree, since loom emits
// what ftrace reads. Kept dependency-free (string + vector) so it can be unit-tested and
// included anywhere.
#pragma once
#include <string>
#include <vector>

namespace recladder {

// A parsed ladder value: either a LEAF (one raw token — a number, a `spectrum:` ref, or
// an expression) or a GROUP (an ordered list of values). Single-element groups unwrap
// during parsing, which is what makes bracketing idempotent.
struct Value {
    bool               isLeaf = true;
    std::string        leaf;      // valid iff isLeaf
    std::vector<Value> items;     // valid iff !isLeaf

    int  depth() const {          // 0 for a leaf, 1 + max child depth for a group
        if (isLeaf) return 0;
        int d = 0;
        for (const auto& c : items) { int cd = c.depth(); if (cd > d) d = cd; }
        return d + 1;
    }
};

// ---- tokenizer ------------------------------------------------------------
// The front end has already lexed the line into words, with `[` / `]` isolated as their
// own marker words. All that is left is to split top-level commas out of the remaining
// words: `0,` -> `0` `,`, `0,1` -> `0` `,` `1`, while `clamp(x,0,1)` stays whole because
// a comma inside parens is at depth > 0.
inline void tokenize(const std::vector<std::string>& words,
                     std::vector<std::string>& out) {
    for (const std::string& w : words) {
        if (w == "[" || w == "]" || w == ",") { out.push_back(w); continue; }
        std::string buf;
        int depth = 0;
        for (char c : w) {
            if (c == '(') { ++depth; buf.push_back(c); }
            else if (c == ')') { if (depth > 0) --depth; buf.push_back(c); }
            else if (c == ',' && depth == 0) {
                if (!buf.empty()) { out.push_back(buf); buf.clear(); }
                out.push_back(",");
            } else buf.push_back(c);
        }
        if (!buf.empty()) out.push_back(buf);
    }
}

// True if the token stream uses any ladder delimiter, i.e. whether the author opted
// into the generalized form at all. A line with none of them is a plain whitespace list
// and callers can keep their original, simpler reading of it.
inline bool usesLadder(const std::vector<std::string>& toks) {
    for (const std::string& t : toks)
        if (t == "," || t == "[" || t == "]") return true;
    return false;
}

// ---- recursive-descent parser (sum -> product -> factor) ------------------

namespace detail {

struct Cursor {
    const std::vector<std::string>* toks;
    size_t i = 0;
    const std::string* peek() const { return i < toks->size() ? &(*toks)[i] : nullptr; }
    const std::string& next() { return (*toks)[i++]; }
};

inline bool parseSum(Cursor& c, Value& out, std::string& err);

// atom leaf, or an explicit `[ … ]` group (the ladder's parentheses)
inline bool parseFactor(Cursor& c, Value& out, std::string& err) {
    const std::string* t = c.peek();
    if (!t) { err = "unexpected end of stop list"; return false; }
    if (*t == "[") {
        c.next();
        if (const std::string* p = c.peek(); p && *p == "]") {   // `[]` — an empty group
            c.next(); out.isLeaf = false; out.items.clear(); return true;
        }
        if (!parseSum(c, out, err)) return false;
        const std::string* close = c.peek();
        if (!close || *close != "]") { err = "unclosed '['"; return false; }
        c.next();
        return true;
    }
    if (*t == "]" || *t == ",") { err = "unexpected '" + *t + "'"; return false; }
    out.isLeaf = true; out.leaf = c.next();
    return true;
}

// whitespace level (`*`): juxtaposed factors -> a group (unwrapped when there is one)
inline bool parseProduct(Cursor& c, Value& out, std::string& err) {
    std::vector<Value> items;
    Value first;
    if (!parseFactor(c, first, err)) return false;
    items.push_back(std::move(first));
    for (;;) {
        const std::string* t = c.peek();
        if (!t || *t == "," || *t == "]") break;
        Value v;
        if (!parseFactor(c, v, err)) return false;
        items.push_back(std::move(v));
    }
    if (items.size() == 1) { out = std::move(items[0]); return true; }
    out.isLeaf = false; out.items = std::move(items);
    return true;
}

// comma level (`+`): products separated by `,` -> a group (unwrapped when there is one)
inline bool parseSum(Cursor& c, Value& out, std::string& err) {
    std::vector<Value> items;
    Value first;
    if (!parseProduct(c, first, err)) return false;
    items.push_back(std::move(first));
    while (const std::string* t = c.peek()) {
        if (*t != ",") break;
        c.next();
        // A TRAILING top-level comma is meaningful, not a typo: it is how a *lone*
        // vector stop says "I am one stop of N components", not N scalar stops
        // (`tint 0 0 0,`). Accept it here and let the group stand at size 1.
        const std::string* p = c.peek();
        if (!p || *p == "]") break;
        Value v;
        if (!parseProduct(c, v, err)) return false;
        items.push_back(std::move(v));
    }
    if (items.size() == 1) { out = std::move(items[0]); return true; }
    out.isLeaf = false; out.items = std::move(items);
    return true;
}

}  // namespace detail

// Parse a ladder token stream. `trailingComma` reports whether the top level ended on a
// comma — the disambiguator that turns a single juxtaposed run into one vector stop.
inline bool parse(const std::vector<std::string>& toks, Value& out,
                  bool& trailingComma, std::string& err) {
    trailingComma = false;
    if (toks.empty()) { err = "empty stop list"; return false; }
    for (size_t i = 0; i + 1 < toks.size(); ++i)
        if (toks[i] == "," && toks[i + 1] == ",") { err = "empty stop (stray comma?)"; return false; }
    if (toks.front() == ",") { err = "empty stop (leading comma?)"; return false; }
    detail::Cursor c{ &toks, 0 };
    if (!detail::parseSum(c, out, err)) return false;
    if (const std::string* t = c.peek()) { err = "unexpected '" + *t + "'"; return false; }
    trailingComma = (toks.back() == ",");
    return true;
}

}  // namespace recladder
