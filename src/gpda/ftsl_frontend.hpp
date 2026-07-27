// ftsl_frontend.hpp — ftrace's AUTHORITATIVE .ftsl front end.
//
// Parses a scene with the shared grammar (tools/loom/loom/grammar/ftsl_scene.epeg,
// compiled to a GPDA graph by loom.grammar.emit_cpp -> ftsl_scene.gen.cpp), then
// reduces the parse tree to the std::vector<ftsl::Block> shape the rest of ftrace
// consumes (see ftsl_reduce.hpp).  One grammar is now the single source of truth
// for the .ftsl language, shared by ftrace and by loom's Python tooling.
//
// History: this started life as a non-authoritative validation shim that ran
// alongside ftrace's hand-written recursive-descent parser and diffed the two
// Block trees, so the mismatch count could be driven to zero across the whole
// corpus before flipping over.  It reached MATCH 2595/2595 (every .ftsl in the
// tree, structurally identical down to Stmt::line), so `parse()` below is now the
// default path.  The old parser is still compiled in behind `-legacy-parser` as a
// one-release escape hatch, and `validate()` still cross-checks the two when
// `-validate-grammar` is given.
//
// Included by ftsl.h *after* ftsl::Block/Stmt/Value are defined.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tokenized.hpp"
#include "gpda_lexer.hpp"
#include "ftsl_reduce.hpp"

// Provided by the generated translation unit ftsl_scene.gen.cpp.
namespace ftsl_gen {
gpda_tok::Graph build_ftsl_scene_graph();
std::vector<gpda_lex::LexRule> ftsl_scene_lex_rules();
}

namespace ftsl_gpda {

// Set by main.cpp's `-legacy-parser` flag: fall back to ftrace's hand-written
// recursive-descent parser instead of the shared grammar.  Escape hatch only —
// slated for removal once the grammar has a release of field time.
inline bool& legacy_flag() {
    static bool v = false;
    return v;
}

inline bool use_legacy() {
    if (legacy_flag()) return true;
    static const bool env = [] {
        const char* e = std::getenv("FTRACE_LEGACY_PARSER");
        return e && *e && std::string(e) != "0";
    }();
    return env;
}

// Set by main.cpp's `-validate-grammar` flag.  The env var is an alternate opt-in.
inline bool& validate_flag() {
    static bool v = false;
    return v;
}

inline bool validate_enabled() {
    if (validate_flag()) return true;
    static const bool env = [] {
        const char* e = std::getenv("FTRACE_VALIDATE_GRAMMAR");
        return e && *e && std::string(e) != "0";
    }();
    return env;
}

// Build the GPDA parser + lexer once (the graph is immutable).
inline gpda_tok::Parser& parser() {
    static gpda_tok::Parser p = [] {
        gpda_tok::Parser q;
        q.graph = ftsl_gen::build_ftsl_scene_graph();
        return q;
    }();
    return p;
}

inline gpda_lex::Lexer& lexer() {
    static gpda_lex::Lexer l(ftsl_gen::ftsl_scene_lex_rules());
    return l;
}

// Parse `src` with the shared grammar into ftrace's Block tree.
// Returns false and sets `err` on a syntax error.  Never throws:
// gpda_tok::ParseError already carries line/col, the exact set of accepted
// continuations and the enclosing rule chain, so its message is used verbatim
// (it already begins "line N, col C: ...", matching ftrace's error convention).
inline bool parse(const std::string& src, std::vector<ftsl::Block>& out,
                  std::string& err) {
    try {
        auto tokens = lexer().tokenize(src);
        auto tree = parser().parse(tokens);
        if (!tree) { err = "scene grammar produced no parse tree"; return false; }
        out = reduce_scene(tree.get());
        return true;
    } catch (const gpda_tok::ParseError& e) {
        err = e.what();
        return false;
    } catch (const std::exception& e) {
        err = std::string("scene grammar error: ") + e.what();
        return false;
    }
}

// Cross-check the two front ends against each other (`-validate-grammar`).
// `blocks_gpda` is the authoritative parse; `src` is re-parsed with the legacy
// hand-written parser and the two trees are structurally diffed.  Returns true on
// a clean match (or when disabled); prints the first few differences to stderr
// otherwise.  Never throws and never affects the render — purely a diagnostic.
//
// The legacy re-parse is supplied by the caller (ftsl.h) as a callback, because
// ftsl::Parser is defined further down that header than this one is included.
template <class LegacyParseFn>
inline bool validate(const std::vector<ftsl::Block>& blocks_gpda,
                     const LegacyParseFn& legacy_parse,
                     const std::string& path = "") {
    if (!validate_enabled()) return true;
    const std::string where = path.empty() ? std::string("<scene>") : path;
    std::vector<ftsl::Block> blocks_old;
    std::string old_err;
    if (!legacy_parse(blocks_old, old_err)) {
        std::fprintf(stderr,
            "[validate-grammar] %s: legacy parser rejected a scene the shared "
            "grammar accepted: %s\n", where.c_str(), old_err.c_str());
        return false;
    }
    Diff d = diff_scene(blocks_old, blocks_gpda);
    if (d.ok()) return true;
    std::fprintf(stderr,
        "[validate-grammar] %s: shared grammar disagrees with ftrace's legacy "
        "parser (%zu diffs):\n", where.c_str(), d.msgs.size());
    std::size_t n = 0;
    for (const auto& m : d.msgs) {
        std::fprintf(stderr, "    %s\n", m.c_str());
        if (++n >= 12) { std::fprintf(stderr, "    ...\n"); break; }
    }
    return false;
}

}  // namespace ftsl_gpda
