// ftsl_shim.hpp — the J3c grammar-validation shim.
//
// Runs the shared .ftsl grammar (loom/grammar/ftsl_scene.epeg, compiled to a
// GPDA graph by loom.grammar.emit_cpp -> ftsl_scene.gen.cpp) alongside ftrace's
// hand-written parser and structurally diffs the two Block trees.  It is
// NON-AUTHORITATIVE: ftrace always renders from its own parse; the shim only
// warns (to stderr) when the shared grammar would produce a different tree, so
// we can drive the mismatch count to zero across the whole scene corpus before
// flipping ftrace's front-end over to the shared grammar as the single source
// of truth.
//
// Enabled by the `-validate-grammar` CLI flag (see main.cpp) or the
// FTRACE_VALIDATE_GRAMMAR environment variable.  Off by default => zero cost.
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

namespace ftsl_shim {

// Set by main.cpp's `-validate-grammar` flag.  The env var is an alternate opt-in.
inline bool& enabled_flag() {
    static bool v = false;
    return v;
}

inline bool is_enabled() {
    if (enabled_flag()) return true;
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

// Compare the shared-grammar parse of `src` against ftrace's own `blocks_old`.
// Returns true on a clean match (or when disabled); prints a warning + the first
// few structural differences to stderr on a mismatch or a GPDA-side parse error.
// Never throws and never affects the render — purely a diagnostic.
inline bool validate(const std::string& src,
                     const std::vector<ftsl::Block>& blocks_old,
                     const std::string& path = "") {
    if (!is_enabled()) return true;
    const std::string where = path.empty() ? std::string("<scene>") : path;
    try {
        auto tokens = lexer().tokenize(src);
        auto tree = parser().parse(tokens);
        if (!tree) throw std::runtime_error("parse returned null");
        std::vector<ftsl::Block> blocks_new = shim::reduce_scene(tree.get());
        shim::Diff d = shim::diff_scene(blocks_old, blocks_new);
        if (d.ok()) return true;
        std::fprintf(stderr,
            "[validate-grammar] %s: shared grammar disagrees with ftrace's "
            "parser (%zu diffs):\n", where.c_str(), d.msgs.size());
        std::size_t n = 0;
        for (const auto& m : d.msgs) {
            std::fprintf(stderr, "    %s\n", m.c_str());
            if (++n >= 12) { std::fprintf(stderr, "    ...\n"); break; }
        }
        return false;
    } catch (const std::exception& e) {
        std::fprintf(stderr,
            "[validate-grammar] %s: shared grammar failed to parse a scene "
            "ftrace accepted: %s\n", where.c_str(), e.what());
        return false;
    }
}

}  // namespace ftsl_shim
