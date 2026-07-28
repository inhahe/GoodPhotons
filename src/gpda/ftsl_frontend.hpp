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
// tree, structurally identical down to Stmt::line) in 0.68.0, held there for ten
// releases with `-legacy-parser` available and unused, and in 0.79.0 the old
// parser, the escape hatch and the cross-check differ were all deleted — there is
// now exactly one implementation of the .ftsl language in ftrace.
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

}  // namespace ftsl_gpda
