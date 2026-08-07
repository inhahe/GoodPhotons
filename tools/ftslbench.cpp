// ftslbench — time the two grammar-driven stages of loading a .ftsl scene.
//
// `ftsl::load` splits into parse (text -> Block tree) and build (Block tree ->
// Scene).  The viewer's per-frame profile showed `parse` is the larger half, and
// at ~200 KB/s it is far too slow for what it does -- so this pins the cost on
// one of the two stages the front end is actually made of:
//
//   tokenize  src text  -> std::vector<gpda_tok::Token>   (regex longest-match lexer)
//   parse     tokens    -> parse tree                     (GPDA graph walk)
//
// It deliberately stops short of reduce_scene(), which would drag in all of
// ftsl.h; if these two already account for the measured time, the reduction is
// not where to look.  Build with tools/ftslbench.bat.
#include <algorithm>
#include <chrono>
#include <utility>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "tokenized.hpp"
#include "gpda_lexer.hpp"

namespace ftsl_gen {
gpda_tok::Graph build_ftsl_scene_graph();
std::vector<gpda_lex::LexRule> ftsl_scene_lex_rules();
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// Canonical, fully-ordered rendering of a parse tree -- one line per node, indented
// by depth, carrying name/value/line/col.  Two builds of the engine that agree on
// this for every scene in the corpus produced the same tree, which is the only
// correctness claim an optimisation inside the parser needs to make.
static void dump_tree(const gpda_tok::ParseNode* n, int depth, std::string& out) {
    out.append((std::size_t)depth * 2, ' ');
    out += n->name;
    out += '\t';
    out += n->value;
    out += '\t';
    out += std::to_string(n->line);
    out += ':';
    out += std::to_string(n->col);
    out += '\n';
    for (const auto& c : n->children) dump_tree(c.get(), depth + 1, out);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: ftslbench <scene.ftsl> [reps]\n"
                     "       ftslbench --dump <scene.ftsl>   (canonical parse tree)\n");
        return 2;
    }
    // --dump: parse once and print the tree, for an old-vs-new equivalence diff.
    // Takes a LIST file (one scene path per line) so the whole corpus is one process
    // rather than thousands of spawns.
    if (std::string(argv[1]) == "--dump") {
        if (argc < 3) { std::fprintf(stderr, "--dump needs a list file\n"); return 2; }
        std::ifstream lf(argv[2]);
        if (!lf) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 2; }
        gpda_lex::Lexer  dlex(ftsl_gen::ftsl_scene_lex_rules());
        gpda_tok::Parser dpar;
        dpar.graph = ftsl_gen::build_ftsl_scene_graph();
        std::string path;
        std::string out;
        while (std::getline(lf, path)) {
            while (!path.empty() && (path.back() == '\r' || path.back() == '\n'))
                path.pop_back();
            if (path.empty()) continue;
            out += "=== ";
            out += path;
            out += '\n';
            std::ifstream df(path);
            if (!df) { out += "CANNOT-OPEN\n"; continue; }
            std::stringstream ds; ds << df.rdbuf();
            try {
                auto tree = dpar.parse(dlex.tokenize(ds.str()));
                if (!tree) { out += "NO-TREE\n"; continue; }
                dump_tree(tree.get(), 0, out);
            } catch (const std::exception& e) {
                // A scene that fails to parse must fail IDENTICALLY in both builds,
                // so the error text is part of the comparison, not a skip.
                out += "ERROR\t";
                out += e.what();
                out += '\n';
            }
        }
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    }

    const int reps = (argc > 2) ? std::atoi(argv[2]) : 20;

    std::ifstream f(argv[1]);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    std::stringstream ss; ss << f.rdbuf();
    const std::string src = ss.str();

    gpda_lex::Lexer  lexer(ftsl_gen::ftsl_scene_lex_rules());
    gpda_tok::Parser parser;
    parser.graph = ftsl_gen::build_ftsl_scene_graph();

    // Warm up (first call touches the graph/regex caches), and report what the
    // token stream is actually made of.  A `skip_types` token (whitespace, comments)
    // is invisible to the grammar but still costs a full expand_all/dedup round in
    // the graph walk, so its share of the stream is a share of the parse time.
    std::size_t ntok = 0;
    {
        auto t = lexer.tokenize(src);
        ntok = t.size();
        parser.parse(t);
        std::vector<std::pair<std::string, std::size_t>> hist;
        for (const auto& tk : t) {
            bool found = false;
            for (auto& h : hist) if (h.first == tk.type) { ++h.second; found = true; break; }
            if (!found) hist.push_back({tk.type, 1});
        }
        std::sort(hist.begin(), hist.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::printf("token stream (%zu tokens):\n", ntok);
        for (const auto& h : hist)
            std::printf("  %-12s %6zu  (%4.1f%%)%s\n", h.first.c_str(), h.second,
                        100.0 * (double)h.second / (double)ntok,
                        parser.graph.skip_types.count(h.first) ? "  [SKIPPED by grammar]" : "");
    }

    // Report the MINIMUM per-rep time, not the mean.  This is single-threaded work
    // on a machine that may be doing other things, and background load can only ever
    // make a sample slower -- so the mean measures the machine's mood while the min
    // measures the code.  (Mean is printed too: a large min/mean gap is itself the
    // signal that the run was contended and the mean should not be quoted.)
    double lexSum = 0.0, parseSum = 0.0;
    double lexMin = 1e300, parseMin = 1e300;
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        auto toks = lexer.tokenize(src);
        double l = ms_since(t0);

        auto t1 = std::chrono::steady_clock::now();
        auto tree = parser.parse(toks);
        double p = ms_since(t1);
        if (!tree) { std::fprintf(stderr, "no parse tree\n"); return 1; }

        lexSum += l;     if (l < lexMin)   lexMin = l;
        parseSum += p;   if (p < parseMin) parseMin = p;
    }

    std::printf("%s: %zu bytes, %zu tokens, %d reps\n", argv[1], src.size(), ntok, reps);
    std::printf("  tokenize  min %7.2f ms  mean %7.2f ms   (%.2f us/token, %.0f KB/s)\n",
                lexMin, lexSum / reps, 1000.0 * lexMin / (double)ntok,
                (double)src.size() / lexMin / 1.024);
    std::printf("  parse     min %7.2f ms  mean %7.2f ms   (%.2f us/token)\n",
                parseMin, parseSum / reps, 1000.0 * parseMin / (double)ntok);
    std::printf("  total     min %7.2f ms  mean %7.2f ms\n",
                lexMin + parseMin, (lexSum + parseSum) / reps);
    return 0;
}
