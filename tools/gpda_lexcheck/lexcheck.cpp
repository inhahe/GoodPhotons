// lexcheck.cpp — differential validation of gpda_lexer.hpp's fast paths.
//
// WHY THIS EXISTS.  src/gpda/gpda_lexer.hpp does not run every rule's regex at
// every position.  It derives, from each pattern's own text, the set of bytes
// that pattern can start with, and skips rules whose first-set excludes the
// current byte; patterns with no metacharacters bypass std::regex entirely and
// match by string compare.  Both are pure speed optimisations *provided* the
// first-set analysis is never too small and the literal path agrees with the
// regex it replaces.  If either assumption breaks, the lexer silently produces
// different tokens — no crash, no diagnostic, just a scene that parses wrong or
// stops parsing.  The `-validate-grammar` cross-check against the hand-written
// parser would once have caught it, but both went away in 0.79.0 — the shared
// grammar is now the only front end, so this check is the only line of defence.
//
// WHAT IT CHECKS.  For every rule in the real FTSL lex table:
//   1. For each of the 256 leading bytes the analyser *excluded*, ask
//      std::regex directly whether any of ~400 probe strings starting with that
//      byte matches.  A match means the prefilter dropped a rule that could
//      have fired — a hard failure.
//   2. For every literal-fast-path rule, check the literal is exactly what its
//      own regex matches.
// And, given a scene file argument, that the fast lexer's whole token stream is
// identical to a reference all-regex longest-match loop, plus throughput for
// both.
//
// Run:  tools\gpda_lexcheck\build.bat  &&  tools\gpda_lexcheck\lexcheck.exe scenes\gallery_settled.ftsl
// Exit code is nonzero if anything disagrees.
#include "gpda_lexer.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

namespace ftsl_gen {
gpda_tok::Graph build_ftsl_scene_graph();
std::vector<gpda_lex::LexRule> ftsl_scene_lex_rules();
}

static int failures = 0;
static void check(bool ok, const std::string& what) {
    if (!ok) { std::printf("  FAIL: %s\n", what.c_str()); ++failures; }
}

int main(int argc, char** argv) {
    auto rules = ftsl_gen::ftsl_scene_lex_rules();
    std::printf("--- %zu lex rules ---\n", rules.size());

    // 1. Differential first-set check.  The probe alphabet deliberately covers
    // every character class the FTSL patterns discriminate on (delimiters,
    // digits, sign, dot, exponent letters, quote, '=') so a mis-analysed
    // pattern has something to match against.
    std::mt19937 rng(12345);
    const std::string alphabet =
        " \t\r\n{}[]#\"=+-.0123456789abcdefghijklmnopqrstuvwxyzABCXYZ_/*";
    std::vector<std::string> suffixes = {""};
    for (int i = 0; i < 400; ++i) {
        std::string s;
        int len = 1 + (int)(rng() % 6);
        for (int j = 0; j < len; ++j) s.push_back(alphabet[rng() % alphabet.size()]);
        suffixes.push_back(s);
    }

    for (const auto& r : rules) {
        std::regex re(r.pattern, std::regex::ECMAScript);
        gpda_lex::detail::ByteSet first;
        bool known = gpda_lex::detail::FirstSet::analyse(r.pattern, first);
        std::string lit;
        bool is_lit = gpda_lex::detail::literal_pattern(r.pattern, lit);
        if (is_lit) { known = true; first.reset(); first.set((unsigned char)lit[0]); }

        int excluded = 0;
        for (int b = 0; b < 256; ++b) {
            if (!known || first.test(b)) { continue; }
            ++excluded;
            for (const auto& suf : suffixes) {
                std::string probe((char)b + suf);
                std::smatch m;
                if (std::regex_search(probe, m, re,
                        std::regex_constants::match_continuous) &&
                    m.length(0) > 0) {
                    check(false, r.name + " prefilter excluded byte " +
                          std::to_string(b) + " but regex matched \"" + probe +
                          "\" (len " + std::to_string(m.length(0)) + ")");
                    break;
                }
            }
        }
        // And the literal fast path must agree with the regex on length.
        if (is_lit) {
            std::smatch m;
            bool ok = std::regex_search(lit, m, re,
                        std::regex_constants::match_continuous) &&
                      (std::size_t)m.length(0) == lit.size();
            check(ok, r.name + " literal fast path disagrees with its regex");
        }
        std::printf("  %-10s %-14s excludes %3d/256 bytes\n", r.name.c_str(),
                    is_lit ? "[literal]" : (known ? "[first-set]" : "[UNKNOWN]"),
                    excluded);
    }

    // 2. Token-stream equivalence + throughput on a real scene.
    if (argc > 1) {
        std::ifstream f(argv[1], std::ios::binary);
        std::ostringstream ss; ss << f.rdbuf();
        std::string src = ss.str();
        if (src.empty()) {
            std::printf("\ncould not read %s\n", argv[1]);
            return 2;
        }
        gpda_lex::Lexer lex(rules);

        auto toks = lex.tokenize(src);
        std::printf("\n--- %s: %zu bytes -> %zu tokens ---\n", argv[1],
                    src.size(), toks.size());

        // Reference tokenizer: the naive all-regex longest-match loop that the
        // fast paths replaced.
        std::vector<std::pair<std::string, std::string>> ref;
        {
            std::vector<std::pair<std::string, std::regex>> c;
            for (const auto& r : rules)
                c.emplace_back(r.name, std::regex(r.pattern, std::regex::ECMAScript));
            std::size_t pos = 0; std::smatch m;
            while (pos < src.size()) {
                const std::string* bn = nullptr; std::size_t bl = 0;
                for (const auto& x : c) {
                    if (std::regex_search(src.cbegin() + pos, src.cend(), m, x.second,
                            std::regex_constants::match_continuous)) {
                        std::size_t len = m.length(0);
                        if (len > bl) { bl = len; bn = &x.first; }
                    }
                }
                if (!bn || !bl) break;
                ref.emplace_back(*bn, src.substr(pos, bl));
                pos += bl;
            }
        }
        // toks carries a trailing EOF sentinel that the reference loop has no
        // notion of, hence the +1.
        check(ref.size() + 1 == toks.size(), "token count matches the reference lexer");
        for (std::size_t i = 0; i < ref.size() && i + 1 < toks.size(); ++i) {
            if (ref[i].first != toks[i].type || ref[i].second != toks[i].value) {
                check(false, "token " + std::to_string(i) + ": ref " +
                      ref[i].first + "/'" + ref[i].second + "' vs fast " +
                      toks[i].type + "/'" + toks[i].value + "'");
                break;
            }
        }

        // Time the reference loop for comparison.
        {
            std::vector<std::pair<std::string, std::regex>> c;
            for (const auto& r : rules)
                c.emplace_back(r.name, std::regex(r.pattern, std::regex::ECMAScript));
            auto r0 = std::chrono::steady_clock::now();
            for (int rep = 0; rep < 5; ++rep) {
                std::size_t pos = 0; std::smatch m;
                while (pos < src.size()) {
                    const std::string* bn = nullptr; std::size_t bl = 0;
                    for (const auto& x : c) {
                        if (std::regex_search(src.cbegin() + pos, src.cend(), m, x.second,
                                std::regex_constants::match_continuous)) {
                            std::size_t len = m.length(0);
                            if (len > bl) { bl = len; bn = &x.first; }
                        }
                    }
                    if (!bn || !bl) break;
                    (void)src.substr(pos, bl);
                    pos += bl;
                }
            }
            auto r1 = std::chrono::steady_clock::now();
            std::printf("  lex (all-regex reference): %.3f ms\n",
                        std::chrono::duration<double, std::milli>(r1 - r0).count() / 5);
        }

        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) (void)lex.tokenize(src);
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
        std::printf("  lex: %.3f ms  (%.2f us/KB)\n", ms, ms * 1000.0 / (src.size() / 1024.0));

        gpda_tok::Parser p; p.graph = ftsl_gen::build_ftsl_scene_graph();
        (void)p.parse(toks);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20; ++i) (void)p.parse(toks);
        t1 = std::chrono::steady_clock::now();
        double pms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
        std::printf("  parse: %.3f ms   total: %.3f ms\n", pms, ms + pms);
    }

    std::printf("\n%s\n", failures ? "FAILURES" : "ALL OK");
    return failures ? 1 : 0;
}
