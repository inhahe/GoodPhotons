// gpda_lexer.hpp — regex longest-match lexer for the tokenized GPDA engine.
//
// A faithful C++ port of loom's Python auto-lexer (loom/grammar/_gpda.py,
// class Lexer): rules are tried in declaration order; the LONGEST match at the
// current position wins; ties are broken by declaration order (earlier rule
// wins).  An 'EOF' sentinel token is appended.  Whitespace/comment skipping is
// NOT done here — those token types are emitted and the parser's skip_types
// makes them invisible (matching the Python split: Lexer.ignore is empty, the
// GraphParser carries skip_types).
//
// The rule table (name, ECMAScript-regex pattern) is generated from the .epeg
// grammar by loom.grammar.emit_cpp, so the grammar stays the single source of
// truth.  Patterns are matched anchored at the cursor via match_continuous.
#pragma once

#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "tokenized.hpp"

namespace gpda_lex {

struct LexRule {
    std::string name;
    std::string pattern;
};

class Lexer {
public:
    explicit Lexer(const std::vector<LexRule>& rules) {
        compiled_.reserve(rules.size());
        for (const auto& r : rules) {
            compiled_.push_back({r.name,
                std::regex(r.pattern, std::regex::ECMAScript)});
        }
    }

    // Tokenize `text`.  Throws std::runtime_error on an unlexable character.
    std::vector<gpda_tok::Token> tokenize(const std::string& text) const {
        std::vector<gpda_tok::Token> out;
        std::size_t pos = 0;
        std::uint32_t line = 1, col = 1;
        const auto begin = text.cbegin();
        const auto end = text.cend();
        std::smatch m;
        while (pos < text.size()) {
            const std::string* best_name = nullptr;
            std::size_t best_len = 0;
            for (const auto& c : compiled_) {
                if (std::regex_search(begin + pos, end, m, c.re,
                        std::regex_constants::match_continuous)) {
                    std::size_t len = static_cast<std::size_t>(m.length(0));
                    // strictly-greater keeps the earliest rule on ties
                    if (len > best_len) {
                        best_len = len;
                        best_name = &c.name;
                    }
                }
            }
            if (best_name == nullptr || best_len == 0) {
                throw std::runtime_error(
                    "gpda_lex: unexpected character '" +
                    std::string(1, text[pos]) + "' at line " +
                    std::to_string(line) + ", col " + std::to_string(col));
            }
            std::string value = text.substr(pos, best_len);
            out.push_back(gpda_tok::Token{*best_name, value, line, col});
            for (char ch : value) {
                if (ch == '\n') { ++line; col = 1; }
                else { ++col; }
            }
            pos += best_len;
        }
        out.push_back(gpda_tok::Token{"EOF", "", line, col});
        return out;
    }

private:
    struct Compiled { std::string name; std::regex re; };
    std::vector<Compiled> compiled_;
};

}  // namespace gpda_lex
