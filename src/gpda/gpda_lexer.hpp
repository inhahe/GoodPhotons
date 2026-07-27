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
//
// PERFORMANCE.  The naive form of this loop — run all N patterns through
// std::regex_search at every token — was the dominant cost of loading a .ftsl
// scene once the shared grammar became ftrace's front end: ~8.7 us per byte of
// scene text, i.e. ~200 ms for the largest scene in the tree, dwarfing the
// graph walk itself.  std::regex costs on the order of a microsecond per call
// and FTSL's table has 16 rules, so nearly every byte paid for 16 of them.
//
// Two structural fixes, both derived from the patterns themselves so the
// grammar stays the single source of truth and no rule table needs annotating:
//
//   1. FIRST-BYTE PREFILTER.  Each pattern is analysed once, at construction,
//      into the set of bytes it can possibly start with.  At each position only
//      the rules whose first-set contains the current byte are tried at all.
//      The analysis is conservative: anything it cannot prove is reported as
//      "unknown", and an unknown rule is always tried, so a prefilter miss can
//      only ever cost speed, never a token.
//
//   2. LITERAL FAST PATH.  A pattern with no metacharacters is just a string
//      (FTSL's table is 9/16 keywords and punctuation).  Those match with a
//      compare instead of a regex.
//
// Both preserve the longest-match / declaration-order-tiebreak semantics
// exactly; std::regex remains the fallback for anything not recognised.
#pragma once

#include <bitset>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

namespace detail {

using ByteSet = std::bitset<256>;

inline void set_range(ByteSet& s, unsigned char lo, unsigned char hi) {
    for (unsigned c = lo; c <= hi; ++c) s.set(c);
}

// The character an escape sequence stands for, or -1 if it stands for a class
// (\d, \w, …) or something we don't model.
inline int escape_char(char c) {
    switch (c) {
        case 'n': return '\n';
        case 'r': return '\r';
        case 't': return '\t';
        case 'f': return '\f';
        case 'v': return '\v';
        case '0': return '\0';
        default:
            // A punctuation escape (\., \[, \\, …) stands for itself.  A letter
            // we don't recognise is a class shorthand or an anchor.
            return (std::isalnum(static_cast<unsigned char>(c)) ? -1
                                                                : (unsigned char)c);
    }
}

// Byte set of a shorthand class escape, or false if unrecognised.
inline bool escape_class(char c, ByteSet& out) {
    ByteSet s;
    switch (std::tolower(static_cast<unsigned char>(c))) {
        case 'd': set_range(s, '0', '9'); break;
        case 'w': set_range(s, 'a', 'z'); set_range(s, 'A', 'Z');
                  set_range(s, '0', '9'); s.set('_'); break;
        case 's': for (char ws : {' ', '\t', '\n', '\r', '\f', '\v'}) s.set((unsigned char)ws);
                  break;
        default: return false;
    }
    if (std::isupper(static_cast<unsigned char>(c))) s.flip();  // \D \W \S
    out = s;
    return true;
}

// Recursive-descent first-set analyser over the ECMAScript subset that .epeg
// lex patterns use: alternation, concatenation, groups, char classes, `.`,
// escapes, literals, and the ?/*/+ quantifiers.
//
// Returns false the moment it meets anything it does not model — the caller
// then treats the rule as "always try", which is always safe.  It must never
// return a set that is too SMALL, so every unmodelled construct is a bail-out
// rather than a guess.
class FirstSet {
public:
    static bool analyse(const std::string& p, ByteSet& out) {
        FirstSet f(p);
        ByteSet s;
        bool nullable = false;
        if (!f.alt(s, nullable) || f.i_ != p.size()) return false;
        // A pattern that can match the empty string can start with anything;
        // such a rule contributes nothing to the prefilter.
        if (nullable) return false;
        out = s;
        return true;
    }

private:
    explicit FirstSet(const std::string& p) : p_(p) {}

    const std::string& p_;
    std::size_t i_ = 0;

    bool eof() const { return i_ >= p_.size(); }
    char peek() const { return p_[i_]; }

    // alt := seq ('|' seq)*
    bool alt(ByteSet& out, bool& nullable) {
        ByteSet acc;
        bool any_nullable = false;
        for (;;) {
            ByteSet s;
            bool n = false;
            if (!seq(s, n)) return false;
            acc |= s;
            any_nullable = any_nullable || n;
            if (!eof() && peek() == '|') { ++i_; continue; }
            break;
        }
        out = acc;
        nullable = any_nullable;
        return true;
    }

    // seq := atom*  — the first-set accumulates across leading nullable atoms.
    bool seq(ByteSet& out, bool& nullable) {
        ByteSet acc;
        bool still_nullable = true;   // no atom consumed yet => matches empty
        while (!eof() && peek() != '|' && peek() != ')') {
            ByteSet s;
            bool n = false;
            if (!atom(s, n)) return false;
            if (still_nullable) acc |= s;
            if (!n) still_nullable = false;
        }
        out = acc;
        nullable = still_nullable;
        return true;
    }

    // atom := element quantifier?
    bool atom(ByteSet& out, bool& nullable) {
        ByteSet s;
        if (!element(s)) return false;
        bool opt = false;
        if (!eof()) {
            char q = peek();
            if (q == '{') return false;                  // {n,m}: not modelled
            if (q == '*' || q == '?' || q == '+') {
                opt = (q != '+');
                ++i_;
                // A lazy modifier (`*?`, `+?`, `??`) doesn't change the
                // first set, but it does need consuming.
                if (!eof() && peek() == '?') ++i_;
            }
        }
        out = s;
        nullable = opt;
        return true;
    }

    bool element(ByteSet& out) {
        if (eof()) return false;
        char c = p_[i_];
        if (c == '(') {
            ++i_;
            // Only plain and non-capturing groups; lookaround changes the
            // first-set rules, so bail on `(?=`, `(?!`, `(?<`.
            if (i_ + 1 < p_.size() && p_[i_] == '?') {
                if (p_[i_ + 1] != ':') return false;
                i_ += 2;
            }
            ByteSet s;
            bool n = false;
            if (!alt(s, n)) return false;
            if (eof() || peek() != ')') return false;
            ++i_;
            if (n) return false;         // nullable group: give up
            out = s;
            return true;
        }
        if (c == '[') return charclass(out);
        if (c == '.') {
            ++i_;
            out.set();
            out.reset('\n');             // ECMAScript `.` excludes newline
            return true;
        }
        if (c == '\\') {
            ++i_;
            if (eof()) return false;
            char e = p_[i_++];
            ByteSet cls;
            if (escape_class(e, cls)) { out = cls; return true; }
            int ch = escape_char(e);
            if (ch < 0) return false;    // anchor or unmodelled escape
            out.reset();
            out.set((unsigned char)ch);
            return true;
        }
        // Anchors and quantifier-position metacharacters have no first-set
        // meaning here; refuse rather than mis-model them.
        if (c == '^' || c == '$' || c == '*' || c == '+' || c == '?' ||
            c == '{' || c == '}' || c == ')') return false;
        ++i_;
        out.reset();
        out.set((unsigned char)c);
        return true;
    }

    // [...] / [^...], with ranges and escapes.
    bool charclass(ByteSet& out) {
        ++i_;                                  // '['
        bool negate = false;
        if (!eof() && peek() == '^') { negate = true; ++i_; }
        ByteSet s;
        bool first = true;
        while (!eof() && (peek() != ']' || first)) {
            first = false;
            int lo;
            if (peek() == '\\') {
                ++i_;
                if (eof()) return false;
                char e = p_[i_++];
                ByteSet cls;
                if (escape_class(e, cls)) { s |= cls; continue; }
                lo = escape_char(e);
                if (lo < 0) return false;
            } else {
                lo = (unsigned char)p_[i_++];
            }
            // range?
            if (i_ + 1 < p_.size() && p_[i_] == '-' && p_[i_ + 1] != ']') {
                ++i_;
                int hi;
                if (peek() == '\\') {
                    ++i_;
                    if (eof()) return false;
                    hi = escape_char(p_[i_++]);
                    if (hi < 0) return false;
                } else {
                    hi = (unsigned char)p_[i_++];
                }
                if (hi < lo) return false;
                set_range(s, (unsigned char)lo, (unsigned char)hi);
            } else {
                s.set((unsigned char)lo);
            }
        }
        if (eof()) return false;               // unterminated class
        ++i_;                                  // ']'
        if (negate) s.flip();
        out = s;
        return true;
    }
};

// A pattern that is just a run of ordinary characters (and character escapes)
// is a plain string and needs no regex engine.
inline bool literal_pattern(const std::string& p, std::string& out) {
    std::string s;
    for (std::size_t i = 0; i < p.size(); ++i) {
        char c = p[i];
        if (c == '\\') {
            if (++i >= p.size()) return false;
            int ch = escape_char(p[i]);
            if (ch < 0) return false;
            s.push_back((char)ch);
            continue;
        }
        // Anything that could quantify, group, anchor or branch disqualifies it.
        if (std::strchr("[](){}.*+?|^$", c) != nullptr) return false;
        s.push_back(c);
    }
    if (s.empty()) return false;
    out = s;
    return true;
}

}  // namespace detail

class Lexer {
public:
    explicit Lexer(const std::vector<LexRule>& rules) {
        compiled_.reserve(rules.size());
        for (const auto& r : rules) {
            Compiled c;
            c.name = r.name;
            c.is_literal = detail::literal_pattern(r.pattern, c.literal);
            if (!c.is_literal) {
                c.re = std::regex(r.pattern, std::regex::ECMAScript);
            }
            if (c.is_literal) {
                c.first.reset();
                c.first.set((unsigned char)c.literal[0]);
                c.first_known = true;
            } else {
                c.first_known = detail::FirstSet::analyse(r.pattern, c.first);
            }
            compiled_.push_back(std::move(c));
        }
    }

    // Tokenize `text`.  Throws std::runtime_error on an unlexable character.
    std::vector<gpda_tok::Token> tokenize(const std::string& text) const {
        std::vector<gpda_tok::Token> out;
        out.reserve(text.size() / 4 + 8);      // ~4 bytes/token on real .ftsl
        std::size_t pos = 0;
        std::uint32_t line = 1, col = 1;
        const auto begin = text.cbegin();
        const auto end = text.cend();
        std::smatch m;
        while (pos < text.size()) {
            const unsigned char cur = (unsigned char)text[pos];
            const std::string* best_name = nullptr;
            std::size_t best_len = 0;
            for (const auto& c : compiled_) {
                // Prefilter: skip rules that provably cannot start here.  An
                // unanalysable rule has first_known == false and is always run.
                if (c.first_known && !c.first.test(cur)) continue;

                std::size_t len;
                if (c.is_literal) {
                    if (text.compare(pos, c.literal.size(), c.literal) != 0)
                        continue;
                    len = c.literal.size();
                } else if (std::regex_search(begin + pos, end, m, c.re,
                               std::regex_constants::match_continuous)) {
                    len = static_cast<std::size_t>(m.length(0));
                } else {
                    continue;
                }
                // strictly-greater keeps the earliest rule on ties
                if (len > best_len) {
                    best_len = len;
                    best_name = &c.name;
                }
            }
            if (best_name == nullptr || best_len == 0) {
                throw std::runtime_error(
                    "gpda_lex: unexpected character '" +
                    std::string(1, text[pos]) + "' at line " +
                    std::to_string(line) + ", col " + std::to_string(col));
            }
            out.push_back(gpda_tok::Token{*best_name,
                                          text.substr(pos, best_len), line, col});
            const std::string& value = out.back().value;
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
    struct Compiled {
        std::string name;
        std::regex re;                 // unused when is_literal
        std::string literal;
        detail::ByteSet first;
        bool first_known = false;
        bool is_literal = false;
    };
    std::vector<Compiled> compiled_;
};

}  // namespace gpda_lex
