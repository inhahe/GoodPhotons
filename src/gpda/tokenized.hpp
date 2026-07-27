// tokenized.hpp — C++ port of gpda.py (tokenized parser)
//
// Operates on a pre-lexed token stream.  Same graph-walking algorithm
// as the scannerless version; the differences are:
//   - Terminals are MatchStr (match token.value) or MatchTok (match token.type)
//   - skip_types let whitespace-like tokens be invisible by default but
//     required when explicitly referenced
//   - No backreferences (named captures exist but not backreferences)
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pool.hpp"

namespace gpda_tok {

// ============================================================================
// Token
// ============================================================================

struct Token {
    std::string type;
    std::string value;
    std::uint32_t line = 0;
    std::uint32_t col = 0;
};

// ============================================================================
// ParseError
// ============================================================================

// A syntax error with everything the caller needs to render a good diagnostic.
//
// A GPDA parse dies at a token position where the cursor set is empty, and at
// that exact moment the engine *knows* the complete set of terminals that would
// have been accepted (the expanded cursor set holds nothing but terminal nodes)
// and the chain of rules it was in the middle of (the cursors' rule stacks).
// Throwing that away and reporting only "unexpected X" wastes the one advantage
// a chart-style parser has over a recursive-descent one, so it is captured here:
//
//   expected  — deduped terminal descriptions, in the graph's own link order
//               (which is the grammar's ordered choice, so the most likely
//               continuation tends to come first).  A MatchStr node contributes
//               its literal in quotes ('{'), a MatchTok node its type (STRING).
//   context   — enclosing rule names, innermost first, duplicates collapsed;
//               the "while parsing a …" part of the message.
//
// `what()` is a one-line rendering of all of it; a caller that wants to format
// its own (prefix the filename, print the offending source line with a caret,
// translate rule names into user-facing nouns) has the fields.
struct ParseError : std::runtime_error {
    std::string  token_type;    // offending token's type  ("" if at end of input)
    std::string  token_value;   // offending token's text  ("" if at end of input)
    std::uint32_t line = 0;
    std::uint32_t col  = 0;
    std::size_t  pos   = 0;     // index into the token vector
    bool         at_eof = false;
    std::vector<std::string> expected;
    std::vector<std::string> context;

    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

// Render "'{', '[' or STRING" — the tail of an expected-set message.  Kept
// out-of-class so callers building their own diagnostics can reuse it.
inline std::string join_expected(const std::vector<std::string>& e,
                                 std::size_t max_items = 6) {
    if (e.empty()) return std::string();
    const std::size_t n = std::min(e.size(), max_items);
    std::string s;
    for (std::size_t i = 0; i < n; ++i) {
        if (i) s += (i + 1 == n && n == e.size()) ? " or " : ", ";
        s += e[i];
    }
    if (n < e.size()) s += ", ...";
    return s;
}

// Render a token's text for a one-line diagnostic.  A token value can itself be
// whitespace (a NEWLINE token in a line-oriented grammar is literally "\n"), and
// splicing that raw would break the message across lines and detach it from the
// line/col it reports — so control characters are escaped, C-style.  Over-long
// values are elided in the middle, keeping both ends recognizable.
inline std::string escape_token_text(const std::string& v,
                                     std::size_t max_len = 40) {
    std::string s;
    s.reserve(v.size());
    for (char c : v) {
        switch (c) {
            case '\n': s += "\\n"; break;
            case '\r': s += "\\r"; break;
            case '\t': s += "\\t"; break;
            case '\\': s += "\\\\"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    s += "\\x";
                    s += kHex[(static_cast<unsigned char>(c) >> 4) & 0xF];
                    s += kHex[static_cast<unsigned char>(c) & 0xF];
                } else {
                    s += c;
                }
        }
    }
    if (s.size() > max_len) {
        const std::size_t keep = (max_len - 3) / 2;
        s = s.substr(0, keep) + "..." + s.substr(s.size() - keep);
    }
    return s;
}

// ============================================================================
// Persistent list (same pattern as the scannerless version)
// ============================================================================

template <typename T>
struct PList : gpda_pool::Refcounted<PList<T>> {
    T head;
    gpda_pool::IntrusivePtr<PList> tail;
    std::size_t length;

    PList(T h, gpda_pool::IntrusivePtr<PList> t, std::size_t len)
        : head(std::move(h)), tail(std::move(t)), length(len) {}

    static void deallocate(PList* p) noexcept {
        gpda_pool::Pool<PList>::instance().destroy(p);
    }

    ~PList() {
        auto t = std::move(tail);
        while (t && t.is_unique()) {
            auto next = std::move(t->tail);
            t.reset();
            t = std::move(next);
        }
    }
};

template <typename T>
using PListPtr = gpda_pool::IntrusivePtr<PList<T>>;

template <typename T>
inline PListPtr<T> plist_push(PListPtr<T> tail, T value) {
    std::size_t len = tail ? tail->length : 0;
    auto* p = gpda_pool::Pool<PList<T>>::instance().make(
        std::move(value), std::move(tail), len + 1);
    return PListPtr<T>(p);
}

template <typename T>
inline PListPtr<T> plist_pop(const PListPtr<T>& list) {
    return list ? list->tail : PListPtr<T>();
}

// Convert a persistent list to a vector in push-order (bottom-up).
template <typename T>
inline std::vector<T> plist_to_vector(const PListPtr<T>& list) {
    std::vector<T> out;
    if (!list) return out;
    out.reserve(list->length);
    for (auto p = list.get(); p; p = p->tail.get()) out.push_back(p->head);
    std::reverse(out.begin(), out.end());
    return out;
}

// ============================================================================
// Graph nodes
// ============================================================================

enum class NodeType : std::uint8_t {
    MatchStr,    // match token by its value (e.g. '+' matches PLUS('+') token)
    MatchTok,    // match token by its type  (e.g. NUMBER matches a NUMBER token)
    RuleRef,     // reference another rule (epsilon)
    Split,       // fan-out / join point (epsilon)
    RuleStart,   // rule entry (epsilon)
    RuleEnd,     // rule exit (epsilon)
    PredNot,     // !(expr) — zero-width negative lookahead
    PredAnd,     // &(expr) — zero-width positive lookahead
    // EBNF `A - B` subtraction: reject this cursor if the sub-graph at
    // pred_start matches the exact token span from stack.top().start_pos
    // to the current tok_pos.
    SubCheckNot,
};

struct Node {
    NodeType type{NodeType::Split};
    std::string value;       // MatchStr: string; MatchTok: type name; RuleRef: rule name
    std::string rule_name;   // RuleStart / RuleEnd: the rule's name
    std::uint32_t pred_start = 0;
    std::uint32_t rule_id = UINT32_MAX;  // for RuleRef — resolved at finalize()
    std::vector<std::uint32_t> links;
    // RuleRef only: `links` republished as a shared_ptr so that pushing a
    // stack frame is one refcount bump instead of a fresh make_shared — a
    // control block *and* a vector heap allocation *and* a copy — on every
    // single traversal (measured at ~19 per input token).  Built once by
    // finalize(); `links` must not change after that.
    //
    // Sharing it also makes dedup()'s prediction merge cheaper *and* more
    // accurate: two frames pushed from the same RuleRef now compare equal by
    // pointer, so the merge path no longer flattens both stacks and rebuilds
    // one just to arrive at the identical return-link set.
    std::shared_ptr<const std::vector<std::uint32_t>> links_shared;
};

struct Rule {
    std::uint32_t start = 0;
    std::uint32_t end = 0;
};

struct Graph {
    std::vector<Node> nodes;
    std::unordered_map<std::string, Rule> rules;
    std::unordered_map<std::string, std::string> lr_meta;
    std::unordered_set<std::string> skip_types;
    // Rule names whose matches produce no ParseNode in the output tree
    // — used for EBNF `A - B` anonymous `_sub_N` rules.
    std::unordered_set<std::string> stripped_names;
    std::string start_rule;

    // Fast-access arrays populated by finalize().  Indexed by rule_id.
    std::vector<std::uint32_t> rule_starts;
    std::vector<std::uint32_t> rule_ends;
    std::vector<std::string>   rule_names;
    std::vector<std::uint8_t>  rule_stripped;  // 1 = don't emit ParseNode
    std::uint32_t              start_rule_id = UINT32_MAX;
    bool                       finalized = false;

    std::uint32_t add_node(NodeType t) {
        nodes.push_back(Node{});
        nodes.back().type = t;
        return static_cast<std::uint32_t>(nodes.size() - 1);
    }

    std::pair<std::uint32_t, std::uint32_t> add_rule(const std::string& name) {
        std::uint32_t s = add_node(NodeType::RuleStart);
        std::uint32_t e = add_node(NodeType::RuleEnd);
        nodes[s].rule_name = name;
        nodes[e].rule_name = name;
        rules[name] = {s, e};
        return {s, e};
    }

    void finalize() {
        if (finalized) return;
        rule_starts.clear(); rule_ends.clear(); rule_names.clear();
        std::unordered_map<std::string, std::uint32_t> by_name;
        by_name.reserve(rules.size());
        for (const auto& kv : rules) {
            auto id = static_cast<std::uint32_t>(rule_starts.size());
            rule_starts.push_back(kv.second.start);
            rule_ends.push_back(kv.second.end);
            rule_names.push_back(kv.first);
            by_name[kv.first] = id;
        }
        for (auto& n : nodes) {
            if (n.type == NodeType::RuleRef) {
                auto it = by_name.find(n.value);
                n.rule_id = (it != by_name.end()) ? it->second : UINT32_MAX;
                n.links_shared = std::make_shared<
                    const std::vector<std::uint32_t>>(n.links);
            }
        }
        rule_stripped.assign(rule_starts.size(), 0);
        for (const auto& name : stripped_names) {
            auto it = by_name.find(name);
            if (it != by_name.end()) rule_stripped[it->second] = 1;
        }
        auto sit = by_name.find(start_rule);
        start_rule_id = (sit != by_name.end()) ? sit->second : UINT32_MAX;
        finalized = true;
    }
};

// ============================================================================
// Parse tree
// ============================================================================

struct ParseNode : gpda_pool::Refcounted<ParseNode> {
    std::string name;   // rule name (for rule matches) or token type (for terminals)
    std::string value;  // token value — set on terminal nodes only
    // Source position, copied off the matched token.  Terminals only: a rule
    // node's position is its first terminal descendant's, which the caller can
    // find with first_pos() rather than have every rule node carry a
    // redundant copy.  Any front-end built on this needs positions to report
    // semantic errors ("line 12: unknown property"), so the engine keeps them
    // instead of forcing callers to re-derive them from the token stream.
    std::uint32_t line = 0;
    std::uint32_t col  = 0;
    std::vector<gpda_pool::IntrusivePtr<ParseNode>> children;

    // Line/col of this subtree's first terminal ({0,0} if it has none).
    // Iterative for the same reason the destructor is: an LR-reconstructed
    // spine is O(N) deep, so recursion here could blow the stack on a long
    // expression.
    std::pair<std::uint32_t, std::uint32_t> first_pos() const {
        std::vector<const ParseNode*> stack{this};
        while (!stack.empty()) {
            const ParseNode* n = stack.back();
            stack.pop_back();
            if (n->children.empty()) {
                if (n->line) return {n->line, n->col};
                continue;
            }
            for (std::size_t i = n->children.size(); i-- > 0; )
                stack.push_back(n->children[i].get());
        }
        return {0, 0};
    }

    static void deallocate(ParseNode* p) noexcept {
        gpda_pool::Pool<ParseNode>::instance().destroy(p);
    }

    // Iterative destructor — prevents stack overflow on deep trees (e.g. the
    // LR-reconstructed spine of a long arithmetic expression, which is O(N)
    // deep).  We move uniquely-owned descendants into a worklist and empty
    // each one's children before its ref drops to zero, so every implicit
    // ~ParseNode call runs against an already-empty children vector.
    ~ParseNode() {
        if (children.empty()) return;
        std::vector<gpda_pool::IntrusivePtr<ParseNode>> worklist;
        worklist.reserve(children.size());
        for (auto& c : children) worklist.push_back(std::move(c));
        children.clear();
        while (!worklist.empty()) {
            auto n = std::move(worklist.back());
            worklist.pop_back();
            if (n.is_unique()) {
                for (auto& c : n->children) worklist.push_back(std::move(c));
                n->children.clear();
            }
            // n's IntrusivePtr releases here; if it hits zero refcount, the
            // pool destroys the node and re-enters ~ParseNode with an
            // already-empty children vector, which short-circuits above.
        }
    }

    std::string pretty(int indent = 0) const;
};
using ParseNodePtr = gpda_pool::IntrusivePtr<ParseNode>;

inline ParseNodePtr make_parse_node() {
    return ParseNodePtr(gpda_pool::Pool<ParseNode>::instance().make());
}

// ============================================================================
// Cursor
// ============================================================================

struct StackEntry {
    std::uint32_t rule_id;
    std::shared_ptr<const std::vector<std::uint32_t>> return_links;
    PListPtr<ParseNodePtr> parent_children;            // reverse insertion order
    std::uint32_t start_pos = 0;                       // tok_pos on rule entry
    std::size_t merge_hash = 0;                        // cumulative (rule_id, start_pos) hash
};

struct Cursor {
    std::uint32_t node;
    PListPtr<StackEntry> stack;
    PListPtr<ParseNodePtr> children;  // reverse insertion order
};

// Ordered-choice disambiguation is implicit in the parser's depth-first
// link-order traversal — see scannerless.hpp for the rationale.

// ============================================================================
// Parser
// ============================================================================

class Parser {
public:
    Graph graph;
    std::size_t max_depth = 200;

    // Parse a token list.  Throws ParseError (a std::runtime_error) on failure,
    // carrying the position, the expected-terminal set and the rule context.
    // Any 'EOF'-typed sentinel at the tail is ignored.
    ParseNodePtr parse(const std::vector<Token>& tokens);

private:
    const std::vector<Token>* tokens_ = nullptr;
    int pred_depth_ = 0;

    // State key for visited / dedup sets.  Owns an IntrusivePtr so the
    // pool can't recycle the underlying PList slot while we still hold
    // its address in the visited set — see scannerless.hpp for details.
    struct StateKey {
        std::uint32_t node_id;
        PListPtr<StackEntry> stack;
    };

    // The epsilon-closure's visited set — probed once per closure step, which
    // makes it the single hottest data structure in the parser.  It is an
    // open-addressed hash table rather than the linear scan it used to be:
    // closures routinely reach 50+ states, and scanning makes the walk
    // quadratic in the closure size.  Measured on a 22 KB input: ~2000 key
    // comparisons per input token before, ~23 after.
    //
    // `items_` still holds the StateKeys in insertion order because each one
    // owns an IntrusivePtr that must keep the PList slot alive for as long as
    // its address serves as a hash key; `slots_` holds 1-based indices into it
    // (0 meaning "empty"), so growing `items_` never invalidates the table.
    class Visited {
        std::vector<StateKey> items_;
        std::vector<std::uint32_t> slots_;
        std::size_t mask_ = 0;

        static std::size_t hash_key(std::uint32_t node_id,
                                    const void* stack) noexcept {
            std::uint64_t h = static_cast<std::uint64_t>(
                                  reinterpret_cast<std::uintptr_t>(stack));
            h ^= static_cast<std::uint64_t>(node_id) * 0x9E3779B97F4A7C15ull;
            h *= 0xFF51AFD7ED558CCDull;
            h ^= h >> 32;
            return static_cast<std::size_t>(h);
        }

        void rehash(std::size_t new_cap) {
            slots_.assign(new_cap, 0u);
            mask_ = new_cap - 1;
            for (std::size_t i = 0; i < items_.size(); ++i) {
                std::size_t j = hash_key(items_[i].node_id,
                                         items_[i].stack.get()) & mask_;
                while (slots_[j]) j = (j + 1) & mask_;
                slots_[j] = static_cast<std::uint32_t>(i + 1);
            }
        }

    public:
        Visited() { items_.reserve(64); rehash(128); }

        void clear() noexcept {
            items_.clear();
            std::fill(slots_.begin(), slots_.end(), 0u);
        }

        // Takes the stack by reference and only copies it on an actual insert.
        // Most probes are duplicates — that is the point of a visited set — and
        // a StateKey copy costs a refcount RMW on the PList, so building one
        // eagerly made the common path pay for the rare one.
        bool insert(std::uint32_t node_id,
                    const PListPtr<StackEntry>& stack) noexcept {
            const PList<StackEntry>* raw = stack.get();
            // Keep the load factor below 1/2 so probe runs stay short.
            if ((items_.size() + 1) * 2 > slots_.size()) {
                rehash(slots_.size() * 2);
            }
            std::size_t j = hash_key(node_id, raw) & mask_;
            while (std::uint32_t s = slots_[j]) {
                const StateKey& x = items_[s - 1];
                if (x.node_id == node_id && x.stack.get() == raw) return false;
                j = (j + 1) & mask_;
            }
            items_.push_back(StateKey{node_id, stack});
            slots_[j] = static_cast<std::uint32_t>(items_.size());
            return true;
        }
    };

    // Pool of Visited objects, reused across expand_all / find_completions
    // iterations (avoids per-call 32-slot vector allocation).  Used as a
    // stack: acquire on entry, release on exit; predicates that recurse
    // get their own from deeper in the stack.
    //
    // Held by pointer, NOT by value: a caller keeps its `Visited&` alive
    // across expand(), which for a predicate node recurses back into
    // expand_all() and acquires another one.  With a vector-of-values that
    // acquire can reallocate and leave every outer frame holding a reference
    // into freed storage — the StateKeys they then push retain PList cursors
    // in a container that is never destroyed, so those cursors leak (and the
    // write itself is a use-after-free).  unique_ptr elements keep the
    // Visited objects at stable addresses.
    std::vector<std::unique_ptr<Visited>> visited_pool_;
    std::size_t visited_in_use_ = 0;

    Visited& acquire_visited() {
        if (visited_in_use_ >= visited_pool_.size()) {
            visited_pool_.push_back(std::make_unique<Visited>());
        }
        auto& v = *visited_pool_[visited_in_use_++];
        v.clear();
        return v;
    }
    void release_visited() { --visited_in_use_; }

    // Drop every pool-allocated reference this parse accumulated.  `Visited`
    // only clears itself on *acquire*, so after the last release each entry
    // still owns the PListPtr cursors it saw — and a Parser routinely outlives
    // a parse (it is commonly a member, or a function-local static).  Holding
    // them pins the whole cursor-stack graph between parses, and at process
    // exit they get torn down after the pool that owns them.  Also drops
    // `tokens_`, which points at a vector local to parse().
    void reset_scratch() noexcept {
        for (auto& v : visited_pool_) v->clear();
        visited_in_use_ = 0;
        tokens_ = nullptr;
    }

    // Runs reset_scratch() on the way out of parse(), thrown-from included.
    struct ScratchGuard {
        Parser* p;
        ~ScratchGuard() { p->reset_scratch(); }
    };

    std::vector<Cursor> expand_all(const std::vector<Cursor>& cursors,
                                   std::uint32_t tok_pos);
    void expand(std::uint32_t node,
                const PListPtr<StackEntry>& stack,
                const PListPtr<ParseNodePtr>& children,
                std::uint32_t tok_pos,
                std::vector<Cursor>& out,
                Visited& visited);

    bool evaluate_predicate(std::uint32_t pred_start, std::uint32_t tok_pos);

    // EBNF `A - B`: True iff the sub-graph at *pred_start* matches the
    // exact token span [start_pos, end_pos).  Used by SubCheckNot.
    bool evaluate_predicate_bounded(std::uint32_t pred_start,
                                    std::uint32_t start_pos,
                                    std::uint32_t end_pos);

    std::vector<ParseNodePtr> find_completions(
        const std::vector<Cursor>& cursors, std::uint32_t tok_pos);
    void find_completion(std::uint32_t node,
                         const PListPtr<StackEntry>& stack,
                         const PListPtr<ParseNodePtr>& children,
                         std::uint32_t tok_pos,
                         std::vector<ParseNodePtr>& out,
                         Visited& visited);

    ParseNodePtr reconstruct_lr(const ParseNodePtr& tree);

    bool token_matches(const Node& n, const Token& tok) const;
    std::vector<Cursor> dedup(const std::vector<Cursor>& cursors);

    // Build the ParseError for a dead cursor set.  `expanded` is the fully
    // epsilon-expanded set at the failure position (terminals only), so it is
    // exactly the set of continuations the grammar would have accepted; `tok`
    // is the offending token, or null when the input simply ran out.
    // `tok` is the offending token, or null at end of input — in which case
    // `last` (the final real token, if any) supplies the position to report.
    ParseError make_error(const std::vector<Cursor>& expanded,
                          const Token* tok, std::size_t pos,
                          const Token* last = nullptr) const;
};

// ============================================================================
// Inline helpers
// ============================================================================

inline bool Parser::token_matches(const Node& n, const Token& tok) const {
    switch (n.type) {
        case NodeType::MatchStr: return tok.value == n.value;
        case NodeType::MatchTok: return tok.type  == n.value;
        default: return false;
    }
}

}  // namespace gpda_tok
