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
    std::vector<gpda_pool::IntrusivePtr<ParseNode>> children;

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

    // Parse a token list.  Throws std::runtime_error on failure.
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

    class Visited {
        std::vector<StateKey> items_;
    public:
        Visited() { items_.reserve(32); }
        void clear() noexcept { items_.clear(); }
        bool insert(StateKey k) noexcept {
            for (const auto& x : items_) {
                if (x.node_id == k.node_id
                        && x.stack.get() == k.stack.get()) return false;
            }
            items_.push_back(std::move(k));
            return true;
        }
    };

    // Pool of Visited objects, reused across expand_all / find_completions
    // iterations (avoids per-call 32-slot vector allocation).  Used as a
    // stack: acquire on entry, release on exit; predicates that recurse
    // get their own from deeper in the stack.
    std::vector<Visited> visited_pool_;
    std::size_t visited_in_use_ = 0;

    Visited& acquire_visited() {
        if (visited_in_use_ >= visited_pool_.size()) {
            visited_pool_.emplace_back();
        }
        auto& v = visited_pool_[visited_in_use_++];
        v.clear();
        return v;
    }
    void release_visited() { --visited_in_use_; }

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

    StateKey state_key(std::uint32_t node,
                       const PListPtr<StackEntry>& stack) const {
        return {node, stack};
    }
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
