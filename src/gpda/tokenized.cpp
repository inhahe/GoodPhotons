// tokenized.cpp — implementation of gpda_tok::Parser

#include "tokenized.hpp"

#include <algorithm>

namespace gpda_tok {

// ============================================================================
// ParseNode::pretty
// ============================================================================

std::string ParseNode::pretty(int indent) const {
    std::string prefix(indent * 2, ' ');
    std::ostringstream os;
    if (children.empty()) {
        // Terminal
        os << prefix << name << ": \"" << value << "\"";
    } else {
        os << prefix << name << ":";
        for (const auto& c : children) {
            os << "\n" << c->pretty(indent + 1);
        }
    }
    return os.str();
}

// ============================================================================
// expand_all / expand
// ============================================================================

std::vector<Cursor> Parser::expand_all(const std::vector<Cursor>& cursors,
                                       std::uint32_t tok_pos) {
    std::vector<Cursor> out;
    for (const auto& c : cursors) {
        Visited& visited = acquire_visited();
        expand(c.node, c.stack, c.children, tok_pos, out, visited);
        release_visited();
    }
    return out;
}

void Parser::expand(std::uint32_t initial_node,
                    const PListPtr<StackEntry>& initial_stack,
                    const PListPtr<ParseNodePtr>& initial_children,
                    std::uint32_t tok_pos,
                    std::vector<Cursor>& out,
                    Visited& visited) {
    struct Item {
        std::uint32_t node;
        PListPtr<StackEntry> stack;
        PListPtr<ParseNodePtr> children;
    };
    std::vector<Item> work;
    work.reserve(32);
    work.push_back({initial_node, initial_stack, initial_children});

    while (!work.empty()) {
        Item it = std::move(work.back());
        work.pop_back();

        while (true) {
            if (it.stack && it.stack->length > max_depth) break;

            auto key = state_key(it.node, it.stack);
            if (!visited.insert(std::move(key))) break;

            const Node& n = graph.nodes[it.node];

            if (n.type == NodeType::Split || n.type == NodeType::RuleStart) {
                if (n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            if (n.type == NodeType::RuleRef) {
                if (n.rule_id >= graph.rule_starts.size()) {
                    throw std::runtime_error("Unknown rule: " + n.value);
                }
                StackEntry entry;
                entry.rule_id = n.rule_id;
                entry.return_links = std::make_shared<
                    const std::vector<std::uint32_t>>(n.links);
                entry.parent_children = it.children;
                entry.start_pos = tok_pos;
                std::size_t prev = it.stack ? it.stack->head.merge_hash : 0;
                entry.merge_hash = prev * 2654435761u
                                   + (entry.rule_id * 65537u + entry.start_pos);
                it.stack = plist_push(it.stack, std::move(entry));
                it.children = nullptr;
                it.node = graph.rule_starts[n.rule_id];
                continue;
            }

            if (n.type == NodeType::RuleEnd) {
                if (!it.stack) break;
                const StackEntry& top = it.stack->head;
                PListPtr<ParseNodePtr> new_children;
                if (top.rule_id < graph.rule_stripped.size() &&
                        graph.rule_stripped[top.rule_id]) {
                    new_children = top.parent_children;
                } else {
                    auto rule_node = make_parse_node();
                    rule_node->name = graph.rule_names[top.rule_id];
                    rule_node->children = plist_to_vector(it.children);
                    new_children = plist_push(top.parent_children,
                                               rule_node);
                }

                auto new_stack = plist_pop(it.stack);
                if (top.return_links->empty()) break;
                for (std::size_t i = top.return_links->size(); i-- > 1; ) {
                    work.push_back({(*top.return_links)[i], new_stack,
                                    new_children});
                }
                it.node = (*top.return_links)[0];
                it.stack = std::move(new_stack);
                it.children = std::move(new_children);
                continue;
            }

            if (n.type == NodeType::PredNot || n.type == NodeType::PredAnd) {
                bool matched = evaluate_predicate(n.pred_start, tok_pos);
                bool passes = (n.type == NodeType::PredAnd && matched) ||
                              (n.type == NodeType::PredNot && !matched);
                if (!passes || n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            if (n.type == NodeType::SubCheckNot) {
                // EBNF A - B: stack.top().start_pos is A's start, tok_pos
                // is A's end.  Reject if B matches exactly that span.
                if (!it.stack) break;
                std::uint32_t sp = it.stack->head.start_pos;
                if (evaluate_predicate_bounded(n.pred_start, sp, tok_pos)) {
                    break;
                }
                if (n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            // Terminal
            out.push_back({it.node, it.stack, it.children});
            break;
        }
    }
}

// ============================================================================
// evaluate_predicate_bounded
// ============================================================================

bool Parser::evaluate_predicate_bounded(std::uint32_t pred_start,
                                        std::uint32_t start_pos,
                                        std::uint32_t end_pos) {
    if (++pred_depth_ > 50) { --pred_depth_; return false; }
    struct Scope { int* p; ~Scope() { --*p; } } scope{&pred_depth_};

    std::vector<Cursor> cursors = {{pred_start, nullptr, nullptr}};
    if (start_pos == end_pos) {
        return !find_completions(cursors, start_pos).empty();
    }
    const auto& toks = *tokens_;
    for (std::size_t i = start_pos; i < end_pos; ++i) {
        auto expanded = expand_all(cursors, static_cast<std::uint32_t>(i));
        std::vector<Cursor> next;
        for (const auto& e : expanded) {
            if (token_matches(graph.nodes[e.node], toks[i])) {
                for (std::uint32_t link : graph.nodes[e.node].links) {
                    next.push_back({link, e.stack, e.children});
                }
            }
        }
        if (next.empty()) return false;
        cursors = dedup(next);
    }
    return !find_completions(cursors, end_pos).empty();
}

// ============================================================================
// evaluate_predicate
// ============================================================================

bool Parser::evaluate_predicate(std::uint32_t pred_start,
                                std::uint32_t tok_pos) {
    if (++pred_depth_ > 50) { --pred_depth_; return false; }
    struct Scope { int* p; ~Scope() { --*p; } } scope{&pred_depth_};

    std::vector<Cursor> cursors = {{pred_start, nullptr, nullptr}};
    if (!find_completions(cursors, tok_pos).empty()) return true;

    const auto& toks = *tokens_;
    for (std::size_t i = tok_pos; i < toks.size(); ++i) {
        auto expanded = expand_all(cursors, static_cast<std::uint32_t>(i));
        std::vector<Cursor> next;
        for (const auto& e : expanded) {
            if (token_matches(graph.nodes[e.node], toks[i])) {
                for (std::uint32_t link : graph.nodes[e.node].links) {
                    next.push_back({link, e.stack, e.children});
                }
            }
        }
        // Predicates see skip tokens the same way the main loop does:
        // skip tokens are optionally-consumable, so keep original cursors
        // alongside matched ones.
        if (graph.skip_types.count(toks[i].type)) {
            for (const auto& c : cursors) next.push_back(c);
        } else if (next.empty()) {
            return false;
        }
        cursors = dedup(next);
        if (!find_completions(cursors,
                               static_cast<std::uint32_t>(i + 1)).empty())
            return true;
    }
    return !find_completions(cursors,
                              static_cast<std::uint32_t>(toks.size()))
                .empty();
}

// ============================================================================
// find_completions / find_completion
// ============================================================================

std::vector<ParseNodePtr> Parser::find_completions(
    const std::vector<Cursor>& cursors, std::uint32_t tok_pos) {
    std::vector<ParseNodePtr> results;
    for (const auto& c : cursors) {
        Visited& visited = acquire_visited();
        find_completion(c.node, c.stack, c.children, tok_pos,
                        results, visited);
        release_visited();
    }
    return results;
}

void Parser::find_completion(std::uint32_t initial_node,
                             const PListPtr<StackEntry>& initial_stack,
                             const PListPtr<ParseNodePtr>& initial_children,
                             std::uint32_t tok_pos,
                             std::vector<ParseNodePtr>& out,
                             Visited& visited) {
    struct Item {
        std::uint32_t node;
        PListPtr<StackEntry> stack;
        PListPtr<ParseNodePtr> children;
    };
    std::vector<Item> work;
    work.reserve(32);
    work.push_back({initial_node, initial_stack, initial_children});

    while (!work.empty()) {
        Item it = std::move(work.back());
        work.pop_back();

        while (true) {
            auto key = state_key(it.node, it.stack);
            if (!visited.insert(std::move(key))) break;

            const Node& n = graph.nodes[it.node];

            if (n.type == NodeType::Split || n.type == NodeType::RuleStart) {
                if (n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            if (n.type == NodeType::RuleRef) {
                if (n.rule_id >= graph.rule_starts.size()) break;
                StackEntry entry;
                entry.rule_id = n.rule_id;
                entry.return_links = std::make_shared<
                    const std::vector<std::uint32_t>>(n.links);
                entry.parent_children = it.children;
                entry.start_pos = tok_pos;
                std::size_t prev = it.stack ? it.stack->head.merge_hash : 0;
                entry.merge_hash = prev * 2654435761u
                                   + (entry.rule_id * 65537u + entry.start_pos);
                it.stack = plist_push(it.stack, std::move(entry));
                it.children = nullptr;
                it.node = graph.rule_starts[n.rule_id];
                continue;
            }

            if (n.type == NodeType::PredNot || n.type == NodeType::PredAnd) {
                bool matched = evaluate_predicate(n.pred_start, tok_pos);
                bool passes = (n.type == NodeType::PredAnd && matched) ||
                              (n.type == NodeType::PredNot && !matched);
                if (!passes || n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            if (n.type == NodeType::SubCheckNot) {
                if (!it.stack) break;
                std::uint32_t sp = it.stack->head.start_pos;
                if (evaluate_predicate_bounded(n.pred_start, sp, tok_pos)) {
                    break;
                }
                if (n.links.empty()) break;
                for (std::size_t i = n.links.size(); i-- > 1; ) {
                    work.push_back({n.links[i], it.stack, it.children});
                }
                it.node = n.links[0];
                continue;
            }

            if (n.type == NodeType::RuleEnd) {
                if (it.stack) {
                    const StackEntry& top = it.stack->head;
                    PListPtr<ParseNodePtr> new_children;
                    if (top.rule_id < graph.rule_stripped.size() &&
                            graph.rule_stripped[top.rule_id]) {
                        new_children = top.parent_children;
                    } else {
                        auto rule_node = make_parse_node();
                        rule_node->name = graph.rule_names[top.rule_id];
                        rule_node->children = plist_to_vector(it.children);
                        new_children = plist_push(top.parent_children,
                                                   rule_node);
                    }

                    auto new_stack = plist_pop(it.stack);
                    if (top.return_links->empty()) break;
                    for (std::size_t i = top.return_links->size(); i-- > 1; ) {
                        work.push_back({(*top.return_links)[i], new_stack,
                                        new_children});
                    }
                    it.node = (*top.return_links)[0];
                    it.stack = std::move(new_stack);
                    it.children = std::move(new_children);
                    continue;
                } else {
                    auto result = make_parse_node();
                    result->name = n.rule_name;
                    result->children = plist_to_vector(it.children);
                    out.push_back(result);
                    break;
                }
            }

            // MatchStr / MatchTok — dead end
            break;
        }
    }
}

// ============================================================================
// dedup
// ============================================================================

std::vector<Cursor> Parser::dedup(const std::vector<Cursor>& cursors) {
    // First-seen wins; combined with link-order traversal in expand(),
    // this implements ordered-choice disambiguation naturally.
    //
    // Prediction merging (Earley-style): see scannerless.cpp for the
    // full rationale.  The dedup key uses (node_id, stack_merge_hash)
    // which excludes return links; on key match we union the return-
    // link sets at every stack frame.
    struct MergeKey {
        std::uint32_t node_id;
        std::size_t   stack_hash;
        bool operator==(const MergeKey& o) const {
            return node_id == o.node_id && stack_hash == o.stack_hash;
        }
    };
    struct Entry { MergeKey key; std::size_t out_idx; };
    std::vector<Entry> seen;
    seen.reserve(cursors.size());

    std::vector<Cursor> out;
    out.reserve(cursors.size());

    for (const auto& c : cursors) {
        std::size_t sh = c.stack ? c.stack->head.merge_hash : 0;
        MergeKey mk{c.node, sh};

        bool found = false;
        for (auto& e : seen) {
            if (e.key == mk) {
                auto sa = out[e.out_idx].stack;
                auto sb = c.stack;
                if (sa.get() != sb.get()) {
                    std::vector<StackEntry> fa, fb;
                    for (auto p = sa.get(); p; p = p->tail.get())
                        fa.push_back(p->head);
                    for (auto p = sb.get(); p; p = p->tail.get())
                        fb.push_back(p->head);
                    if (fa.size() == fb.size()) {
                        bool any_diff = false;
                        for (std::size_t i = 0; i < fa.size(); ++i) {
                            if (fa[i].return_links.get()
                                    != fb[i].return_links.get()) {
                                any_diff = true;
                                auto merged = std::make_shared<
                                    std::vector<std::uint32_t>>(
                                        *fa[i].return_links);
                                for (auto id : *fb[i].return_links) {
                                    if (std::find(merged->begin(),
                                                  merged->end(), id)
                                            == merged->end()) {
                                        merged->push_back(id);
                                    }
                                }
                                fa[i].return_links = std::move(merged);
                            }
                        }
                        if (any_diff) {
                            PListPtr<StackEntry> ns;
                            for (std::size_t i = fa.size(); i-- > 0; ) {
                                ns = plist_push(ns, std::move(fa[i]));
                            }
                            out[e.out_idx].stack = std::move(ns);
                        }
                    }
                }
                found = true;
                break;
            }
        }
        if (!found) {
            seen.push_back({mk, out.size()});
            out.push_back(c);
        }
    }
    return out;
}

// ============================================================================
// reconstruct_lr
// ============================================================================

ParseNodePtr Parser::reconstruct_lr(const ParseNodePtr& tree) {
    if (!tree || tree->children.empty()) return tree;

    // Recurse; return the input tree unchanged if no child was rebuilt
    // and this rule isn't itself LR.  Eliminates per-parse allocations
    // for non-LR subtrees (most of the tree for arith-style grammars).
    std::vector<ParseNodePtr> new_children;
    new_children.reserve(tree->children.size());
    bool any_changed = false;
    for (const auto& c : tree->children) {
        auto nc = reconstruct_lr(c);
        if (nc.get() != c.get()) any_changed = true;
        new_children.push_back(std::move(nc));
    }

    auto it = graph.lr_meta.find(tree->name);
    if (it == graph.lr_meta.end()) {
        if (!any_changed) return tree;
        auto n = make_parse_node();
        n->name = tree->name;
        n->value = tree->value;
        n->children = std::move(new_children);
        return n;
    }

    const std::string& tail_name = it->second;
    std::vector<ParseNodePtr> base, tails;
    for (const auto& c : new_children) {
        if (c->name == tail_name) tails.push_back(c);
        else                       base.push_back(c);
    }

    if (tails.empty()) {
        auto n = make_parse_node();
        n->name = tree->name;
        n->children = std::move(base);
        return n;
    }

    auto cur = make_parse_node();
    cur->name = tree->name;
    cur->children = base;
    for (const auto& t : tails) {
        auto nxt = make_parse_node();
        nxt->name = tree->name;
        nxt->children.reserve(1 + t->children.size());
        nxt->children.push_back(cur);
        for (const auto& tc : t->children) nxt->children.push_back(tc);
        cur = nxt;
    }
    return cur;
}

// ============================================================================
// parse — main entry point
// ============================================================================

ParseNodePtr Parser::parse(const std::vector<Token>& tokens) {
    // Filter out EOF sentinels (match Python behavior)
    std::vector<Token> real;
    real.reserve(tokens.size());
    for (const auto& t : tokens) {
        if (t.type != "EOF") real.push_back(t);
    }
    tokens_ = &real;

    graph.finalize();
    if (graph.start_rule_id >= graph.rule_starts.size()) {
        throw std::runtime_error("No start rule: " + graph.start_rule);
    }
    std::uint32_t start_node = graph.rule_starts[graph.start_rule_id];

    std::vector<Cursor> cursors;
    cursors.push_back({start_node, nullptr, nullptr});

    std::size_t n = real.size();
    for (std::size_t tok_pos = 0; tok_pos < n; ++tok_pos) {
        const Token& tok = real[tok_pos];
        auto expanded = expand_all(cursors,
                                    static_cast<std::uint32_t>(tok_pos));

        std::vector<Cursor> next;
        for (const auto& e : expanded) {
            if (token_matches(graph.nodes[e.node], tok)) {
                auto leaf = make_parse_node();
                leaf->name  = tok.type;
                leaf->value = tok.value;
                auto new_children = plist_push(e.children, leaf);
                for (std::uint32_t link : graph.nodes[e.node].links) {
                    next.push_back({link, e.stack, new_children});
                }
            }
        }

        if (graph.skip_types.count(tok.type)) {
            // Skip tokens are optional — keep the "was-ignored" cursors too.
            for (const auto& c : cursors) next.push_back(c);
        } else if (next.empty()) {
            std::ostringstream os;
            os << "Unexpected " << tok.type << " '" << tok.value
               << "' at line " << tok.line << ", col " << tok.col;
            throw std::runtime_error(os.str());
        }

        cursors = dedup(next);
    }

    auto completions = find_completions(cursors,
                                         static_cast<std::uint32_t>(n));
    if (completions.empty()) {
        throw std::runtime_error("Parse failed: unexpected end of input");
    }

    ParseNodePtr tree = completions[0];
    if (!graph.lr_meta.empty()) tree = reconstruct_lr(tree);
    return tree;
}

}  // namespace gpda_tok
