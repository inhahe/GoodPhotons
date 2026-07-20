"""
VENDORED — DO NOT EDIT BY HAND.
===============================
This is a pinned, verbatim copy of the GPDA tokenized graph parser, vendored
into loom so its ``.ftsl`` parsing has no external dependency.

    Upstream : D:\\visual studio projects\\GraphParser  (gpda.py)
    Commit   : 1ac4cbf9b0f50def4f0962156db247b96dc1ad69  (2026-04-14)
    sha256   : c376d1658c660479…  (first 16 hex of the pinned gpda.py)

The file is fully self-contained (only ``import re``).  To refresh, re-copy
``gpda.py`` from the upstream repo, re-apply this header, and bump the commit /
hash above.  Public entry points used by loom: ``load_grammar(text)`` ->
``GrammarParser``; ``.parse(text)`` -> ``ParseNode`` (``.name`` / ``.children``
/ ``.value``).

--- original module docstring follows ------------------------------------

gpda.py - GPDA Graph Parser
============================
Implementation of the GPDA parsing algorithm with zero lookahead/lookbehind.

Grammar rules are converted to a directed graph of nodes. The parser walks
the graph following ALL viable paths simultaneously, pruning paths that
don't match the current input token. No lookahead or lookbehind is needed.

Based on: https://exalumen.blog/2025/03/06/epeg-a-new-way-of-parsing/

Supports EPEG operators:
    |       alternation
    *       zero-or-more (greedy)
    *?      zero-or-more (non-greedy)
    +       one-or-more (greedy)
    +?      one-or-more (non-greedy)
    ?       optional (greedy)
    ??      optional (non-greedy)
    {n}     exact repetition
    {n,m}   bounded repetition
    &       AND predicate (zero-width: succeed if expr matches)
    !       NOT predicate (zero-width: succeed if expr does NOT match)
    name:=()  named capture (inline rule)
    ()      grouping

Convention:
    'text'      string literal  — matches token by value
    /regex/     inline regex    — auto-generates a lexer token
    name:=/re/  named inline token
    UPPERCASE   token reference — matches token by type name
    lowercase   rule reference  — expands into that rule's sub-graph

Whitespace and skip tokens:
    A token-shaped rule ending with `@skip` is ignored by default:

        _ws = /[ \\t]+/ @skip

    This makes whitespace invisible in most places.  But when the grammar
    explicitly references a skip token (e.g. `LET _ws IDENT`), it is
    required at that position — solving the classic "keyword must be
    followed by whitespace" problem.

Usage:
    from gpda import load_grammar

    # Self-contained grammar — auto-lexer built from inline /regex/
    p = load_grammar('''
        start expr
        _ws = /[ \\t]+/ @skip
        NUMBER = /[0-9]+/
        expr = NUMBER '+' NUMBER
    ''')
    tree = p.parse("12 + 34")

    # Or pass pre-lexed tokens (when you have a custom lexer):
    tree = p.parse(my_lexer.tokenize(source))

Left recursion:
    Both direct and indirect (mutual) left recursion are supported.

    Direct:   expr = expr '+' term | term
    Indirect: A = B 'x' | 'y'
              B = A 'z' | 'w'

    Left-recursive rules are detected and rewritten at build time
    (Paull's algorithm for mutual cycles, then direct transformation).
    Parse trees are reconstructed into the intended left-associative
    shape automatically.
"""

import re


# ========================== Lexer ==========================

class Token:
    """A lexer token with type, value, and source position."""
    __slots__ = ('type', 'value', 'line', 'col')

    def __init__(self, type, value, line=0, col=0):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


class Lexer:
    """Simple regex-based lexer.  Rules are tried in order; first match wins.

    rules:  list of (token_name, regex_pattern) pairs
    ignore: set of token names to discard (e.g. whitespace)
    """

    def __init__(self, rules, ignore=None):
        self.rules = [(name, re.compile(pattern)) for name, pattern in rules]
        self.ignore = ignore or set()

    def tokenize(self, text):
        """Tokenize with longest-match semantics.  Ties are broken by
        declaration order (earlier rule wins)."""
        tokens = []
        pos = 0
        line = 1
        col = 1
        while pos < len(text):
            best_name = None
            best_len = 0
            best_value = None
            for name, pattern in self.rules:
                m = pattern.match(text, pos)
                if m:
                    length = m.end() - pos
                    if length > best_len:
                        best_name = name
                        best_len = length
                        best_value = m.group()
            if best_name is None:
                raise SyntaxError(
                    f"Unexpected character {text[pos]!r} "
                    f"at line {line}, col {col}")
            if best_name not in self.ignore:
                tokens.append(Token(best_name, best_value, line, col))
            for ch in best_value:
                if ch == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
            pos += best_len
        tokens.append(Token('EOF', '', line, col))
        return tokens


# ========================== Graph Nodes ==========================

class NT:
    """Node types for the grammar graph."""
    MATCH_STR  = 1   # Match a token by its value
    MATCH_TOK  = 2   # Match a token by its type name
    RULE_REF   = 3   # Reference to another rule (epsilon transition)
    SPLIT      = 4   # Fan-out / join point (epsilon transition)
    RULE_START = 5   # Entry point of a rule (epsilon transition)
    RULE_END   = 6   # Exit point of a rule (epsilon transition)
    PRED_NOT   = 7   # !(expr) — succeed if expr does NOT match (zero-width)
    PRED_AND   = 8   # &(expr) — succeed if expr DOES match (zero-width)
    # EBNF `-` subtraction: reject this cursor if .value's sub-graph
    # matches the token span from stack[-1].start_pos to the current
    # tok_pos exactly.  Used as the second step of an anonymous stripped
    # rule `_sub_N = A <SUB_CHECK_NOT(B)>` that wraps `A - B`.
    SUB_CHECK_NOT = 9
    ACTION     = 10  # semantic action ({{ ... }}); node.value is Python code


_ACTION_GLOBALS = {'__builtins__': __builtins__}


class Node:
    """A node in the grammar graph."""
    _next_id = 0

    __slots__ = ('id', 'type', 'value', 'links', 'rule')

    def __init__(self, ntype, value=None, rule=None):
        self.id = Node._next_id
        Node._next_id += 1
        self.type = ntype
        self.value = value   # string for MATCH_STR, token name for MATCH_TOK,
                             # rule name for RULE_REF
        self.links = []      # ordered list of next nodes (order = priority)
        self.rule = rule     # rule name for RULE_START / RULE_END

    def __repr__(self):
        names = {NT.MATCH_STR: 'STR', NT.MATCH_TOK: 'TOK', NT.RULE_REF: 'REF',
                 NT.SPLIT: 'SPLIT', NT.RULE_START: 'START', NT.RULE_END: 'END',
                 NT.PRED_NOT: '!', NT.PRED_AND: '&'}
        return f"Node({self.id}, {names.get(self.type, '?')}, {self.value!r})"


# ========================== Parse Tree ==========================

class ParseNode:
    """A node in the output parse tree."""
    __slots__ = ('name', 'children', 'value')

    def __init__(self, name, children=None, value=None):
        self.name = name
        self.children = children or []
        self.value = value  # non-None only for terminal (leaf) nodes

    def __repr__(self):
        if self.value is not None:
            return f"'{self.value}'"
        kids = ', '.join(repr(c) for c in self.children)
        return f"{self.name}({kids})"

    def pretty(self, indent=0):
        prefix = '  ' * indent
        if self.value is not None:
            return f"{prefix}{self.name}: {self.value!r}"
        lines = [f"{prefix}{self.name}:"]
        for child in self.children:
            lines.append(child.pretty(indent + 1))
        return '\n'.join(lines)


# ========================== Left-Recursion Tree Reconstruction ==========================

def _reconstruct_lr(tree, lr_meta):
    """Rebuild a left-associative tree from a transformed (right-recursive) parse.

    The grammar transformation turns
        ``R = R α | β``  into  ``R = β _R_lr_tail*`` / ``_R_lr_tail = α``

    which produces a flat tree:  R([β, _tail(α₁), _tail(α₂), ...])

    This function folds it back into the intended left-recursive shape:
        R([R([R([β]), α₁]), α₂])
    """
    if not tree.children:
        return tree  # terminal / empty rule — nothing to do

    # Recurse.  Track whether any child actually changed; if nothing did
    # and we're not an LR rule, return the input unchanged to avoid
    # allocating a copy for subtrees that contain no LR rule.
    any_changed = False
    children = []
    for c in tree.children:
        nc = _reconstruct_lr(c, lr_meta)
        if nc is not c:
            any_changed = True
        children.append(nc)

    if tree.name not in lr_meta:
        if not any_changed:
            return tree
        return ParseNode(tree.name, children, value=tree.value)

    tail_name = lr_meta[tree.name]

    # Partition children into base elements and tail elements
    base = []
    tails = []
    for child in children:
        if child.name == tail_name:
            tails.append(child)
        else:
            base.append(child)

    if not tails:
        return ParseNode(tree.name, base, value=tree.value)

    # Fold left: wrap the base, then each tail
    current = ParseNode(tree.name, list(base))
    for tail in tails:
        current = ParseNode(tree.name, [current] + list(tail.children))

    return current


# ========================== Precedence Rewriting ==========================

def _apply_precedence(rules, precedence):
    """Rewrite rules matching the precedence-ladder pattern.

    For each rule ``R`` whose alternatives include binary patterns of
    shape ``[ref_to_R, str_op, ref_to_R]`` where ``str_op`` is in the
    declared precedence table, generate a ladder of rules — one per
    precedence level — so the grammar becomes unambiguous and dedup
    can collapse cursors linearly.

    Associativity:
        left:     ``R_k = R_k OPS R_{k+1} | R_{k+1}`` (LR elim handles)
        right:    ``R_k = R_{k+1} OPS R_k | R_{k+1}``
        nonassoc: ``R_k = R_{k+1} (OPS R_{k+1})?``
    """
    if not precedence:
        return rules

    # op_info key: ('string', '+') or ('name', 'PLUS').
    op_info = {}
    for level_index, (assoc, ops) in enumerate(precedence):
        for op_kind, op_value in ops:
            key = (op_kind, op_value)
            if key in op_info:
                raise SyntaxError(
                    f"operator {op_value!r} listed in multiple "
                    f"precedence declarations")
            op_info[key] = (level_index, assoc)

    def op_key(node):
        t = node.get('type')
        v = node.get('value')
        if t == 'string':
            return ('string', v)
        if t in ('rule', 'token'):
            return ('name', v)
        return None

    out = {}
    for name, alts in rules.items():
        binary_alts = []
        atom_alts = []
        for alt in alts:
            ok = (len(alt) == 3
                  and isinstance(alt[0], dict)
                  and alt[0].get('type') == 'rule'
                  and alt[0].get('value') == name
                  and 'modifier' not in alt[0]
                  and isinstance(alt[1], dict)
                  and 'modifier' not in alt[1]
                  and isinstance(alt[2], dict)
                  and alt[2].get('type') == 'rule'
                  and alt[2].get('value') == name
                  and 'modifier' not in alt[2])
            key = op_key(alt[1]) if ok else None
            if ok and key is not None and key in op_info:
                level, assoc = op_info[key]
                binary_alts.append((level, assoc, alt[1]))
            else:
                atom_alts.append(alt)

        if not binary_alts:
            out[name] = alts
            continue

        by_level = {}
        assoc_by_level = {}
        for level, assoc, op_node in binary_alts:
            by_level.setdefault(level, []).append(op_node)
            if level in assoc_by_level and assoc_by_level[level] != assoc:
                raise SyntaxError(
                    f"rule {name!r}: operators at the same precedence "
                    f"level {level} have conflicting associativity")
            assoc_by_level[level] = assoc

        sorted_levels = sorted(by_level.keys())
        if not atom_alts:
            raise SyntaxError(
                f"rule {name!r}: precedence ladder has no atom "
                f"alternatives")

        atom_name = f"_{name}_atom"
        level_names = {}
        for i, lvl in enumerate(sorted_levels):
            level_names[lvl] = name if i == 0 else f"_{name}_p{lvl}"
        def higher_of(i):
            return level_names[sorted_levels[i + 1]] \
                if i + 1 < len(sorted_levels) else atom_name

        for i, lvl in enumerate(sorted_levels):
            this_name = level_names[lvl]
            higher = higher_of(i)
            op_nodes = by_level[lvl]
            assoc = assoc_by_level[lvl]
            # Reuse original op nodes verbatim so 'string' / 'token' /
            # 'rule' types are preserved in the rewritten grammar.
            op_group = {'type': 'group',
                         'alternatives': [[dict(op_node)]
                                          for op_node in op_nodes]}
            if assoc == 'left':
                new_alts = [
                    [{'type': 'rule', 'value': this_name},
                     op_group,
                     {'type': 'rule', 'value': higher}],
                    [{'type': 'rule', 'value': higher}],
                ]
            elif assoc == 'right':
                new_alts = [
                    [{'type': 'rule', 'value': higher},
                     op_group,
                     {'type': 'rule', 'value': this_name}],
                    [{'type': 'rule', 'value': higher}],
                ]
            else:  # nonassoc
                new_alts = [
                    [{'type': 'rule', 'value': higher},
                     {'type': 'group',
                      'alternatives': [[op_group,
                                        {'type': 'rule', 'value': higher}]],
                      'modifier': '?'}],
                ]
            out[this_name] = new_alts
        out[atom_name] = atom_alts

    return out


# ========================== Graph Builder ==========================

class GraphBuilder:
    """Converts a grammar dictionary into a graph of Nodes.

    Expected grammar format:
        {
            'start': 'rule_name',
            'rules': {
                'rule_name': [ [elem, elem, ...],   # alternative 1
                               [elem, elem, ...] ], # alternative 2
                ...
            }
        }

    Each element is a dict:
        {'type': 'string', 'value': '...'}
        {'type': 'token',  'value': 'TOKEN_NAME'}
        {'type': 'rule',   'value': 'rule_name'}
        {'type': 'group',  'alternatives': [[elems], ...]}
    with an optional 'modifier' key: '*', '*?', '+', '+?', '?', '??',
    or {'min': n, 'max': m, 'greedy': bool}.
    """

    def __init__(self):
        self.rules = {}       # rule_name -> (start_node, end_node)
        self.start_rule = None
        self.stripped_names = set()  # rules whose matches are invisible
        self.capture_names = set()   # rule names that act as captures
        self._sub_counter = 0        # for EBNF `A - B` anon rule names

    def build(self, grammar):
        """Build the graph.  Returns (rules_dict, start_rule_name, lr_meta, stripped_names, capture_names)."""
        self.start_rule = grammar['start']
        rules_dict = grammar['rules']

        # Apply precedence declarations: rewrites ambiguous binary-op
        # rules into a precedence ladder so dedup can collapse cursors
        # linearly instead of paying Catalan-exponential exploration.
        precedence = grammar.get('precedence', [])
        if precedence:
            rules_dict = _apply_precedence(rules_dict, precedence)

        # Scan for capture names (for action scope building).
        for name, alts in rules_dict.items():
            for alt in alts:
                self._collect_captures(alt, self.capture_names)

        # Phase 1: eliminate indirect (mutual) left recursion
        # Uses Paull's algorithm: substitute rules within each cycle until
        # only direct left recursion remains, then transform that too.
        rules_dict, lr_meta_indirect = self._eliminate_indirect_lr(rules_dict)

        # Phase 2: detect and transform any remaining direct left recursion
        # (rules that were NOT part of a multi-rule cycle)
        lr_rules = self._detect_direct_lr(rules_dict)
        if lr_rules:
            rules_dict, lr_meta_direct = self._transform_lr(rules_dict,
                                                            lr_rules)
        else:
            lr_meta_direct = {}

        self.lr_meta = {**lr_meta_indirect, **lr_meta_direct}

        # First pass: create entry/exit sentinel nodes for every rule
        for name in rules_dict:
            start = Node(NT.RULE_START, rule=name)
            end = Node(NT.RULE_END, rule=name)
            self.rules[name] = (start, end)

        # Second pass: fill in the internal graph for each rule
        for name, alternatives in rules_dict.items():
            start, end = self.rules[name]
            for alt in alternatives:
                chain_s, chain_e = self._build_seq(alt)
                start.links.append(chain_s)
                chain_e.links.append(end)

        return (self.rules, self.start_rule, self.lr_meta,
                self.stripped_names, self.capture_names)

    @staticmethod
    def _collect_captures(elements, result):
        """Recursively find capture names in a list of elements."""
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            t = elem.get('type')
            if t == 'capture':
                result.add(elem['name'])
                for alt in elem.get('alternatives', []):
                    GraphBuilder._collect_captures(alt, result)
            elif t in ('group', 'and', 'not'):
                for alt in elem.get('alternatives', []):
                    GraphBuilder._collect_captures(alt, result)
            elif t == 'subtract':
                GraphBuilder._collect_captures([elem['left']], result)
                GraphBuilder._collect_captures([elem['right']], result)

    # ---- left recursion: core helpers --------------------------------

    @staticmethod
    def _is_lr_alt(alt, name):
        """True if *alt* starts with a bare rule reference to *name*."""
        return (alt
                and alt[0].get('type') == 'rule'
                and alt[0].get('value') == name
                and 'modifier' not in alt[0])

    @staticmethod
    def _detect_direct_lr(rules):
        """Find rules whose first alternative starts with a self-reference."""
        lr = set()
        for name, alts in rules.items():
            for alt in alts:
                if GraphBuilder._is_lr_alt(alt, name):
                    lr.add(name)
                    break
        return lr

    @staticmethod
    def _transform_single_lr(alts, name):
        """Transform one directly left-recursive rule.

        ``R = R α₁ | R α₂ | β₁ | β₂``  →
        ``R = β₁ _R_lr_tail* | β₂ _R_lr_tail*``
        ``_R_lr_tail = α₁ | α₂``

        Returns ``(new_alts, tail_name, tail_alts)`` or ``None``.
        """
        import copy
        lr_alts = []
        base_alts = []
        for alt in alts:
            if GraphBuilder._is_lr_alt(alt, name):
                tail = alt[1:]
                if tail:
                    lr_alts.append(copy.deepcopy(tail))
            else:
                base_alts.append(copy.deepcopy(alt))

        if not lr_alts or not base_alts:
            return None

        tail_name = f'_{name}_lr_tail'
        tail_ref = {'type': 'rule', 'value': tail_name, 'modifier': '*'}
        new_alts = [base + [dict(tail_ref)] for base in base_alts]
        return new_alts, tail_name, lr_alts

    @staticmethod
    def _transform_lr(rules, lr_rules):
        """Transform all directly left-recursive rules in *rules*."""
        new_rules = dict(rules)
        lr_meta = {}
        for name in lr_rules:
            result = GraphBuilder._transform_single_lr(new_rules[name], name)
            if result is None:
                continue
            new_alts, tail_name, tail_alts = result
            new_rules[name] = new_alts
            new_rules[tail_name] = tail_alts
            lr_meta[name] = tail_name
        return new_rules, lr_meta

    # ---- left recursion: indirect (mutual) ----------------------------

    @staticmethod
    def _left_corner_graph(rules):
        """Directed graph: edge R→S when an alternative of R starts with
        a bare reference to grammar rule S."""
        graph = {name: set() for name in rules}
        for name, alts in rules.items():
            for alt in alts:
                if (alt
                        and alt[0].get('type') == 'rule'
                        and 'modifier' not in alt[0]
                        and alt[0]['value'] in rules):
                    graph[name].add(alt[0]['value'])
        return graph

    @staticmethod
    def _find_sccs(graph):
        """Tarjan's algorithm.  Returns a list of strongly-connected
        components (each a list of node names)."""
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = set()
        sccs = []

        def _visit(v):
            index[v] = lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph.get(v, ()):
                if w not in index:
                    _visit(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in graph:
            if v not in index:
                _visit(v)
        return sccs

    @staticmethod
    def _substitute_rule(alts, target_name, target_alts):
        """In *alts*, replace every alternative whose first element is a
        bare reference to *target_name* by inlining *target_alts*.

        ``[target_name rest…]``  →  ``[tₐ₁… rest…], [tₐ₂… rest…], …``
        """
        import copy
        out = []
        for alt in alts:
            if GraphBuilder._is_lr_alt(alt, target_name):
                rest = alt[1:]
                for ta in target_alts:
                    out.append(copy.deepcopy(ta) + copy.deepcopy(rest))
            else:
                out.append(alt)
        return out

    @staticmethod
    def _eliminate_indirect_lr(rules):
        """Eliminate indirect (mutual) left recursion using Paull's
        algorithm.

        For each strongly-connected component of the left-corner graph
        that contains more than one rule:

        1.  Order the rules.
        2.  For each rule Rᵢ, substitute all lower-indexed rules Rⱼ
            (j < i) whose alternatives appear as the leftmost symbol.
        3.  Eliminate any direct left recursion created by step 2.

        After processing, no mutual cycles remain.  Any remaining direct
        left recursion (in rules outside a multi-rule SCC) is left for
        ``_detect_direct_lr`` / ``_transform_lr``.

        Returns ``(new_rules, lr_meta)``.
        """
        graph = GraphBuilder._left_corner_graph(rules)
        sccs = GraphBuilder._find_sccs(graph)

        result = dict(rules)
        lr_meta = {}
        rule_order = list(rules.keys())

        for scc in sccs:
            if len(scc) <= 1:
                continue

            # Preserve the author's rule ordering within the SCC
            scc.sort(key=lambda n: rule_order.index(n)
                     if n in rule_order else len(rule_order))

            for i, ri in enumerate(scc):
                # Step 1: substitute all lower-indexed SCC members
                for j in range(i):
                    rj = scc[j]
                    result[ri] = GraphBuilder._substitute_rule(
                        result[ri], rj, result[rj])

                # Step 2: eliminate any direct LR that the substitution
                # created — this is essential for maintaining Paull's
                # invariant (Rᵢ never starts with Rₖ where k ≤ i).
                xform = GraphBuilder._transform_single_lr(result[ri], ri)
                if xform:
                    new_alts, tail_name, tail_alts = xform
                    result[ri] = new_alts
                    result[tail_name] = tail_alts
                    lr_meta[ri] = tail_name

        return result, lr_meta

    # ---- sequence / element / atom ----------------------------------

    def _build_seq(self, elements):
        """Build a chain for a sequence of elements."""
        if not elements:
            n = Node(NT.SPLIT)
            return n, n
        first_s, first_e = self._build_elem(elements[0])
        prev_e = first_e
        for elem in elements[1:]:
            s, e = self._build_elem(elem)
            prev_e.links.append(s)
            prev_e = e
        return first_s, prev_e

    def _build_elem(self, elem):
        """Build nodes for one element (atom + optional modifier)."""
        atom_s, atom_e = self._build_atom(elem)
        mod = elem.get('modifier')
        if not mod:
            return atom_s, atom_e
        if mod == '*':
            return self._star(atom_s, atom_e, greedy=True)
        if mod == '*?':
            return self._star(atom_s, atom_e, greedy=False)
        if mod == '+':
            return self._plus(atom_s, atom_e, greedy=True)
        if mod == '+?':
            return self._plus(atom_s, atom_e, greedy=False)
        if mod == '?':
            return self._opt(atom_s, atom_e, greedy=True)
        if mod == '??':
            return self._opt(atom_s, atom_e, greedy=False)
        if isinstance(mod, dict):
            return self._repeat(atom_s, atom_e, mod, elem)
        return atom_s, atom_e

    def _build_atom(self, elem):
        """Build node(s) for the atom part of an element (ignoring modifier)."""
        t = elem['type']
        if t == 'string':
            n = Node(NT.MATCH_STR, elem['value'])
            return n, n
        if t == 'token':
            n = Node(NT.MATCH_TOK, elem['value'])
            return n, n
        if t == 'rule':
            n = Node(NT.RULE_REF, elem['value'])
            return n, n
        if t == 'group':
            return self._build_alts(elem['alternatives'])
        if t == 'capture':
            name = elem['name']
            start = Node(NT.RULE_START, rule=name)
            end = Node(NT.RULE_END, rule=name)
            inner_s, inner_e = self._build_alts(elem['alternatives'])
            start.links.append(inner_s)
            inner_e.links.append(end)
            self.rules[name] = (start, end)
            ref = Node(NT.RULE_REF, name)
            return ref, ref
        if t in ('and', 'not'):
            pred_start = Node(NT.RULE_START, rule='_pred')
            pred_end = Node(NT.RULE_END, rule='_pred')
            inner_s, inner_e = self._build_alts(elem['alternatives'])
            pred_start.links.append(inner_s)
            inner_e.links.append(pred_end)
            ptype = NT.PRED_AND if t == 'and' else NT.PRED_NOT
            pred = Node(ptype, value=pred_start)
            return pred, pred
        if t == 'subtract':
            # EBNF `A - B`: match A, reject if B also matches the same
            # token span.  Wrap in anonymous stripped rule
            #   _sub_N = A  <SUB_CHECK_NOT(B)>
            # so stack[-1].start_pos (= when the anon rule was entered)
            # is A's start.  SUB_CHECK_NOT runs B as a bounded sub-parse
            # from start_pos to tok_pos.
            anon = f'_sub_{self._sub_counter}'
            self._sub_counter += 1
            rs = Node(NT.RULE_START, rule=anon)
            re_ = Node(NT.RULE_END, rule=anon)
            self.rules[anon] = (rs, re_)
            self.stripped_names.add(anon)

            a_s, a_e = self._build_elem(elem['left'])

            b_pred_s = Node(NT.RULE_START, rule='_sub_pred')
            b_pred_e = Node(NT.RULE_END, rule='_sub_pred')
            b_inner_s, b_inner_e = self._build_elem(elem['right'])
            b_pred_s.links.append(b_inner_s)
            b_inner_e.links.append(b_pred_e)

            check = Node(NT.SUB_CHECK_NOT, value=b_pred_s)
            rs.links.append(a_s)
            a_e.links.append(check)
            check.links.append(re_)

            ref = Node(NT.RULE_REF, anon)
            return ref, ref
        if t == 'action':
            n = Node(NT.ACTION, elem['code'])
            return n, n
        raise ValueError(f"Unknown element type: {t}")

    def _build_alts(self, alternatives):
        """Build a fan-out / fan-in structure for alternatives."""
        if len(alternatives) == 1:
            return self._build_seq(alternatives[0])
        split = Node(NT.SPLIT)
        join = Node(NT.SPLIT)
        for alt in alternatives:
            s, e = self._build_seq(alt)
            split.links.append(s)
            e.links.append(join)
        return split, join

    # ---- modifiers ---------------------------------------------------

    def _star(self, s, e, greedy):
        """Zero-or-more:  entry --[try/skip]--> atom/exit, loop back."""
        entry = Node(NT.SPLIT)
        exit_ = Node(NT.SPLIT)
        if greedy:
            entry.links = [s, exit_]
        else:
            entry.links = [exit_, s]
        e.links.append(entry)          # loop back after matching
        return entry, exit_

    def _plus(self, s, e, greedy):
        """One-or-more:  must match once, then optionally loop."""
        loop = Node(NT.SPLIT)
        exit_ = Node(NT.SPLIT)
        e.links.append(loop)
        if greedy:
            loop.links = [s, exit_]
        else:
            loop.links = [exit_, s]
        return s, exit_                # entry is the atom itself (mandatory)

    def _opt(self, s, e, greedy):
        """Optional:  entry --[try/skip]--> atom/exit."""
        entry = Node(NT.SPLIT)
        exit_ = Node(NT.SPLIT)
        if greedy:
            entry.links = [s, exit_]
        else:
            entry.links = [exit_, s]
        e.links.append(exit_)
        return entry, exit_

    def _repeat(self, s, e, mod, elem):
        """Bounded repetition {n}, {n,}, {n,m}, {,m}."""
        lo = mod.get('min', 0)
        hi = mod.get('max', lo)
        greedy = mod.get('greedy', True)

        # Build `lo` mandatory copies chained together
        parts = []
        for _ in range(lo):
            cs, ce = self._build_atom(elem)
            parts.append((cs, ce))
        # Chain mandatory copies
        for i in range(len(parts) - 1):
            parts[i][1].links.append(parts[i + 1][0])

        # Unbounded upper ({n,}): after the mandatory copies, a
        # Kleene-star of 0-or-more additional copies.
        if hi == -1:
            cs, ce = self._build_atom(elem)
            star_s, star_e = self._star(cs, ce, greedy)
            if parts:
                parts[-1][1].links.append(star_s)
                return parts[0][0], star_e
            return star_s, star_e

        # Build `hi - lo` optional copies
        opt_parts = []
        for _ in range(hi - lo):
            cs, ce = self._build_atom(elem)
            os, oe = self._opt(cs, ce, greedy)
            opt_parts.append((os, oe))
        for i in range(len(opt_parts) - 1):
            opt_parts[i][1].links.append(opt_parts[i + 1][0])

        # Stitch mandatory -> optional -> exit
        exit_ = Node(NT.SPLIT)
        if opt_parts:
            last_opt_e = opt_parts[-1][1]
            last_opt_e.links.append(exit_)
            if parts:
                parts[-1][1].links.append(opt_parts[0][0])
                return parts[0][0], exit_
            return opt_parts[0][0], exit_
        if parts:
            parts[-1][1].links.append(exit_)
            return parts[0][0], exit_
        # {0,0} — matches nothing
        return exit_, exit_



# ========================== Graph Parser ==========================

class GraphParser:
    """The core graph-walking parser.

    Maintains a set of cursors (active positions in the graph).  For each
    input token, every cursor is expanded through epsilon transitions
    (SPLIT, RULE_START, RULE_REF, RULE_END) to reach terminal nodes
    (MATCH_STR, MATCH_TOK).  Terminals that match the current token advance;
    the rest are pruned.  No lookahead or lookbehind is used.
    """

    MAX_DEPTH = 200  # stack depth limit to catch left recursion

    def __init__(self, rules, start_rule, lr_meta=None, skip_types=None,
                 stripped_names=None, capture_names=None):
        self.rules = rules          # {name: (start_node, end_node)}
        self.start_rule = start_rule
        self.lr_meta = lr_meta or {} # rule_name -> tail_rule_name
        # skip_types: set of token-type names that are "invisible" by default
        # but can be explicitly referenced in the grammar.
        self.skip_types = skip_types or set()
        # stripped_names: rules whose RULE_END produces no ParseNode in
        # the output tree (e.g. EBNF `A - B` anonymous rules).
        self.stripped_names = stripped_names or set()
        # capture_names: rule names that act as named captures; when
        # such a rule finishes, its effective value is bound in the
        # caller's caps under the rule's name, so `{{ ... }}` actions
        # can reference it.
        self.capture_names = capture_names or set()

    def parse(self, tokens):
        """Parse a token list.  Returns a ParseNode tree.

        Raises SyntaxError on invalid input.
        """
        start_node = self.rules[self.start_rule][0]
        # Cursor: (node, stack, acc, caps).  `caps` is a dict of
        # {capture_name: value} bindings visible to semantic actions
        # in the current rule scope.
        cursors = [(start_node, (), [], {})]

        real_tokens = [t for t in tokens if t.type != 'EOF']
        self._tokens = real_tokens

        for tok_pos, token in enumerate(real_tokens):
            expanded = self._expand_all(cursors, tok_pos)

            next_cursors = []
            for term_node, stack, acc, caps in expanded:
                if self._matches(term_node, token):
                    new_acc = acc + [ParseNode(token.type, value=token.value)]
                    for link in term_node.links:
                        next_cursors.append((link, stack, new_acc, caps))

            if token.type in self.skip_types:
                next_cursors.extend(cursors)
            elif not next_cursors:
                self._raise_error(token, expanded)

            cursors = self._dedup(next_cursors)

        end_pos = len(real_tokens)
        completions = self._find_completions(cursors, end_pos)
        if not completions:
            if real_tokens:
                last = real_tokens[-1]
                raise SyntaxError(
                    f"Unexpected end of input after {last.value!r} "
                    f"at line {last.line}, col {last.col}")
            else:
                completions = self._find_completions(
                    [(start_node, (), [], {})], 0)
                if not completions:
                    raise SyntaxError("Empty input does not match grammar")

        tree = completions[0]
        if self.lr_meta:
            tree = _reconstruct_lr(tree, self.lr_meta)
        return tree

    # ---- epsilon expansion ------------------------------------------

    def _expand_all(self, cursors, tok_pos=0):
        results = []
        for node, stack, acc, caps in cursors:
            self._expand(node, stack, acc, tok_pos, caps, results, set())
        return results

    def _expand(self, node, stack, acc, tok_pos, caps, results, visited):
        if len(stack) > self.MAX_DEPTH:
            return

        stack_key = tuple((r, rl) for r, rl, *_ in stack)
        cap_key = tuple(sorted(caps.items())) if caps else ()
        key = (node.id, stack_key, cap_key)
        if key in visited:
            return
        visited.add(key)

        ntype = node.type

        if ntype == NT.SPLIT or ntype == NT.RULE_START:
            for link in node.links:
                self._expand(link, stack, acc, tok_pos, caps,
                             results, visited)

        elif ntype == NT.RULE_REF:
            rule_name = node.value
            if rule_name not in self.rules:
                raise ValueError(f"Unknown rule reference: {rule_name!r}")
            rule_start = self.rules[rule_name][0]
            # Entering a rule starts a fresh capture scope so the
            # callee's captures don't leak into the caller's.  Parent
            # caps are saved on the stack and restored at RULE_END.
            new_stack = stack + ((rule_name, frozenset(node.links),
                                   acc, tok_pos, caps),)
            self._expand(rule_start, new_stack, [], tok_pos, {},
                         results, visited)

        elif ntype in (NT.PRED_NOT, NT.PRED_AND):
            matches = self._evaluate_predicate(node.value, tok_pos, caps)
            passes = (ntype == NT.PRED_AND and matches) or \
                     (ntype == NT.PRED_NOT and not matches)
            if passes:
                for link in node.links:
                    self._expand(link, stack, acc, tok_pos, caps,
                                 results, visited)

        elif ntype == NT.RULE_END:
            if stack:
                (rule_name, return_links, parent_acc, _sp,
                 parent_caps) = stack[-1]
                new_stack = stack[:-1]
                # Pick up the action result (if any) and fall back to a
                # single-child value when no action was declared —
                # mirrors bison's default `$$ = $1`.
                action_result = caps.get('__result__', None)
                effective_value = action_result
                if effective_value is None and len(acc) == 1 \
                        and acc[0].value is not None:
                    effective_value = acc[0].value
                if rule_name in self.stripped_names:
                    new_acc = parent_acc
                else:
                    rule_node = ParseNode(rule_name, list(acc),
                                          value=effective_value)
                    new_acc = parent_acc + [rule_node]
                # Restore caller's capture scope; add this rule's
                # capture binding if it's a capture name.
                new_caps = parent_caps
                if rule_name in self.capture_names:
                    cap_val = effective_value
                    new_caps = {**parent_caps, rule_name: cap_val}
                for link in return_links:
                    self._expand(link, new_stack, new_acc,
                                 tok_pos, new_caps, results, visited)

        elif ntype == NT.SUB_CHECK_NOT:
            if not stack:
                return
            start_pos = stack[-1][3]
            if self._evaluate_predicate_bounded(node.value, start_pos,
                                                 tok_pos, caps):
                return
            for link in node.links:
                self._expand(link, stack, acc, tok_pos, caps,
                             results, visited)

        elif ntype == NT.ACTION:
            # Evaluate the action's Python expression with the current
            # captures as locals, plus _tokens/_start/_end bookkeeping.
            # Stash the result in caps['__result__'] for RULE_END.
            start_pos = stack[-1][3] if stack else 0
            scope = {k: v for k, v in caps.items()
                     if not k.startswith('__')}
            scope['_tokens'] = self._tokens[start_pos:tok_pos]
            scope['_start']  = start_pos
            scope['_end']    = tok_pos
            try:
                result = eval(node.value, _ACTION_GLOBALS, scope)
            except Exception as e:
                raise RuntimeError(
                    f"semantic action error: {e!r} in action "
                    f"{node.value!r}") from e
            new_caps = {**caps, '__result__': result}
            for link in node.links:
                self._expand(link, stack, acc, tok_pos, new_caps,
                             results, visited)

        elif ntype in (NT.MATCH_STR, NT.MATCH_TOK):
            results.append((node, stack, acc, caps))

    # ---- predicate evaluation ----------------------------------------

    def _evaluate_predicate(self, pred_start, tok_pos, caps):
        self._pred_depth = getattr(self, '_pred_depth', 0) + 1
        if self._pred_depth > 50:
            self._pred_depth -= 1
            return False
        try:
            tokens = self._tokens
            cursors = [(pred_start, (), [], caps)]
            if self._find_completions(cursors, tok_pos):
                return True
            for i in range(tok_pos, len(tokens)):
                expanded = self._expand_all(cursors, i)
                next_c = []
                for tnode, stk, ac, cp in expanded:
                    if self._matches(tnode, tokens[i]):
                        new_acc = ac + [ParseNode(tokens[i].type,
                                                  value=tokens[i].value)]
                        for link in tnode.links:
                            next_c.append((link, stk, new_acc, cp))
                if tokens[i].type in self.skip_types:
                    next_c.extend(cursors)
                elif not next_c:
                    return False
                cursors = self._dedup(next_c)
                if self._find_completions(cursors, i + 1):
                    return True
            return bool(self._find_completions(cursors, len(tokens)))
        finally:
            self._pred_depth -= 1

    def _evaluate_predicate_bounded(self, pred_start, start_pos, end_pos,
                                     caps):
        self._pred_depth = getattr(self, '_pred_depth', 0) + 1
        if self._pred_depth > 50:
            self._pred_depth -= 1
            return False
        try:
            tokens = self._tokens
            cursors = [(pred_start, (), [], caps)]
            if start_pos == end_pos:
                return bool(self._find_completions(cursors, start_pos))
            for i in range(start_pos, end_pos):
                expanded = self._expand_all(cursors, i)
                next_c = []
                for tnode, stk, ac, cp in expanded:
                    if self._matches(tnode, tokens[i]):
                        new_acc = ac + [ParseNode(tokens[i].type,
                                                  value=tokens[i].value)]
                        for link in tnode.links:
                            next_c.append((link, stk, new_acc, cp))
                if not next_c:
                    return False
                cursors = self._dedup(next_c)
            return bool(self._find_completions(cursors, end_pos))
        finally:
            self._pred_depth -= 1

    # ---- completion check -------------------------------------------

    def _find_completions(self, cursors, tok_pos=None):
        if tok_pos is None:
            tok_pos = len(self._tokens) if hasattr(self, '_tokens') else 0
        results = []
        for node, stack, acc, caps in cursors:
            self._find_completion(node, stack, acc, tok_pos, caps,
                                  results, set())
        return results

    def _find_completion(self, node, stack, acc, tok_pos, caps, results,
                         visited):
        stack_key = tuple((r, rl) for r, rl, *_ in stack)
        cap_key = tuple(sorted(caps.items())) if caps else ()
        key = (node.id, stack_key, cap_key)
        if key in visited:
            return
        visited.add(key)

        ntype = node.type

        if ntype == NT.SPLIT or ntype == NT.RULE_START:
            for link in node.links:
                self._find_completion(link, stack, acc,
                                      tok_pos, caps, results, visited)

        elif ntype == NT.RULE_REF:
            rule_name = node.value
            if rule_name in self.rules:
                rule_start = self.rules[rule_name][0]
                new_stack = stack + ((rule_name, frozenset(node.links),
                                      acc, tok_pos, caps),)
                self._find_completion(rule_start, new_stack, [],
                                      tok_pos, {}, results, visited)

        elif ntype in (NT.PRED_NOT, NT.PRED_AND):
            matches = self._evaluate_predicate(node.value, tok_pos, caps)
            passes = (ntype == NT.PRED_AND and matches) or \
                     (ntype == NT.PRED_NOT and not matches)
            if passes:
                for link in node.links:
                    self._find_completion(link, stack, acc, tok_pos,
                                          caps, results, visited)

        elif ntype == NT.SUB_CHECK_NOT:
            if not stack:
                return
            start_pos = stack[-1][3]
            if self._evaluate_predicate_bounded(node.value, start_pos,
                                                 tok_pos, caps):
                return
            for link in node.links:
                self._find_completion(link, stack, acc, tok_pos, caps,
                                      results, visited)

        elif ntype == NT.ACTION:
            start_pos = stack[-1][3] if stack else 0
            scope = {k: v for k, v in caps.items()
                     if not k.startswith('__')}
            scope['_tokens'] = self._tokens[start_pos:tok_pos]
            scope['_start']  = start_pos
            scope['_end']    = tok_pos
            try:
                result = eval(node.value, _ACTION_GLOBALS, scope)
            except Exception as e:
                raise RuntimeError(
                    f"semantic action error: {e!r} in action "
                    f"{node.value!r}") from e
            new_caps = {**caps, '__result__': result}
            for link in node.links:
                self._find_completion(link, stack, acc, tok_pos,
                                      new_caps, results, visited)

        elif ntype == NT.RULE_END:
            if stack:
                (rule_name, return_links, parent_acc, _sp,
                 parent_caps) = stack[-1]
                new_stack = stack[:-1]
                action_result = caps.get('__result__', None)
                effective_value = action_result
                if effective_value is None and len(acc) == 1 \
                        and acc[0].value is not None:
                    effective_value = acc[0].value
                if rule_name in self.stripped_names:
                    new_acc = parent_acc
                else:
                    rule_node = ParseNode(rule_name, list(acc),
                                          value=effective_value)
                    new_acc = parent_acc + [rule_node]
                new_caps = parent_caps
                if rule_name in self.capture_names:
                    cap_val = effective_value
                    new_caps = {**parent_caps, rule_name: cap_val}
                for link in return_links:
                    self._find_completion(link, new_stack, new_acc,
                                          tok_pos, new_caps,
                                          results, visited)
            else:
                action_result = caps.get('__result__', None)
                effective_value = action_result
                if effective_value is None and len(acc) == 1 \
                        and acc[0].value is not None:
                    effective_value = acc[0].value
                results.append(ParseNode(node.rule, list(acc),
                                          value=effective_value))

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _matches(node, token):
        if node.type == NT.MATCH_STR:
            return token.value == node.value
        if node.type == NT.MATCH_TOK:
            return token.type == node.value
        return False

    @staticmethod
    def _dedup(cursors):
        """First-seen wins.  Combined with link-order traversal in
        _expand, this implements ordered choice.

        Prediction merging (Earley-style): the dedup key excludes
        return links from stack frames.  Cursors that differ only in
        where a rule will return to are doing identical work inside
        that rule, so we merge them into one cursor carrying the
        union of return-link sets.  At RULE_END the merged cursor
        forks into one continuation per return link.  This keeps the
        cursor set polynomial on deep precedence chains (without it,
        N precedence levels can create 2^N stacks).
        """
        seen = {}
        for cursor in cursors:
            node, stack, acc, caps = cursor
            stack_key = tuple(
                (r, tp, tuple(sorted(ca.items())) if ca else ())
                for r, rl, a, tp, ca in stack
            )
            cap_key = tuple(sorted(caps.items())) if caps else ()
            key = (node.id, stack_key, cap_key)
            if key not in seen:
                seen[key] = cursor
            else:
                # Merge return links at every stack level.
                e = seen[key]
                merged = tuple(
                    (r, rl_e | rl_n, a, tp, ca)
                    for (r, rl_e, a, tp, ca), (_, rl_n, *_)
                    in zip(e[1], stack)
                )
                seen[key] = (e[0], merged, e[2], e[3])
        return list(seen.values())

    @staticmethod
    def _raise_error(token, expanded):
        expected = set()
        for term_node, *_ in expanded:
            if term_node.type == NT.MATCH_STR:
                expected.add(f"'{term_node.value}'")
            elif term_node.type == NT.MATCH_TOK:
                expected.add(term_node.value)
        exp_str = ', '.join(sorted(expected)) if expected else '(nothing)'
        raise SyntaxError(
            f"Unexpected {token.type} {token.value!r} "
            f"at line {token.line}, col {token.col}.  "
            f"Expected: {exp_str}")


# ========================== EPEG Bootstrap Parser ==========================

class EPEGBootstrap:
    """Recursive-descent parser for EPEG grammar files.

    Produces the grammar dict expected by GraphBuilder.  This is the
    'bootstrap' parser: it parses EPEG using conventional techniques so
    that the graph parser can then take over for everything else.

    EPEG format:
        start <rule_name>

        rule_name = sequence ('|' sequence)*
        sequence  = element+
        element   = atom ('*'|'*?'|'+'|'+?'|'?'|'??'|'{n}'|'{n,m}')?
        atom      = 'string' | TOKEN_NAME | rule_name | '(' alternatives ')'
                  | '&' atom | '!' atom

    Convention: ALL_CAPS names are token references; others are rule refs.
    """

    EPEG_LEXER = Lexer([
        ('COMMENT',  r'#[^\n]*'),
        ('NEWLINE',  r'\n'),
        ('WS',       r'[ \t\r]+'),
        ('START_KW', r'start\b'),
        ('STARQ',    r'\*\?'),
        ('PLUSQ',    r'\+\?'),
        ('QQMARK',   r'\?\?'),
        ('STAR',     r'\*'),
        ('PLUS',     r'\+'),
        ('QMARK',    r'\?'),
        ('OR',       r'\|'),
        ('AND_OP',   r'&'),
        ('NOT_OP',   r'!'),
        ('CAPTURE',  r':='),
        ('EQUALS',   r'='),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        # ACTION must precede LBRACE so `{{` starts an action-block
        # rather than two `{` tokens (which would try to open a `{n,m}`
        # repetition modifier).
        ('ACTION',   r'\{\{[\s\S]*?\}\}'),
        ('LBRACE',   r'\{'),
        ('RBRACE',   r'\}'),
        ('COMMA',    r','),
        ('NUMBER',   r'\d+'),
        ('MINUS',    r'-'),
        ('SKIP_KW',  r'@skip\b'),
        ('KW_KW',    r'@keyword\b'),
        ('LEFT_KW',  r'@left\b'),
        ('RIGHT_KW', r'@right\b'),
        ('NONASSOC_KW', r'@nonassoc\b'),
        ('REGEX',    r'/(?:\\.|[^/\\])*/[a-z]*'),
        ('STRING',   r"'[^']*'|\"[^\"]*\""),
        ('NAME',     r'[a-zA-Z_][a-zA-Z_0-9]*'),
    ], ignore={'WS', 'COMMENT'})

    def __init__(self, text):
        self.tokens = self.EPEG_LEXER.tokenize(text)
        self.pos = 0
        self._skip_newlines()

    # ---- token helpers -----------------------------------------------

    def _peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) \
            else Token('EOF', '')

    def _advance(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _expect(self, ttype):
        tok = self._peek()
        if tok.type != ttype:
            raise SyntaxError(
                f"Expected {ttype}, got {tok.type} {tok.value!r} "
                f"at line {tok.line}, col {tok.col}")
        return self._advance()

    def _skip_newlines(self):
        while self._peek().type == 'NEWLINE':
            self._advance()

    def _at(self, *types):
        return self._peek().type in types

    # ---- grammar rules -----------------------------------------------

    def parse(self):
        """Parse the EPEG text.  Returns a grammar dict."""
        rules = {}
        skip_info = {}
        start = None
        precedence = []  # (assoc, [op_literals]) in declaration order

        while not self._at('EOF'):
            self._skip_newlines()
            if self._at('EOF'):
                break

            if self._at('START_KW'):
                self._advance()
                start = self._expect('NAME').value
                self._skip_newlines()
            elif self._at('KW_KW'):
                # Top-level @keyword declarations are accepted for
                # grammar portability but are semantically a no-op in
                # the tokenised parser — the lexer's longest-match rule
                # already ensures `'class'` only matches a full word,
                # never a prefix of `'classroom'`.  Consume and ignore.
                self._advance()
                while self._at('STRING'):
                    self._advance()
                self._skip_newlines()
            elif self._at('LEFT_KW') or self._at('RIGHT_KW') \
                    or self._at('NONASSOC_KW'):
                tok = self._advance()
                assoc = {'LEFT_KW': 'left',
                         'RIGHT_KW': 'right',
                         'NONASSOC_KW': 'nonassoc'}[tok.type]
                ops = []  # list of ('string', value) or ('name', value)
                while self._at('STRING') or self._at('NAME'):
                    if self._at('STRING'):
                        s = self._advance().value
                        ops.append(('string', s[1:-1]))
                    else:
                        n = self._advance().value
                        ops.append(('name', n))
                if not ops:
                    tt = self._peek()
                    raise SyntaxError(
                        f"@{assoc} expects at least one operator (string "
                        f"literal or token name) at line {tt.line}")
                precedence.append((assoc, ops))
                self._skip_newlines()
            elif self._at('NAME'):
                name, alts, skip = self._parse_rule()
                rules[name] = alts
                if skip:
                    skip_info[name] = True
                self._skip_newlines()
            else:
                tok = self._peek()
                raise SyntaxError(
                    f"Expected rule or 'start', got {tok.type} {tok.value!r} "
                    f"at line {tok.line}")

        if not start and rules:
            start = next(iter(rules))

        return {'start': start, 'rules': rules, 'skip_info': skip_info,
                'precedence': precedence}

    def _parse_rule(self):
        name = self._expect('NAME').value
        self._expect('EQUALS')
        alts = self._parse_alternatives()
        skip = False
        if self._at('SKIP_KW'):
            self._advance()
            skip = True
        return name, alts, skip

    def _parse_alternatives(self):
        alts = [self._parse_sequence()]
        while True:
            self._skip_newlines()
            if self._at('OR'):
                self._advance()
                alts.append(self._parse_sequence())
            else:
                break
        return alts

    def _parse_sequence(self):
        elems = []
        while self._at('STRING', 'NAME', 'START_KW',
                        'LPAREN', 'AND_OP', 'NOT_OP', 'REGEX',
                        'KW_KW'):
            elems.append(self._parse_element())
        # Optional trailing semantic action: {{ python_expr }}.  Encoded
        # as a pseudo-element that the graph builder turns into an
        # NT.ACTION node.
        if self._at('ACTION'):
            tok = self._advance()
            # Strip outer {{ and }}; keep inner verbatim (stripped).
            code = tok.value[2:-2].strip()
            elems.append({'type': 'action', 'code': code})
        return elems

    def _parse_element(self):
        elem = self._parse_atom()
        mod = self._parse_modifier()
        if mod is not None:
            elem['modifier'] = mod
        # EPEG subtraction: `A - B` matches A but rejects if B matches
        # the same span.  Binds looser than modifiers (so `A* - B` means
        # `(A*) - B`) and tighter than juxtaposition / `|`.
        if self._at('MINUS'):
            self._advance()
            right = self._parse_atom()
            rmod = self._parse_modifier()
            if rmod is not None:
                right['modifier'] = rmod
            elem = {'type': 'subtract', 'left': elem, 'right': right}
        return elem

    @staticmethod
    def _split_regex(raw):
        """Given a raw '/pattern/flags' token value, return (pattern, flags)."""
        # Find the last '/' that isn't escaped
        i = len(raw) - 1
        while i >= 0 and raw[i] != '/':
            i -= 1
        return raw[1:i], raw[i + 1:]

    def _parse_atom(self):
        tok = self._peek()

        if tok.type == 'STRING':
            self._advance()
            return {'type': 'string', 'value': tok.value[1:-1]}

        if tok.type == 'REGEX':
            self._advance()
            pat, flags = self._split_regex(tok.value)
            return {'type': 'regex', 'pattern': pat, 'flags': flags}

        if tok.type == 'KW_KW':
            # Inline @keyword 'string' — no-op in tokenised parser
            # (lexer longest-match handles the discrimination).  Consume
            # the @keyword then return the wrapped literal unchanged.
            self._advance()
            inner = self._parse_atom()
            if inner.get('type') != 'string':
                raise SyntaxError(
                    f"@keyword must prefix a string literal, got "
                    f"{inner.get('type')}")
            return inner

        if tok.type in ('NAME', 'START_KW'):
            self._advance()
            name = tok.value
            # Named capture/token: name:=atom  or  name:=(alternatives)
            if self._at('CAPTURE'):
                self._advance()
                if self._at('LPAREN'):
                    self._advance()
                    alts = self._parse_alternatives()
                    self._expect('RPAREN')
                    return {'type': 'capture', 'name': name,
                            'alternatives': alts}
                inner = self._parse_atom()
                return {'type': 'capture', 'name': name,
                        'alternatives': [[inner]]}
            if name.isupper():
                return {'type': 'token', 'value': name}
            return {'type': 'rule', 'value': name}

        if tok.type == 'LPAREN':
            self._advance()
            alts = self._parse_alternatives()
            self._expect('RPAREN')
            return {'type': 'group', 'alternatives': alts}

        if tok.type in ('AND_OP', 'NOT_OP'):
            op_type = 'and' if tok.type == 'AND_OP' else 'not'
            self._advance()
            inner = self._parse_atom()
            if inner.get('type') == 'group':
                alts = inner['alternatives']
            else:
                alts = [[inner]]
            return {'type': op_type, 'alternatives': alts}

        raise SyntaxError(
            f"Expected atom, got {tok.type} {tok.value!r} "
            f"at line {tok.line}, col {tok.col}")

    def _parse_modifier(self):
        tok = self._peek()
        if tok.type == 'STARQ':
            self._advance(); return '*?'
        if tok.type == 'STAR':
            self._advance(); return '*'
        if tok.type == 'PLUSQ':
            self._advance(); return '+?'
        if tok.type == 'PLUS':
            self._advance(); return '+'
        if tok.type == 'QQMARK':
            self._advance(); return '??'
        if tok.type == 'QMARK':
            self._advance(); return '?'
        if tok.type == 'LBRACE':
            return self._parse_brace_mod()
        return None

    def _parse_brace_mod(self):
        self._expect('LBRACE')
        # Forms: {n}, {n,}, {n,m}, {,m}
        if self._at('COMMA'):
            self._advance()
            lo = 0
            hi = int(self._expect('NUMBER').value)
            self._expect('RBRACE')
            mod = {'min': lo, 'max': hi, 'greedy': True}
        else:
            lo = int(self._expect('NUMBER').value)
            if self._at('COMMA'):
                self._advance()
                if self._at('NUMBER'):
                    hi = int(self._advance().value)
                else:
                    hi = -1
                self._expect('RBRACE')
                mod = {'min': lo, 'max': hi, 'greedy': True}
            else:
                self._expect('RBRACE')
                mod = {'min': lo, 'max': lo, 'greedy': True}
        if self._at('QMARK'):
            self._advance()
            mod['greedy'] = False
        return mod


# ========================== Token Resolution ==========================

def _resolve_tokens(grammar):
    """Extract token definitions from the grammar.

    Rules whose body is a single bare /regex/ become token definitions.
    Inline /regex/ in other positions gets an auto-generated token name.

    Returns a new grammar dict with:
        - 'rules'        — grammar rules (no 'regex' elements remain)
        - 'tokens'       — dict of {name: (pattern, flags)}
        - 'skip_types'   — set of token names marked `skip`
    """
    import copy
    import re as re_module
    rules = copy.deepcopy(grammar['rules'])
    skip_info = grammar.get('skip_info', {})

    tokens = {}          # name -> (pattern, flags)
    skip_types = set()
    counter = [0]
    regex_cache = {}     # (pattern, flags) -> token_name (dedup)

    def intern_regex(pattern, flags):
        key = (pattern, flags)
        if key in regex_cache:
            return regex_cache[key]
        counter[0] += 1
        name = f'_regex_{counter[0]}'
        tokens[name] = (pattern, flags)
        regex_cache[key] = name
        return name

    # Step 1: token-shaped rules become explicit token definitions.
    token_shaped = []
    for name, alts in rules.items():
        if (len(alts) == 1 and len(alts[0]) == 1
                and alts[0][0].get('type') == 'regex'
                and 'modifier' not in alts[0][0]):
            elem = alts[0][0]
            tokens[name] = (elem['pattern'], elem['flags'])
            regex_cache[(elem['pattern'], elem['flags'])] = name
            if skip_info.get(name):
                skip_types.add(name)
            token_shaped.append(name)
    for name in token_shaped:
        del rules[name]

    # Step 2: walk remaining rules to (a) replace inline /regex/ with
    # token references and (b) collect literal strings for the lexer.
    literals = set()

    def process(elem):
        if elem.get('type') == 'regex':
            tok_name = intern_regex(elem['pattern'], elem['flags'])
            out = {'type': 'token', 'value': tok_name}
            if 'modifier' in elem:
                out['modifier'] = elem['modifier']
            return out
        if elem.get('type') == 'rule' and elem['value'] in tokens:
            # Reference to what is now a token, not a rule
            out = {'type': 'token', 'value': elem['value']}
            if 'modifier' in elem:
                out['modifier'] = elem['modifier']
            return out
        if elem.get('type') == 'string':
            literals.add(elem['value'])
            return elem  # MATCH_STR stays — matched by value on whichever
                         # token the lexer produces.
        if 'alternatives' in elem:
            elem = dict(elem)
            elem['alternatives'] = [
                [process(e) for e in alt]
                for alt in elem['alternatives']
            ]
        return elem

    for name in rules:
        rules[name] = [[process(e) for e in alt] for alt in rules[name]]

    # Step 3: for each unique literal string, create a lexer token so the
    # auto-lexer knows how to produce it.  Longer literals first so they
    # beat shorter ones on ties.
    for lit in sorted(literals, key=lambda s: (-len(s), s)):
        counter[0] += 1
        tokens[f'_lit_{counter[0]}'] = (re_module.escape(lit), '')

    return {
        'start': grammar['start'],
        'rules': rules,
        'tokens': tokens,
        'skip_types': skip_types,
        'precedence': grammar.get('precedence', []),
    }


def _build_auto_lexer(tokens, skip_types):
    """Build a Lexer from auto-generated token definitions.

    Token priority: longer literals first (to disambiguate keywords from
    pattern matches), then declaration order.
    """
    if not tokens:
        return None

    # Regex flags — passed through Python re's inline flag syntax:
    #   i: case-insensitive
    #   x: verbose (whitespace + # comments ignored)
    #   u: Unicode (Python str patterns are Unicode by default, so this
    #      is effectively a no-op in tokenized — accepted for parity
    #      with the scannerless parser's /u flag)
    #   s: dot-matches-newline
    #   m: multi-line (^/$ match per line; we don't use anchors but
    #      accept the flag for user convenience)
    def compile_pat(pattern, flags):
        py = ''.join(c for c in 'iuxsm' if c in flags)
        return f'(?{py}){pattern}' if py else pattern

    # Put more-specific (longer literal-like) patterns first
    rules = []
    for name, (pat, flags) in tokens.items():
        rules.append((name, compile_pat(pat, flags)))

    # Lexer doesn't filter skip tokens here — parser handles them
    return Lexer(rules)


# ========================== Grammar Parser ==========================

class GrammarParser:
    """Combines an auto-generated Lexer with a GraphParser.

    Use .parse(text) for strings (auto-lexed) or .parse(tokens) for
    pre-lexed token lists.
    """

    def __init__(self, lexer, parser):
        self.lexer = lexer
        self.parser = parser

    def parse(self, source):
        if isinstance(source, str):
            if self.lexer is None:
                raise TypeError(
                    "Grammar defines no tokens; provide pre-lexed Token list")
            tokens = self.lexer.tokenize(source)
        else:
            tokens = source
        return self.parser.parse(tokens)


# ========================== EBNF Bootstrap (ISO 14977) ==================

class EBNFBootstrap:
    """Recursive-descent parser for ISO 14977 EBNF grammars.

    Supported syntax:
        syntax_rule       = meta_identifier '=' definitions_list ';'
        definitions_list  = single_definition ( '|' single_definition )*
        single_definition = syntactic_term ( ',' syntactic_term )*
        syntactic_term    = syntactic_factor ( '-' syntactic_factor )?
        syntactic_factor  = [ integer '*' ] syntactic_primary
        syntactic_primary = terminal_string | meta_identifier
                          | '[' definitions_list ']'     (optional)
                          | '{' definitions_list '}'     (zero-or-more)
                          | '(' definitions_list ')'     (grouping)
        terminal_string   = "'" ... "'" | '"' ... '"'
        meta_identifier   = letter ( letter | digit | '_' | '-' )*
        comments          = '(*' ... '*)'

    Rejected (SyntaxError):
        '?special sequence?' — implementation-defined, not interpretable
        All EPEG extensions (regex, @directives, predicates, actions, etc.)

    In tokenized mode, terminal strings match on a token's ``value``
    field (bring-your-own-lexer) — ISO EBNF has no concept of tokens,
    so the caller supplies pre-lexed tokens directly to ``GraphParser.parse``.
    """

    _LEXER_RULES = [
        ('COMMENT', r'\(\*(?:[^*]|\*(?!\)))*\*\)'),
        ('WS',      r'[ \t\r\n]+'),
        ('LBRACE',  r'\{'),
        ('RBRACE',  r'\}'),
        ('LBRACK',  r'\['),
        ('RBRACK',  r'\]'),
        ('LPAREN',  r'\('),
        ('RPAREN',  r'\)'),
        ('SEMI',    r';'),
        ('EQUALS',  r'='),
        ('COMMA',   r','),
        ('OR',      r'\|'),
        ('STAR',    r'\*'),
        ('MINUS',   r'-'),
        ('SPECIAL', r'\?[^?]*\?'),
        ('NUMBER',  r'\d+'),
        ('STRING',  r'"[^"]*"' + r"|'[^']*'"),
        ('NAME',    r'[a-zA-Z][a-zA-Z0-9_\-]*'),
    ]

    def __init__(self, text):
        self.tokens = self._tokenize(text)
        self.pos = 0

    def _tokenize(self, text):
        compiled = [(n, re.compile(p)) for n, p in self._LEXER_RULES]
        ignore = {'WS', 'COMMENT'}
        toks, p, line, col = [], 0, 1, 1
        while p < len(text):
            for name, pat in compiled:
                m = pat.match(text, p)
                if m:
                    val = m.group()
                    if name not in ignore:
                        toks.append((name, val, line, col))
                    for ch in val:
                        if ch == '\n': line += 1; col = 1
                        else:          col += 1
                    p = m.end()
                    break
            else:
                raise SyntaxError(
                    f"Unexpected {text[p]!r} at line {line}, col {col}")
        toks.append(('EOF', '', line, col))
        return toks

    def _peek(self):       return self.tokens[self.pos]
    def _peek_at(self, k): return self.tokens[self.pos + k]
    def _advance(self):
        t = self.tokens[self.pos]; self.pos += 1; return t
    def _at(self, *types): return self._peek()[0] in types
    def _expect(self, tp):
        t = self._peek()
        if t[0] != tp:
            raise SyntaxError(
                f"Expected {tp}, got {t[0]} {t[1]!r} at line {t[2]}")
        return self._advance()

    def parse(self):
        rules, order = {}, []
        while not self._at('EOF'):
            name = self._expect('NAME')[1]
            self._expect('EQUALS')
            alts = self._parse_definitions_list()
            self._expect('SEMI')
            rules[name] = alts
            order.append(name)
        start = order[0] if order else None
        return {'start': start, 'rules': rules}

    def _parse_definitions_list(self):
        alts = [self._parse_single_definition()]
        while self._at('OR'):
            self._advance()
            alts.append(self._parse_single_definition())
        return alts

    def _parse_single_definition(self):
        terms = [self._parse_syntactic_term()]
        while self._at('COMMA'):
            self._advance()
            terms.append(self._parse_syntactic_term())
        return terms

    def _parse_syntactic_term(self):
        factor = self._parse_syntactic_factor()
        if self._at('MINUS'):
            self._advance()
            right = self._parse_syntactic_factor()
            return {'type': 'subtract', 'left': factor, 'right': right}
        return factor

    def _parse_syntactic_factor(self):
        if self._at('NUMBER') and self._peek_at(1)[0] == 'STAR':
            n = int(self._advance()[1])
            self._advance()
            primary = self._parse_syntactic_primary()
            return {'type': 'group', 'alternatives': [[primary]],
                    'modifier': {'min': n, 'max': n, 'greedy': True}}
        return self._parse_syntactic_primary()

    def _parse_syntactic_primary(self):
        t = self._peek()
        if t[0] == 'STRING':
            self._advance()
            return {'type': 'string', 'value': t[1][1:-1]}
        if t[0] == 'NAME':
            self._advance()
            return {'type': 'rule', 'value': t[1]}
        if t[0] == 'LPAREN':
            self._advance()
            alts = self._parse_definitions_list()
            self._expect('RPAREN')
            return {'type': 'group', 'alternatives': alts}
        if t[0] == 'LBRACK':
            self._advance()
            alts = self._parse_definitions_list()
            self._expect('RBRACK')
            return {'type': 'group', 'alternatives': alts, 'modifier': '?'}
        if t[0] == 'LBRACE':
            self._advance()
            alts = self._parse_definitions_list()
            self._expect('RBRACE')
            return {'type': 'group', 'alternatives': alts, 'modifier': '*'}
        if t[0] == 'SPECIAL':
            raise SyntaxError(
                f"?special? sequences are implementation-defined and not "
                f"supported (line {t[2]})")
        raise SyntaxError(
            f"Expected primary, got {t[0]} {t[1]!r} at line {t[2]}")


# ========================== High-Level API ==========================

def load_grammar(text, ebnf=False):
    """Parse a grammar string and return a parser.

    By default, parses EPEG (GPDA's PEG-family grammar syntax with regex tokens,
    @directives, predicates, actions, etc.) and returns a
    ``GrammarParser`` that can accept raw text (it runs the auto-lexer
    first).

    With ``ebnf=True``, parses ISO 14977 EBNF — a strict subset where
    ``A - B`` subtraction is supported but no EPEG extensions, no
    regex tokens, no auto-lexer.  Returns a ``GraphParser`` directly;
    the caller supplies pre-lexed tokens (bring-your-own-lexer).  The
    grammar's terminal strings match a token's ``value`` field.
    """
    if ebnf:
        grammar = EBNFBootstrap(text).parse()
        builder = GraphBuilder()
        rules, start, lr_meta, stripped, captures = builder.build(grammar)
        return GraphParser(rules, start, lr_meta,
                           stripped_names=stripped,
                           capture_names=captures)
    bootstrap = EPEGBootstrap(text)
    grammar = bootstrap.parse()
    grammar = _resolve_tokens(grammar)
    lexer = _build_auto_lexer(grammar['tokens'], grammar['skip_types'])
    builder = GraphBuilder()
    rules, start, lr_meta, stripped, captures = builder.build(grammar)
    parser = GraphParser(rules, start, lr_meta,
                         skip_types=grammar['skip_types'],
                         stripped_names=stripped,
                         capture_names=captures)
    return GrammarParser(lexer, parser)


def parse(grammar_text, source):
    """One-shot: parse source against an EPEG grammar string."""
    return load_grammar(grammar_text).parse(source)


# ========================== EPEG Self-Description ==========================

# This grammar describes the EPEG format itself, written in EPEG.
# It can be parsed by load_grammar() and then used with the graph parser
# to parse other EPEG files — completing the bootstrap.

EPEG_GRAMMAR = r"""
start grammar

grammar = NEWLINE* decl+ NEWLINE*
decl = start_decl | rule
start_decl = START_KW NAME newlines
rule = NAME EQUALS alternatives newlines
alternatives = sequence (NEWLINE* OR sequence)*
sequence = element+
element = atom modifier? (MINUS atom modifier?)?
atom = STRING | NAME | LPAREN alternatives RPAREN
modifier = STARQ | STAR | PLUSQ | PLUS | QQMARK | QMARK
         | LBRACE NUMBER RBRACE
         | LBRACE NUMBER COMMA NUMBER RBRACE
newlines = NEWLINE+
"""


# ========================== Graph Visualization ==========================

def dump_graph(rules, start_rule=None):
    """Print a human-readable dump of the grammar graph (for debugging)."""
    for name, (start, end) in rules.items():
        marker = " (START)" if name == start_rule else ""
        print(f"\n=== Rule: {name}{marker} ===")
        visited = set()
        _dump_node(start, visited, indent=0)


def _dump_node(node, visited, indent):
    if node.id in visited:
        print(f"{'  ' * indent}-> (back to Node {node.id})")
        return
    visited.add(node.id)
    label = {NT.MATCH_STR: f"MATCH '{node.value}'",
             NT.MATCH_TOK: f"MATCH [{node.value}]",
             NT.RULE_REF:  f"CALL {node.value}",
             NT.SPLIT:     "SPLIT",
             NT.RULE_START: f"START({node.rule})",
             NT.RULE_END:   f"END({node.rule})"}
    print(f"{'  ' * indent}[{node.id}] {label.get(node.type, '?')}")
    for link in node.links:
        _dump_node(link, visited, indent + 1)


# ========================== Tests ==========================

if __name__ == '__main__':
    passed = 0
    failed = 0

    def check(name, condition):
        global passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            failed += 1

    # ---- Test 1: simple string matching ----
    print("\n--- Test 1: simple string matching ---")
    p = load_grammar("start g\ng = 'hello' 'world'")
    lx = Lexer([('WS', r'[ \t]+'), ('WORD', r'[a-z]+')], ignore={'WS'})
    tree = p.parse(lx.tokenize("hello world"))
    check("root is 'g'", tree.name == 'g')
    check("2 children", len(tree.children) == 2)
    check("first child 'hello'", tree.children[0].value == 'hello')
    check("second child 'world'", tree.children[1].value == 'world')

    # ---- Test 2: alternatives ----
    print("\n--- Test 2: alternatives ---")
    p = load_grammar("start c\nc = 'a' | 'b' | 'c'")
    lx = Lexer([('CH', r'[a-c]')])
    for ch in 'abc':
        tree = p.parse(lx.tokenize(ch))
        check(f"'{ch}' matches", tree.children[0].value == ch)

    # ---- Test 3: star (zero-or-more) ----
    print("\n--- Test 3: star ---")
    p = load_grammar("start items\nitems = 'a'*")
    lx = Lexer([('CH', r'[a-z]')])
    # Empty input
    tree = p.parse(lx.tokenize(""))
    check("empty match", len(tree.children) == 0)
    # Three items
    tree = p.parse(lx.tokenize("aaa"))
    check("three a's", len(tree.children) == 3)

    # ---- Test 4: plus (one-or-more) ----
    print("\n--- Test 4: plus ---")
    p = load_grammar("start items\nitems = 'a'+")
    lx = Lexer([('CH', r'[a-z]')])
    tree = p.parse(lx.tokenize("aaa"))
    check("three a's", len(tree.children) == 3)
    try:
        p.parse(lx.tokenize(""))
        check("empty fails", False)
    except SyntaxError:
        check("empty fails", True)

    # ---- Test 5: optional ----
    print("\n--- Test 5: optional ---")
    p = load_grammar("start g\ng = 'a' 'b'?")
    lx = Lexer([('CH', r'[a-z]')])
    tree = p.parse(lx.tokenize("ab"))
    check("'ab' -> 2 children", len(tree.children) == 2)
    tree = p.parse(lx.tokenize("a"))
    check("'a' -> 1 child", len(tree.children) == 1)

    # ---- Test 6: nested rules ----
    print("\n--- Test 6: nested rules ---")
    p = load_grammar("""
        start top
        top = inner 'z'
        inner = 'x' 'y'
    """)
    lx = Lexer([('CH', r'[a-z]')])
    tree = p.parse(lx.tokenize("xyz"))
    check("root is 'top'", tree.name == 'top')
    check("inner rule matched", tree.children[0].name == 'inner')
    check("inner has x, y", len(tree.children[0].children) == 2)
    check("z at end", tree.children[1].value == 'z')

    # ---- Test 7: expression grammar ----
    print("\n--- Test 7: expression grammar ---")
    p = load_grammar("""
        start expr
        expr = term ('+' term)*
        term = factor ('*' factor)*
        factor = NUMBER | '(' expr ')'
    """)
    lx = Lexer([
        ('WS', r'[ \t]+'),
        ('NUMBER', r'\d+'),
        ('PLUS', r'\+'),
        ('TIMES', r'\*'),
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
    ], ignore={'WS'})

    tree = p.parse(lx.tokenize("1+2*3"))
    print(tree.pretty())
    check("root is 'expr'", tree.name == 'expr')
    # expr should have: term, '+', term
    check("expr has 3 children", len(tree.children) == 3)
    check("first child is term", tree.children[0].name == 'term')
    check("second child is '+'", tree.children[1].value == '+')
    check("third child is term", tree.children[2].name == 'term')
    # second term should have: factor, '*', factor
    term2 = tree.children[2]
    check("term2 has 3 children", len(term2.children) == 3)

    tree = p.parse(lx.tokenize("(1+2)*3"))
    check("parenthesized expr", tree.name == 'expr')
    print(tree.pretty())

    # ---- Test 8: repetition {n} and {n,m} ----
    print("\n--- Test 8: repetition {n} and {n,m} ---")
    p = load_grammar("start g\ng = 'a'{3}")
    lx = Lexer([('CH', r'[a-z]')])
    tree = p.parse(lx.tokenize("aaa"))
    check("{3} matches exactly 3", len(tree.children) == 3)
    try:
        p.parse(lx.tokenize("aa"))
        check("{3} rejects 2", False)
    except SyntaxError:
        check("{3} rejects 2", True)

    p = load_grammar("start g\ng = 'a'{2,4}")
    tree = p.parse(lx.tokenize("aaa"))
    check("{2,4} matches 3", len(tree.children) == 3)
    tree = p.parse(lx.tokenize("aa"))
    check("{2,4} matches 2", len(tree.children) == 2)
    try:
        p.parse(lx.tokenize("a"))
        check("{2,4} rejects 1", False)
    except SyntaxError:
        check("{2,4} rejects 1", True)

    # ---- Test 9: non-greedy star ----
    print("\n--- Test 9: non-greedy star ---")
    # Both greedy and non-greedy should parse correctly when the overall
    # grammar is unambiguous.  With 'a'*? 'a', for "aaa" both must consume
    # all tokens — the non-greedy star just explores shorter matches first.
    p = load_grammar("start g\ng = 'a'*? 'a'")
    lx = Lexer([('CH', r'[a-z]')])
    tree = p.parse(lx.tokenize("aaa"))
    check("non-greedy 'a'*? 'a' parses 'aaa'", tree.name == 'g')
    check("total 3 children", len(tree.children) == 3)

    # ---- Test 10: self-bootstrap ----
    print("\n--- Test 10: EPEG self-bootstrap ---")
    # Parse the EPEG grammar description using the bootstrap parser
    bootstrap = EPEGBootstrap(EPEG_GRAMMAR)
    epeg_grammar_dict = bootstrap.parse()
    check("bootstrap parsed EPEG grammar",
          'grammar' in epeg_grammar_dict['rules'])
    check("start rule is 'grammar'",
          epeg_grammar_dict['start'] == 'grammar')

    # Build a graph parser from the EPEG grammar
    builder = GraphBuilder()
    epeg_rules, epeg_start, _, _, _ = builder.build(epeg_grammar_dict)
    epeg_parser = GraphParser(epeg_rules, epeg_start)

    # Now use the graph parser to parse a simple test grammar
    test_grammar_text = """start expr
expr = term ('+' term)*
term = NUMBER
"""
    test_tokens = EPEGBootstrap.EPEG_LEXER.tokenize(test_grammar_text)
    tree = epeg_parser.parse(test_tokens)
    check("graph parser parsed test grammar", tree.name == 'grammar')
    print(tree.pretty())

    # Also parse the EPEG grammar itself (full self-bootstrap)
    epeg_tokens = EPEGBootstrap.EPEG_LEXER.tokenize(EPEG_GRAMMAR)
    tree = epeg_parser.parse(epeg_tokens)
    check("full self-bootstrap: EPEG parsed itself", tree.name == 'grammar')
    print(tree.pretty())

    # ---- Test 11: comments and blank lines ----
    print("\n--- Test 11: comments and blank lines ---")
    # Verify the bootstrap parser handles comments and blank lines
    grammar_with_comments = """
# This is a comment at the top

start expr

# Expression grammar
expr = term ('+' term)*

# Term is just a number for now
term = NUMBER
# trailing comment
"""
    p = load_grammar(grammar_with_comments)
    lx = Lexer([
        ('WS', r'[ \t]+'), ('NUMBER', r'\d+'), ('PLUS', r'\+'),
    ], ignore={'WS'})
    tree = p.parse(lx.tokenize("1+2+3"))
    check("comments: bootstrap parses grammar with comments",
          tree.name == 'expr')
    check("comments: 5 children (3 terms + 2 pluses)",
          len(tree.children) == 5)

    # Verify the graph parser (EPEG self-description) also handles them
    commented_tokens = EPEGBootstrap.EPEG_LEXER.tokenize(
        grammar_with_comments)
    tree = epeg_parser.parse(commented_tokens)
    check("comments: graph parser handles comments and blank lines",
          tree.name == 'grammar')

    # ---- Test 12: recursive grammar (from blog post) ----
    print("\n--- Test 12: recursive grammar (from blog post) ---")
    p = load_grammar("""
        start s
        s = a '#'
        a = 'a' a 'b'
          | 'c'
    """)
    lx = Lexer([('CH', r'[a-z#]')])
    tree = p.parse(lx.tokenize("c#"))
    check("S = A '#', A = 'c' parses 'c#'", tree.name == 's')
    tree = p.parse(lx.tokenize("acb#"))
    check("'acb#' parses", tree.name == 's')
    check("'acb#' has nested a", tree.children[0].name == 'a')
    tree = p.parse(lx.tokenize("aacbb#"))
    check("'aacbb#' parses (double nesting)", tree.name == 's')

    # ---- Test 13: direct left recursion ----
    print("\n--- Test 13: direct left recursion ---")
    p = load_grammar("""
        start expr
        expr = expr '+' term | term
        term = NUMBER
    """)
    lx = Lexer([
        ('WS', r'[ \t]+'), ('NUMBER', r'\d+'), ('PLUS', r'\+'),
    ], ignore={'WS'})

    tree = p.parse(lx.tokenize("1"))
    check("lr: single number",
          tree.name == 'expr' and tree.children[0].name == 'term')

    tree = p.parse(lx.tokenize("1+2"))
    check("lr: 1+2 root is expr", tree.name == 'expr')
    # Left-associative: expr(expr(term(1)), '+', term(2))
    check("lr: 1+2 left child is expr", tree.children[0].name == 'expr')
    check("lr: 1+2 left-left is term",
          tree.children[0].children[0].name == 'term')
    check("lr: 1+2 operator is '+'", tree.children[1].value == '+')
    check("lr: 1+2 right is term", tree.children[2].name == 'term')
    print(tree.pretty())

    tree = p.parse(lx.tokenize("1+2+3"))
    check("lr: 1+2+3 root is expr", tree.name == 'expr')
    # expr(expr(expr(term(1)), '+', term(2)), '+', term(3))
    lhs = tree.children[0]
    check("lr: 1+2+3 left is expr", lhs.name == 'expr')
    check("lr: 1+2+3 left-left is expr", lhs.children[0].name == 'expr')
    check("lr: 1+2+3 left-left-left is term(1)",
          lhs.children[0].children[0].name == 'term')
    print(tree.pretty())

    # ---- Test 14: left recursion with multiple operators ----
    print("\n--- Test 14: left recursion with multiple operators ---")
    p = load_grammar("""
        start expr
        expr = expr '+' term
             | expr '-' term
             | term
        term = NUMBER
    """)
    lx = Lexer([
        ('WS', r'[ \t]+'), ('NUMBER', r'\d+'),
        ('PLUS', r'\+'), ('MINUS', r'-'),
    ], ignore={'WS'})

    tree = p.parse(lx.tokenize("1+2-3"))
    check("lr multi-op: root is expr", tree.name == 'expr')
    # expr(expr(expr(term(1)), '+', term(2)), '-', term(3))
    check("lr multi-op: outer op is '-'", tree.children[1].value == '-')
    check("lr multi-op: inner op is '+'",
          tree.children[0].children[1].value == '+')
    print(tree.pretty())

    # ---- Test 15: left recursion with mixed alternatives ----
    print("\n--- Test 15: left recursion with non-recursive alternative ---")
    p = load_grammar("""
        start expr
        expr = expr '+' term | '(' expr ')' | term
        term = NUMBER
    """)
    lx = Lexer([
        ('WS', r'[ \t]+'), ('NUMBER', r'\d+'), ('PLUS', r'\+'),
        ('LPAREN', r'\('), ('RPAREN', r'\)'),
    ], ignore={'WS'})

    tree = p.parse(lx.tokenize("(1+2)+3"))
    check("lr mixed: parses (1+2)+3", tree.name == 'expr')
    # outer: expr(expr((...)), '+', term(3))
    check("lr mixed: outer op is '+'", tree.children[1].value == '+')
    # inner: expr( '(' expr(expr(term(1)), '+', term(2)) ')' )
    inner = tree.children[0]
    check("lr mixed: inner is expr", inner.name == 'expr')
    print(tree.pretty())

    # ---- Test 16: indirect (mutual) left recursion — 2-rule cycle ----
    print("\n--- Test 16: mutual left recursion (2-rule cycle) ---")
    p = load_grammar("""
        start a
        a = b 'x' | 'y'
        b = a 'z' | 'w'
    """)
    lx = Lexer([('CH', r'[a-z]')])
    # a → 'y'
    tree = p.parse(lx.tokenize("y"))
    check("mutual lr: 'y' parses", tree.name == 'a')
    # a → b 'x' → 'w' 'x'
    tree = p.parse(lx.tokenize("wx"))
    check("mutual lr: 'wx' parses", tree.name == 'a')
    # a → b 'x' → a 'z' 'x' → 'y' 'z' 'x'
    tree = p.parse(lx.tokenize("yzx"))
    check("mutual lr: 'yzx' parses", tree.name == 'a')
    # a → ... → 'w' 'x' 'z' 'x'
    tree = p.parse(lx.tokenize("wxzx"))
    check("mutual lr: 'wxzx' parses", tree.name == 'a')
    # a → ... → 'y' 'z' 'x' 'z' 'x'
    tree = p.parse(lx.tokenize("yzxzx"))
    check("mutual lr: 'yzxzx' (deep nesting)", tree.name == 'a')
    # Reject invalid
    try:
        p.parse(lx.tokenize("xz"))
        check("mutual lr: 'xz' rejected", False)
    except SyntaxError:
        check("mutual lr: 'xz' rejected", True)
    print(tree.pretty())

    # ---- Test 17: indirect left recursion — 3-rule cycle ----
    print("\n--- Test 17: mutual left recursion (3-rule cycle) ---")
    p = load_grammar("""
        start a
        a = b 'x' | 'p'
        b = c 'y' | 'q'
        c = a 'z' | 'r'
    """)
    lx = Lexer([('CH', r'[a-z]')])
    # a → 'p'
    tree = p.parse(lx.tokenize("p"))
    check("3-cycle: 'p' parses", tree.name == 'a')
    # a → b 'x' → 'q' 'x'
    tree = p.parse(lx.tokenize("qx"))
    check("3-cycle: 'qx' parses", tree.name == 'a')
    # a → b 'x' → c 'y' 'x' → 'r' 'y' 'x'
    tree = p.parse(lx.tokenize("ryx"))
    check("3-cycle: 'ryx' parses", tree.name == 'a')
    # a → b 'x' → c 'y' 'x' → a 'z' 'y' 'x' → 'p' 'z' 'y' 'x'
    tree = p.parse(lx.tokenize("pzyx"))
    check("3-cycle: 'pzyx' (full cycle)", tree.name == 'a')
    # Two full cycles: a→b→c→a→b→c→a → 'p' 'z' 'y' 'x' 'z' 'y' 'x'
    tree = p.parse(lx.tokenize("pzyxzyx"))
    check("3-cycle: 'pzyxzyx' (two cycles)", tree.name == 'a')
    print(tree.pretty())

    # ---- Test 18: mutual + direct left recursion in same rule ----
    print("\n--- Test 18: mutual + direct left recursion ---")
    p = load_grammar("""
        start a
        a = a 'x' | b 'y' | 'z'
        b = a 'w' | 'v'
    """)
    lx = Lexer([('CH', r'[a-z]')])
    # a → 'z'
    tree = p.parse(lx.tokenize("z"))
    check("mutual+direct: 'z' parses", tree.name == 'a')
    # a → a 'x' → 'z' 'x'
    tree = p.parse(lx.tokenize("zx"))
    check("mutual+direct: 'zx' parses", tree.name == 'a')
    # a → b 'y' → 'v' 'y'
    tree = p.parse(lx.tokenize("vy"))
    check("mutual+direct: 'vy' parses", tree.name == 'a')
    # a → a 'x' → b 'y' 'x' → 'v' 'y' 'x'
    tree = p.parse(lx.tokenize("vyx"))
    check("mutual+direct: 'vyx' parses", tree.name == 'a')
    # a → b 'y' → a 'w' 'y' → 'z' 'w' 'y'
    tree = p.parse(lx.tokenize("zwy"))
    check("mutual+direct: 'zwy' (mutual cycle)", tree.name == 'a')
    # a → a 'x' → a 'x' 'x' → 'z' 'x' 'x'
    tree = p.parse(lx.tokenize("zxx"))
    check("mutual+direct: 'zxx' (repeated direct)", tree.name == 'a')
    print(tree.pretty())

    # ---- Test 19: NOT predicate (!) ----
    print("\n--- Test 19: NOT predicate ---")
    # !stop: stop matches any WORD, so !stop always fails → reject
    p = load_grammar("""
        start items
        items = (!stop WORD)+
        stop = WORD
    """)
    lx = Lexer([('WS', r'[ \t]+'), ('WORD', r'[a-z]+')], ignore={'WS'})
    try:
        p.parse(lx.tokenize("hello"))
        check("!stop rejects when stop matches", False)
    except SyntaxError:
        check("!stop rejects when stop matches", True)

    # Stop at a specific word value
    p = load_grammar("""
        start sentence
        sentence = (!'end' WORD)* 'end'
    """)
    lx = Lexer([('WS', r'[ \t]+'), ('WORD', r'[a-z]+')], ignore={'WS'})
    tree = p.parse(lx.tokenize("hello world end"))
    check("!: stop before 'end'", tree.name == 'sentence')
    check("!: 3 children (2 words + end)", len(tree.children) == 3)
    try:
        p.parse(lx.tokenize("hello end world"))
        check("!: 'hello end world' fails", False)
    except SyntaxError:
        check("!: 'hello end world' fails", True)

    # ---- Test 20: AND predicate (&) ----
    print("\n--- Test 20: AND predicate ---")
    p = load_grammar("""
        start g
        g = &'hello' WORD WORD
    """)
    lx = Lexer([('WS', r'[ \t]+'), ('WORD', r'[a-z]+')], ignore={'WS'})
    tree = p.parse(lx.tokenize("hello world"))
    check("&: passes when first word is 'hello'", tree.name == 'g')
    try:
        p.parse(lx.tokenize("other world"))
        check("&: 'other world' rejected", False)
    except SyntaxError:
        check("&: 'other world' rejected", True)

    # ---- Test 21: nested predicates ----
    print("\n--- Test 21: nested predicates ---")
    # Double negation: !!X ≡ &X
    p = load_grammar("""
        start g
        g = !!'hello' WORD
    """)
    lx = Lexer([('WS', r'[ \t]+'), ('WORD', r'[a-z]+')], ignore={'WS'})
    tree = p.parse(lx.tokenize("hello"))
    check("!!: double neg passes for 'hello'", tree.name == 'g')
    try:
        p.parse(lx.tokenize("other"))
        check("!!: 'other' rejected", False)
    except SyntaxError:
        check("!!: 'other' rejected", True)

    # ---- Test 22: named captures ----
    print("\n--- Test 22: named captures ---")
    p = load_grammar("""
        start assign
        assign = IDENT '=' val:=(NUMBER | STRING)
    """)
    lx = Lexer([
        ('WS', r'[ \t]+'), ('IDENT', r'[a-z]+'),
        ('NUMBER', r'\d+'), ('STRING', r"'[^']*'"),
        ('EQ', r'='),
    ], ignore={'WS'})
    tree = p.parse(lx.tokenize("x = 42"))
    check("capture: root is assign", tree.name == 'assign')
    # Should have a child named 'val'
    val_children = [c for c in tree.children if c.name == 'val']
    check("capture: has 'val' child", len(val_children) == 1)
    check("capture: val contains NUMBER",
          val_children[0].children[0].value == '42')
    print(tree.pretty())

    tree = p.parse(lx.tokenize("y = 'hello'"))
    val_children = [c for c in tree.children if c.name == 'val']
    check("capture: val for string",
          val_children[0].children[0].value == "'hello'")

    # ---- Test 23: inline regex auto-generates tokens ----
    print("\n--- Test 23: inline regex ---")
    p = load_grammar(r"""
        start expr
        expr = /[0-9]+/ '+' /[0-9]+/
    """)
    tree = p.parse("12+34")
    check("inline regex: parses '12+34'", tree.name == 'expr')
    check("inline regex: 3 children", len(tree.children) == 3)
    check("inline regex: first is 12", tree.children[0].value == '12')
    check("inline regex: last is 34", tree.children[2].value == '34')
    print(tree.pretty())

    # ---- Test 24: token-shaped rules ----
    print("\n--- Test 24: token-shaped rules ---")
    p = load_grammar(r"""
        start expr
        NUMBER = /[0-9]+/
        expr = NUMBER '+' NUMBER
    """)
    tree = p.parse("12+34")
    check("token-shaped: parses", tree.name == 'expr')
    check("token-shaped: NUMBER type",
          tree.children[0].name == 'NUMBER')

    # ---- Test 25: skip marker ----
    print("\n--- Test 25: skip marker (whitespace) ---")
    p = load_grammar(r"""
        start expr
        _ws = /[ \t]+/ @skip
        NUMBER = /[0-9]+/
        expr = NUMBER '+' NUMBER
    """)
    tree = p.parse("12 + 34")
    check("skip: whitespace ignored", tree.name == 'expr')
    check("skip: parse tree has no ws", len(tree.children) == 3)
    tree = p.parse("12+34")
    check("skip: works with no whitespace", tree.name == 'expr')
    tree = p.parse("12      +    34")
    check("skip: multiple whitespace ignored", tree.name == 'expr')
    print(tree.pretty())

    # ---- Test 26: skip visible when explicitly referenced ----
    print("\n--- Test 26: skip visible when referenced ---")
    # LAMBDA_L-style: keyword must be followed by whitespace
    p = load_grammar(r"""
        start prog
        _ws = /[ \t]+/ @skip
        LET = /let/
        IDENT = /[a-zA-Z_][a-zA-Z_0-9]*/
        EQ = /=/
        NUMBER = /[0-9]+/
        prog = LET _ws IDENT EQ NUMBER
    """)
    tree = p.parse("let x=42")
    check("skip-referenced: 'let x=42' parses", tree.name == 'prog')
    try:
        p.parse("letx=42")
        check("skip-referenced: 'letx=42' rejected", False)
    except SyntaxError:
        check("skip-referenced: 'letx=42' rejected", True)
    # Note: "letx" would likely be tokenized as single IDENT anyway,
    # but the real test is that _ws is required where explicit.

    # ---- Test 27: inline named token (name:=/regex/) ----
    print("\n--- Test 27: named inline token ---")
    p = load_grammar(r"""
        start expr
        _ws = /[ \t]+/ @skip
        expr = num:=/[0-9]+/ '+' num:=/[0-9]+/
    """)
    tree = p.parse("12 + 34")
    check("named inline: root is expr", tree.name == 'expr')
    # Should have captured children named 'num'
    num_kids = [c for c in tree.children if c.name == 'num']
    check("named inline: 2 num captures", len(num_kids) == 2)
    print(tree.pretty())

    # ---- Test 28: end-to-end expression grammar ----
    print("\n--- Test 28: end-to-end expression grammar ---")
    p = load_grammar(r"""
        start expr
        _ws = /[ \t]+/ @skip
        NUMBER = /[0-9]+/
        expr = expr '+' term | term
        term = term '*' factor | factor
        factor = NUMBER | '(' expr ')'
    """)
    tree = p.parse("1 + 2 * 3")
    check("e2e: parses with whitespace", tree.name == 'expr')
    tree = p.parse("(1+2) * 3")
    check("e2e: parses parens", tree.name == 'expr')
    print(tree.pretty())

    # ---- Test (precedence): @left / @right / @nonassoc ----
    print("\n--- Precedence declarations ---")
    p = load_grammar(r"""
        start expr
        @left '+' '-'
        @left '*' '/'
        _ws = /[ \t]+/ @skip
        NUMBER = /[0-9]+/
        expr = expr '+' expr | expr '-' expr
             | expr '*' expr | expr '/' expr
             | NUMBER
    """)
    # 1+2*3 — top operator is `+` (lower precedence is outer)
    tree = p.parse("1+2*3")
    check("prec: 1+2*3 top operator literal is '+'",
          any(c.value == '+' for c in tree.children))
    # 1+2+3 — left-assoc, the left child should itself be an expr (1+2)
    tree = p.parse("1+2+3")
    check("prec: 1+2+3 left-assoc (top left is another expr)",
          tree.children[0].name == 'expr'
          and any(c.value == '+' for c in tree.children[0].children))

    # Right-assoc
    p = load_grammar(r"""
        start expr
        @right '^'
        NUMBER = /[0-9]+/
        expr = expr '^' expr | NUMBER
    """)
    check("prec right: 2^3^4 parses",
          p.parse('2^3^4').name == 'expr')

    # Nonassoc
    p = load_grammar(r"""
        start expr
        @nonassoc '=='
        @left '+'
        NUMBER = /[0-9]+/
        expr = expr '==' expr | expr '+' expr | NUMBER
    """)
    check("prec nonassoc: 1+2==3+4 parses",
          p.parse('1+2==3+4').name == 'expr')
    threw = False
    try: p.parse('1==2==3')
    except SyntaxError: threw = True
    check("prec nonassoc: 1==2==3 rejected", threw)

    # Token-reference operators in declarations.
    p = load_grammar(r"""
        start expr
        @left PLUS MINUS
        @left TIMES DIVIDE
        NUMBER = /[0-9]+/
        PLUS   = /\+/
        MINUS  = /-/
        TIMES  = /\*/
        DIVIDE = /\//
        expr = expr PLUS expr | expr MINUS expr
             | expr TIMES expr | expr DIVIDE expr
             | NUMBER
    """)
    tree = p.parse('1+2*3')
    # Top operator should be a PLUS token (lower precedence is outer)
    check("prec token-ref ops: 1+2*3 top op is PLUS",
          any(c.name == 'PLUS' for c in tree.children))

    # ---- Summary ----
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        print("SOME TESTS FAILED")
    else:
        print("ALL TESTS PASSED")
