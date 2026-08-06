// Minimal, dependency-free JSON parser (RFC 8259 subset, enough for glTF 2.0).
// Recursive-descent over a std::string; produces a Value tree of null/bool/number/
// string/array/object. Insertion order is not needed, so members are held in a
// key-SORTED vector giving O(log n) lookup by binary search. Numbers are parsed as
// double (glTF indices fit exactly in a double's 53-bit mantissa). Used by
// src/gltf.h (glTF/GLB), src/viewer_gui.cpp (the loom bridge protocol) and
// src/curvedrive.h (the loom CurveDrive sidecar).
//
// PERFORMANCE — why members are a sorted vector and NOT a std::map.
// std::vector only *moves* its elements when reallocating if the element type is
// nothrow-move-constructible; otherwise it must *copy* them, to preserve push_back's
// strong exception guarantee. A JSON Value copies DEEPLY, so a single non-noexcept
// member type turns every array growth into a full recursive clone of everything
// parsed so far — quadratic, and catastrophic on the big numeric arrays these
// sidecars are mostly made of. `std::map`'s move constructor is NOT noexcept on MSVC
// (it may allocate a sentinel node), and that alone poisoned Value's implicit move:
// measured 0.86 MB at ~5 MB/s, ~170 ms, which was ~96% of the viewer's per-frame
// sidecar adoption cost and roughly half of a played frame.
// std::string and std::vector both move noexcept, so with the map gone Value's
// implicit move is noexcept *honestly* — the guarantee is a property of the members
// rather than an assertion we could not keep. Do not reintroduce a std::map here, and
// if you add a member, keep it nothrow-movable (tools/jsonbench.cpp measures this).
#pragma once
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

namespace minijson {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool        b = false;
    double      num = 0.0;
    std::string str;
    std::vector<Value>          arr;
    // Key-sorted, duplicate-free; see the performance note at the top of this file.
    // `kv.first` / `kv.second` still work, so range-for over `obj` is unchanged.
    using Member = std::pair<std::string, Value>;
    std::vector<Member>         obj;

    bool isNull()   const { return type == Null; }
    bool isObject() const { return type == Object; }
    bool isArray()  const { return type == Array; }
    bool isNumber() const { return type == Number; }
    bool isString() const { return type == String; }

    // Object member lookup; nullptr when absent or not an object. `obj` is kept
    // key-sorted by parseObject, so this is the same O(log n) the std::map gave.
    const Value* find(const std::string& key) const {
        if (type != Object) return nullptr;
        auto it = std::lower_bound(obj.begin(), obj.end(), key,
                                   [](const Member& m, const std::string& k) {
                                       return m.first < k;
                                   });
        return (it == obj.end() || it->first != key) ? nullptr : &it->second;
    }
    // Typed accessors with defaults (safe on any node type).
    double asNumber(double d = 0.0) const { return type == Number ? num : d; }
    int    asInt(int d = 0)         const { return type == Number ? (int)num : d; }
    bool   asBool(bool d = false)   const { return type == Bool ? b : d; }
    const std::string& asString(const std::string& d) const { return type == String ? str : d; }

    // Convenience: object member as number/int with a default.
    double numAt(const std::string& key, double d = 0.0) const {
        const Value* v = find(key); return v ? v->asNumber(d) : d;
    }
    int intAt(const std::string& key, int d = 0) const {
        const Value* v = find(key); return v ? v->asInt(d) : d;
    }
};

// The whole performance note at the top of this file rests on this one property, and
// nothing about a std::map member LOOKS slow — which is precisely why the regression
// went unnoticed for so long. Assert it, so reintroducing a throwing-move member is a
// compile error rather than a silent ~18x parse slowdown.
static_assert(std::is_nothrow_move_constructible<Value>::value,
              "minijson::Value must be nothrow-move-constructible, or std::vector will "
              "deep-COPY every element on reallocation (see the note at the top).");

class Parser {
public:
    Parser(const std::string& s) : s_(s), i_(0) {}
    bool parse(Value& out, std::string& err) {
        skipWs();
        if (!parseValue(out)) { err = err_.empty() ? "JSON parse error" : err_; return false; }
        skipWs();
        return true;   // trailing content tolerated (GLB pads the JSON chunk with spaces)
    }
private:
    const std::string& s_;
    size_t i_;
    std::string err_;

    void skipWs() {
        while (i_ < s_.size()) {
            char c = s_[i_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') ++i_;
            else break;
        }
    }
    bool fail(const std::string& m) { if (err_.empty()) err_ = m; return false; }

    bool parseValue(Value& v) {
        skipWs();
        if (i_ >= s_.size()) return fail("unexpected end");
        char c = s_[i_];
        switch (c) {
            case '{': return parseObject(v);
            case '[': return parseArray(v);
            case '"': { v.type = Value::String; return parseString(v.str); }
            case 't': case 'f': return parseBool(v);
            case 'n': return parseNull(v);
            default:  return parseNumber(v);
        }
    }
    // Put an object's members in key order so find() can binary-search, and drop
    // duplicate keys. stable_sort + unique keeps the FIRST occurrence of a repeated
    // key, which is exactly what the previous std::map::emplace did.
    static void sortMembers(Value& v) {
        std::stable_sort(v.obj.begin(), v.obj.end(),
                         [](const Value::Member& a, const Value::Member& b) {
                             return a.first < b.first;
                         });
        v.obj.erase(std::unique(v.obj.begin(), v.obj.end(),
                                [](const Value::Member& a, const Value::Member& b) {
                                    return a.first == b.first;
                                }),
                    v.obj.end());
    }
    bool parseObject(Value& v) {
        v.type = Value::Object;
        ++i_;  // '{'
        skipWs();
        if (i_ < s_.size() && s_[i_] == '}') { ++i_; return true; }
        while (true) {
            skipWs();
            if (i_ >= s_.size() || s_[i_] != '"') return fail("expected key string");
            std::string key;
            if (!parseString(key)) return false;
            skipWs();
            if (i_ >= s_.size() || s_[i_] != ':') return fail("expected ':'");
            ++i_;
            Value child;
            if (!parseValue(child)) return false;
            // Append now, sort once on close (below): inserting in order would be O(n)
            // per member, i.e. quadratic on a wide object.
            v.obj.emplace_back(std::move(key), std::move(child));
            skipWs();
            if (i_ >= s_.size()) return fail("unterminated object");
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == '}') { ++i_; sortMembers(v); return true; }
            return fail("expected ',' or '}'");
        }
    }
    bool parseArray(Value& v) {
        v.type = Value::Array;
        ++i_;  // '['
        skipWs();
        if (i_ < s_.size() && s_[i_] == ']') { ++i_; return true; }
        while (true) {
            Value child;
            if (!parseValue(child)) return false;
            v.arr.push_back(std::move(child));
            skipWs();
            if (i_ >= s_.size()) return fail("unterminated array");
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == ']') { ++i_; return true; }
            return fail("expected ',' or ']'");
        }
    }
    bool parseString(std::string& out) {
        ++i_;  // opening quote
        out.clear();
        while (i_ < s_.size()) {
            char c = s_[i_++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i_ >= s_.size()) break;
                char e = s_[i_++];
                switch (e) {
                    case '"':  out += '"';  break;
                    case '\\': out += '\\'; break;
                    case '/':  out += '/';  break;
                    case 'b':  out += '\b'; break;
                    case 'f':  out += '\f'; break;
                    case 'n':  out += '\n'; break;
                    case 'r':  out += '\r'; break;
                    case 't':  out += '\t'; break;
                    case 'u': {
                        if (i_ + 4 > s_.size()) return fail("bad \\u escape");
                        unsigned code = 0;
                        for (int k = 0; k < 4; ++k) {
                            char h = s_[i_++]; code <<= 4;
                            if (h >= '0' && h <= '9') code |= (h - '0');
                            else if (h >= 'a' && h <= 'f') code |= (h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= (h - 'A' + 10);
                            else return fail("bad hex in \\u");
                        }
                        // Minimal UTF-8 encode of the BMP code point (surrogate pairs
                        // are not needed for glTF names/keys in practice).
                        if (code < 0x80) out += (char)code;
                        else if (code < 0x800) {
                            out += (char)(0xC0 | (code >> 6));
                            out += (char)(0x80 | (code & 0x3F));
                        } else {
                            out += (char)(0xE0 | (code >> 12));
                            out += (char)(0x80 | ((code >> 6) & 0x3F));
                            out += (char)(0x80 | (code & 0x3F));
                        }
                        break;
                    }
                    default: return fail("bad escape");
                }
            } else {
                out += c;
            }
        }
        return fail("unterminated string");
    }
    bool parseBool(Value& v) {
        if (s_.compare(i_, 4, "true") == 0)  { v.type = Value::Bool; v.b = true;  i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { v.type = Value::Bool; v.b = false; i_ += 5; return true; }
        return fail("bad literal");
    }
    bool parseNull(Value& v) {
        if (s_.compare(i_, 4, "null") == 0) { v.type = Value::Null; i_ += 4; return true; }
        return fail("bad literal");
    }
    bool parseNumber(Value& v) {
        size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        bool any = false;
        while (i_ < s_.size()) {
            char c = s_[i_];
            if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                c == '+' || c == '-') { ++i_; any = true; }
            else break;
        }
        if (!any) return fail("bad number");
        v.type = Value::Number;
        v.num = std::strtod(s_.c_str() + start, nullptr);
        return true;
    }
};

inline bool parse(const std::string& text, Value& out, std::string& err) {
    Parser p(text);
    return p.parse(out, err);
}

}  // namespace minijson
