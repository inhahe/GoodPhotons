// Minimal, dependency-free JSON parser (RFC 8259 subset, enough for glTF 2.0).
// Recursive-descent over a std::string; produces a Value tree of null/bool/number/
// string/array/object. Objects preserve insertion order isn't needed here, so a
// std::map keyed by string is used for O(log n) member lookup. Numbers are parsed as
// double (glTF indices fit exactly in a double's 53-bit mantissa). Not a general
// high-performance parser — just correct and small. Used only by src/gltf.h.
#pragma once
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <cstdint>

namespace minijson {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object };
    Type type = Null;
    bool        b = false;
    double      num = 0.0;
    std::string str;
    std::vector<Value>          arr;
    std::map<std::string, Value> obj;

    bool isNull()   const { return type == Null; }
    bool isObject() const { return type == Object; }
    bool isArray()  const { return type == Array; }
    bool isNumber() const { return type == Number; }
    bool isString() const { return type == String; }

    // Object member lookup; nullptr when absent or not an object.
    const Value* find(const std::string& key) const {
        if (type != Object) return nullptr;
        auto it = obj.find(key);
        return it == obj.end() ? nullptr : &it->second;
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
            v.obj.emplace(std::move(key), std::move(child));
            skipWs();
            if (i_ >= s_.size()) return fail("unterminated object");
            if (s_[i_] == ',') { ++i_; continue; }
            if (s_[i_] == '}') { ++i_; return true; }
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
