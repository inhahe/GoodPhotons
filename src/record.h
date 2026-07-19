// Parametric records — a named FTSL data structure (see ROADMAP_records.md).
//
// A record is a bank of per-channel look-up tables over a shared scalar domain
// [lo,hi]. A per-hit driver scalar samples every channel at once; each channel whose
// name matches a real material slot fills that slot at the driven value. Channel
// names are arbitrary (name-by-destination): a name that matches no slot is simply
// not auto-bound (still addressable by dot), and a slot with no channel is ignored.
//
// STAGE 1 (this file): the STRUCTURAL model only — channels, their stops, and each
// stop's redistributed domain position + raw value token. Compiling stop tokens into
// scalar programs / colours, and the sampling function, land in later stages.
#pragma once
#include <string>
#include <vector>

enum class RecInterp { Nearest, Linear, Smooth };

// One stop in a channel LUT.
struct RecStop {
    double      pos    = 0.0;    // domain position in [lo,hi] after redistribution
    bool        pinned = false;  // author gave an explicit p:<pos> prefix
    std::string token;           // raw value token (number / expression / spectrum:ref)
};

// One channel: a named LUT (its name is matched to a material slot at bind time).
struct RecChannel {
    std::string          name;
    std::vector<RecStop> stops;  // author order == ascending pos after redistribution
};

struct Record {
    std::string              name;
    double                   lo = 0.0, hi = 1.0;
    RecInterp                interp = RecInterp::Linear;
    std::vector<RecChannel>  channels;

    int channelIndex(const std::string& n) const {
        for (size_t i = 0; i < channels.size(); ++i)
            if (channels[i].name == n) return (int)i;
        return -1;
    }
};
