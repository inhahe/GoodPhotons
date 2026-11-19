// THE MERGE LATTICE OF MODE U (vertex connection and merging): the dense grid of cells a pass bins
// its light vertices into, shared by the host's VcmGrid (vcm.h) and the device session's grid
// (render_cuda.cu, vcmSessionPass) so that the two size it by one rule.
#pragma once
#include <algorithm>
#include <cmath>

// The most cells a lattice takes: 64 M, 256 MB of int offsets, and cell indices are ints.
constexpr long long kVcmMaxCells = 1LL << 26;

// A lattice: its corner `lo`, stored as S (the vector type's scalar the caller keeps it in, so a
// float device grid rounds it as it always has), its cell width and its cells a side.
template <class S>
struct VcmLattice {
    S lo[3];
    double cell;
    int n[3];
    long long cells() const { return (long long)n[0] * n[1] * n[2]; }
};

// The lattice over the box [mn, mx] of a pass's light vertices for cells `cell` wide (the merge
// radius): the box padded half a cell each side, at most 2e6 cells an axis, and past `maxCells` cells
// in all the cells grown until it holds them -- a merge tests distance against the radius, not the
// cell, so a cell wider than the radius only adds candidates, where a dense lattice at a fine radius
// would ask for gigabytes and wrap the int index. Each growth widens the cells by the cube root of
// the excess (and 1 %), which a box flat along an axis or held at the axis clamp shrinks by less than
// that, so it goes on until the lattice holds; it always gets there, the cells widening at least 1 %
// each time.
template <class S>
inline VcmLattice<S> vcmLattice(const double mn[3], const double mx[3], double cell,
                                long long maxCells = kVcmMaxCells) {
    VcmLattice<S> g;
    for (;;) {
        for (int a = 0; a < 3; ++a) {
            g.lo[a] = (S)(mn[a] - cell * 0.5);
            const S ext = (S)(mx[a] - (double)g.lo[a] + cell * 0.5);
            g.n[a] = std::max(1, (int)std::ceil(std::min((double)ext / cell, 2e6)));
        }
        g.cell = cell;
        if (g.cells() <= maxCells) return g;
        cell *= std::cbrt((double)g.cells() / (double)maxCells) * 1.01;
    }
}
