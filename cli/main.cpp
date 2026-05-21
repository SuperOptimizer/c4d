// c4d CLI — encode / decode / info / compact. Thin wrapper over the library.
// Fleshed out as the library API lands; stub for now so the build links.
#include "c4d/core.hpp"
#include <cstdio>
#include <string_view>

static int usage() {
    std::fprintf(stderr,
        "c4d — compress4d archive tool\n"
        "usage:\n"
        "  c4d info <archive.c4d>\n"
        "  c4d encode <raw_volume> <archive.c4d> --shape Z,Y,X [--q N]\n"
        "  c4d decode <archive.c4d> <raw_volume>\n"
        "  c4d compact <in.c4d> <out.c4d>\n");
    return 2;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    std::string_view cmd = argv[1];
    if (cmd == "info" || cmd == "encode" || cmd == "decode" || cmd == "compact") {
        std::fprintf(stderr, "c4d %.*s: not yet implemented\n",
                     static_cast<int>(cmd.size()), cmd.data());
        return 1;
    }
    return usage();
}
