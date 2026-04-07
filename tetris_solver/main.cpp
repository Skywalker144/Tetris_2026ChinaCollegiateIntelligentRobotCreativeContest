#include "mcts_parallel.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct RGB {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

std::string trim(const std::string& s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

char parsePieceToken(const std::string& token) {
    if (token.size() != 1) {
        throw std::invalid_argument("Piece token must be one character");
    }
    const char piece = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
    if (Tetris::PIECE_ROTATIONS.find(piece) == Tetris::PIECE_ROTATIONS.end()) {
        throw std::invalid_argument("Unknown piece token: " + token);
    }
    return piece;
}

std::unordered_map<char, int> parseCounts(const std::string& text) {
    std::unordered_map<char, int> counts;
    if (text.empty()) {
        return counts;
    }

    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }

        const std::size_t eq = item.find('=');
        if (eq == std::string::npos) {
            throw std::invalid_argument("Invalid counts item: " + item);
        }

        const char piece = parsePieceToken(trim(item.substr(0, eq)));
        int count = std::stoi(trim(item.substr(eq + 1)));
        if (count < 0 || count > 5) {
            throw std::invalid_argument("Count must be in [0, 5]");
        }
        counts[piece] = count;
    }

    return counts;
}

std::vector<char> parseOrder(const std::string& text) {
    std::vector<char> order;
    if (text.empty()) {
        return order;
    }

    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }
        order.push_back(parsePieceToken(item));
    }

    return order;
}

std::vector<char> parseSequence(const std::string& text) {
    std::vector<char> sequence;
    if (text.empty()) {
        return sequence;
    }

    std::unordered_map<char, int> freq;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (item.empty()) {
            continue;
        }
        char piece = parsePieceToken(item);
        freq[piece] += 1;
        if (freq[piece] > 5) {
            throw std::invalid_argument("Each piece can appear at most 5 times");
        }
        sequence.push_back(piece);
    }

    return sequence;
}

std::vector<char> buildSequenceFromOrderAndCounts(
    const std::vector<char>& order,
    const std::unordered_map<char, int>& counts
) {
    if (order.empty()) {
        throw std::invalid_argument("Order is required when SEQUENCE is empty");
    }

    std::vector<char> sequence;
    for (char piece : order) {
        auto it = counts.find(piece);
        const int count = it == counts.end() ? 0 : it->second;
        if (count > 5) {
            throw std::invalid_argument("Count for a piece cannot exceed 5");
        }
        for (int i = 0; i < count; ++i) {
            sequence.push_back(piece);
        }
    }
    return sequence;
}

void fillRect(
    std::vector<RGB>& pixels,
    int img_w,
    int img_h,
    int x0,
    int y0,
    int x1,
    int y1,
    const RGB& color
) {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(img_w, x1);
    y1 = std::min(img_h, y1);
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            pixels[y * img_w + x] = color;
        }
    }
}

void drawHorizontal(
    std::vector<RGB>& pixels,
    int img_w,
    int img_h,
    int x0,
    int x1,
    int y,
    int thickness,
    const RGB& color
) {
    fillRect(pixels, img_w, img_h, x0, y, x1, y + thickness, color);
}

void drawVertical(
    std::vector<RGB>& pixels,
    int img_w,
    int img_h,
    int x,
    int y0,
    int y1,
    int thickness,
    const RGB& color
) {
    fillRect(pixels, img_w, img_h, x, y0, x + thickness, y1, color);
}

void renderBoard(
    const std::vector<std::vector<int>>& board,
    const std::vector<std::vector<int>>& piece_ids,
    int full_rows,
    const std::string& output_path
) {
    const int h = static_cast<int>(board.size());
    const int w = h > 0 ? static_cast<int>(board[0].size()) : 0;
    const int tile = 28;
    const int margin = 1;

    const int img_w = w * tile + margin;
    const int img_h = h * tile + margin;

    const std::array<RGB, 8> cmap = {
        RGB{0xf4, 0xf1, 0xde},
        RGB{0xf1, 0xc4, 0x0f},
        RGB{0x8e, 0x44, 0xad},
        RGB{0x2e, 0xcc, 0x71},
        RGB{0x29, 0x80, 0xb9},
        RGB{0x8b, 0x5a, 0x2b},
        RGB{0xe6, 0x7e, 0x22},
        RGB{0xe7, 0x4c, 0x3c},
    };

    const RGB grid_color{0xb7, 0xc0, 0xcd};
    const RGB border_color{0x00, 0x00, 0x00};

    std::vector<RGB> pixels(img_w * img_h, cmap[0]);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int v = board[y][x];
            if (v < 0) {
                v = 0;
            }
            if (v >= static_cast<int>(cmap.size())) {
                v = static_cast<int>(cmap.size()) - 1;
            }
            fillRect(
                pixels,
                img_w,
                img_h,
                x * tile + 1,
                y * tile + 1,
                (x + 1) * tile,
                (y + 1) * tile,
                cmap[v]
            );
        }
    }

    for (int x = 0; x <= w; ++x) {
        drawVertical(pixels, img_w, img_h, x * tile, 0, img_h, 1, grid_color);
    }
    for (int y = 0; y <= h; ++y) {
        drawHorizontal(pixels, img_w, img_h, 0, img_w, y * tile, 1, grid_color);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int pid = piece_ids[y][x];
            if (pid == 0) {
                continue;
            }
            if (y == 0 || piece_ids[y - 1][x] != pid) {
                drawHorizontal(pixels, img_w, img_h, x * tile, (x + 1) * tile + 1, y * tile, 2, border_color);
            }
            if (y == h - 1 || piece_ids[y + 1][x] != pid) {
                drawHorizontal(
                    pixels,
                    img_w,
                    img_h,
                    x * tile,
                    (x + 1) * tile + 1,
                    (y + 1) * tile - 1,
                    2,
                    border_color
                );
            }
            if (x == 0 || piece_ids[y][x - 1] != pid) {
                drawVertical(pixels, img_w, img_h, x * tile, y * tile, (y + 1) * tile + 1, 2, border_color);
            }
            if (x == w - 1 || piece_ids[y][x + 1] != pid) {
                drawVertical(
                    pixels,
                    img_w,
                    img_h,
                    (x + 1) * tile - 1,
                    y * tile,
                    (y + 1) * tile + 1,
                    2,
                    border_color
                );
            }
        }
    }

    std::filesystem::path out(output_path);
    if (out.has_parent_path()) {
        std::filesystem::create_directories(out.parent_path());
    }

    std::ofstream ofs(output_path, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open output image: " + output_path);
    }

    ofs << "P6\n" << img_w << " " << img_h << "\n255\n";
    for (const RGB& px : pixels) {
        ofs.write(reinterpret_cast<const char*>(&px), 3);
    }

    std::cout << "Final full rows: " << full_rows << "\n";
}

void runGame(
    const std::vector<char>& sequence,
    int simulations,
    double exploration_constant,
    int num_threads,
    double virtual_loss,
    bool normalize_reward,
    const std::string& output_path
) {
    Tetris game(10, 14, sequence);
    TetrisState state = game.getInitialState();

    MCTSParallel::Args args;
    args.num_simulations = simulations;
    args.exploration_constant = exploration_constant;
    args.normalize_reward = normalize_reward;
    args.num_threads = num_threads;
    args.virtual_loss = virtual_loss;

    MCTSParallel mcts(game, args);

    int placed = 0;
    while (!game.isTerminal(state)) {
        std::optional<Action> action = mcts.search(state);
        if (!action.has_value()) {
            break;
        }
        state = game.getNextState(state, *action);
        ++placed;
    }

    renderBoard(state.board, state.piece_ids, state.score, output_path);

    std::cout << "Piece sequence: [";
    for (std::size_t i = 0; i < sequence.size(); ++i) {
        std::cout << sequence[i] << (i + 1 == sequence.size() ? "" : ",");
    }
    std::cout << "]\n";
    std::cout << "Placed pieces: " << placed << " / " << sequence.size() << "\n";
    std::cout << "Image saved to: " << output_path << "\n";
}

}

int main() {
    const std::string SEQUENCE = "";
    const std::string ORDER = "S,J,Z,L,T,O,I";
    const std::string COUNTS = "L=3,J=2,S=3,Z=1,T=2,O=3,I=1";

    const int NUM_SIMULATIONS = 1000000;
    const double C = 1.41;
    const int NUM_THREADS = 48;
    const double VIRTUAL_LOSS = 0.3;
    const bool NORMALIZE_REWARD = true;
    const std::string OUTPUT_IMAGE = "output/final_board.ppm";

    try {
        std::vector<char> sequence;
        if (!SEQUENCE.empty()) {
            sequence = parseSequence(SEQUENCE);
        } else {
            const std::vector<char> order = parseOrder(ORDER);
            const std::unordered_map<char, int> counts = parseCounts(COUNTS);
            sequence = buildSequenceFromOrderAndCounts(order, counts);
        }

        if (sequence.empty()) {
            throw std::invalid_argument("Final sequence is empty. Fill SEQUENCE or set ORDER+COUNTS.");
        }

        runGame(
            sequence,
            NUM_SIMULATIONS,
            C,
            NUM_THREADS,
            VIRTUAL_LOSS,
            NORMALIZE_REWARD,
            OUTPUT_IMAGE
        );
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
