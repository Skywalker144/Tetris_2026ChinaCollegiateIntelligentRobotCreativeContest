#include "mcts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

MCTS::MCTS(Tetris& game, Args args, std::optional<unsigned int> seed)
    : game_(game),
      args_(args),
      reward_scale_(std::max(1.0, static_cast<double>(game_.getMaxReward()))),
      rng_(seed.has_value() ? *seed : std::random_device{}()) {}

bool MCTS::Node::isFullyExpanded(Tetris& game) {
    if (!untried_initialized) {
        untried_actions = game.getLegalActions(state);
        untried_initialized = true;
    }
    return untried_actions.empty();
}

double MCTS::Node::getUcb(double exploration_constant) const {
    if (n == 0) {
        return std::numeric_limits<double>::infinity();
    }

    const double q = v / static_cast<double>(n);
    const int parent_visits = std::max(parent ? parent->n : 0, 1);
    const double u = exploration_constant * std::sqrt(std::log(static_cast<double>(parent_visits)) / n);
    return q + u;
}

void MCTS::Node::update(double value) {
    v += value;
    n += 1;
}

std::optional<Action> MCTS::search(const TetrisState& state) {
    if (game_.isTerminal(state)) {
        return std::nullopt;
    }

    Node root(game_.cloneState(state));

    for (int i = 0; i < args_.num_simulations; ++i) {
        Node* node = &root;

        while (!game_.isTerminal(node->state) && node->isFullyExpanded(game_) && !node->children.empty()) {
            node = select(node);
        }

        if (!game_.isTerminal(node->state) && !node->isFullyExpanded(game_)) {
            node = expand(node);
        }

        const double value = simulate(node->state);
        backpropagate(node, value);
    }

    if (root.children.empty()) {
        return std::nullopt;
    }

    auto best_it = std::max_element(
        root.children.begin(),
        root.children.end(),
        [](const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) { return a->n < b->n; }
    );
    return (*best_it)->action_taken;
}

MCTS::Node* MCTS::select(Node* node) {
    auto best_it = std::max_element(
        node->children.begin(),
        node->children.end(),
        [this](const std::unique_ptr<Node>& a, const std::unique_ptr<Node>& b) {
            return a->getUcb(args_.exploration_constant) < b->getUcb(args_.exploration_constant);
        }
    );
    return best_it->get();
}

MCTS::Node* MCTS::expand(Node* node) {
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node->untried_actions.size()) - 1);
    int action_idx = dist(rng_);
    Action action = node->untried_actions[action_idx];

    node->untried_actions[action_idx] = node->untried_actions.back();
    node->untried_actions.pop_back();

    TetrisState next_state = game_.getNextState(node->state, action);
    node->children.push_back(std::make_unique<Node>(std::move(next_state), action, node));
    return node->children.back().get();
}

double MCTS::simulate(const TetrisState& state) {
    TetrisState current_state = game_.cloneState(state);

    while (!game_.isTerminal(current_state)) {
        std::vector<Action> legal_actions = game_.getLegalActions(current_state);
        if (legal_actions.empty()) {
            break;
        }
        std::uniform_int_distribution<int> dist(0, static_cast<int>(legal_actions.size()) - 1);
        const Action action = legal_actions[dist(rng_)];
        current_state = game_.getNextState(current_state, action);
    }

    const double reward = game_.getReward(current_state);
    if (args_.normalize_reward) {
        return reward / static_cast<double>(reward_scale_);
    }
    return reward;
}

void MCTS::backpropagate(Node* node, double value) {
    while (node != nullptr) {
        node->update(value);
        node = node->parent;
    }
}
