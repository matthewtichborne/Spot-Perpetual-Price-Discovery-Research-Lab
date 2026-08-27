#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace spot_perp_lab {

struct MarketView {
    std::span<const std::int64_t> event_time_ns;
    std::span<const std::int64_t> aggregate_trade_id;
    std::span<const double> price;
    std::span<const double> quantity;
    std::span<const double> notional;
    std::span<const double> signed_quantity;
    std::span<const double> signed_notional;
    std::span<const bool> is_buyer_maker;
};

struct MarketBars {
    std::vector<double> last_price;
    std::vector<double> quantity;
    std::vector<double> notional;
    std::vector<double> signed_quantity;
    std::vector<double> signed_notional;
    std::vector<std::int64_t> trade_count;
    std::vector<std::int64_t> buyer_trade_count;
    std::vector<std::int64_t> seller_trade_count;
};

struct ReplayResult {
    std::vector<std::int64_t> decision_time_ns;
    MarketBars spot;
    MarketBars perpetual;
};

ReplayResult replay_two_markets(
    const MarketView& spot,
    const MarketView& perpetual,
    std::int64_t start_ns,
    std::int64_t end_ns,
    std::int64_t interval_ns
);

}  // namespace spot_perp_lab
