#include "spot_perp_lab/replay.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace spot_perp_lab {
namespace {

void validate_market(const MarketView& market, std::int64_t start_ns, std::int64_t end_ns) {
    const auto size = market.event_time_ns.size();
    if (market.aggregate_trade_id.size() != size || market.price.size() != size ||
        market.quantity.size() != size || market.notional.size() != size ||
        market.signed_quantity.size() != size || market.signed_notional.size() != size ||
        market.is_buyer_maker.size() != size) {
        throw std::invalid_argument("market event columns must have equal lengths");
    }
    if (!std::is_sorted(market.event_time_ns.begin(), market.event_time_ns.end())) {
        throw std::invalid_argument("market event timestamps must be non-decreasing");
    }
    if (size > 0 && (market.event_time_ns.front() < start_ns ||
                     market.event_time_ns.back() >= end_ns)) {
        throw std::invalid_argument("market event timestamp falls outside replay grid");
    }
}

MarketBars empty_bars(std::size_t bars) {
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    return {
        std::vector<double>(bars, nan),
        std::vector<double>(bars, 0.0),
        std::vector<double>(bars, 0.0),
        std::vector<double>(bars, 0.0),
        std::vector<double>(bars, 0.0),
        std::vector<std::int64_t>(bars, 0),
        std::vector<std::int64_t>(bars, 0),
        std::vector<std::int64_t>(bars, 0),
    };
}

void add_event(
    const MarketView& market,
    std::size_t event,
    MarketBars& bars,
    std::int64_t start_ns,
    std::int64_t interval_ns
) {
    const auto bucket = static_cast<std::size_t>(
        (market.event_time_ns[event] - start_ns) / interval_ns
    );
    bars.last_price[bucket] = market.price[event];
    bars.quantity[bucket] += market.quantity[event];
    bars.notional[bucket] += market.notional[event];
    bars.signed_quantity[bucket] += market.signed_quantity[event];
    bars.signed_notional[bucket] += market.signed_notional[event];
    ++bars.trade_count[bucket];
    if (market.is_buyer_maker[event]) {
        ++bars.seller_trade_count[bucket];
    } else {
        ++bars.buyer_trade_count[bucket];
    }
}

void forward_fill(std::vector<double>& prices) {
    auto last = std::numeric_limits<double>::quiet_NaN();
    for (auto& price : prices) {
        if (std::isnan(price)) {
            price = last;
        } else {
            last = price;
        }
    }
}

}  // namespace

ReplayResult replay_two_markets(
    const MarketView& spot,
    const MarketView& perpetual,
    std::int64_t start_ns,
    std::int64_t end_ns,
    std::int64_t interval_ns
) {
    if (interval_ns <= 0 || end_ns <= start_ns || (end_ns - start_ns) % interval_ns != 0) {
        throw std::invalid_argument("grid bounds must define positive whole intervals");
    }
    validate_market(spot, start_ns, end_ns);
    validate_market(perpetual, start_ns, end_ns);
    const auto bar_count = static_cast<std::size_t>((end_ns - start_ns) / interval_ns);
    ReplayResult result;
    result.decision_time_ns.resize(bar_count);
    for (std::size_t index = 0; index < bar_count; ++index) {
        result.decision_time_ns[index] =
            start_ns + static_cast<std::int64_t>(index + 1) * interval_ns;
    }
    result.spot = empty_bars(bar_count);
    result.perpetual = empty_bars(bar_count);

    std::size_t spot_index = 0;
    std::size_t perpetual_index = 0;
    while (spot_index < spot.event_time_ns.size() ||
           perpetual_index < perpetual.event_time_ns.size()) {
        const bool take_spot =
            perpetual_index >= perpetual.event_time_ns.size() ||
            (spot_index < spot.event_time_ns.size() &&
             spot.event_time_ns[spot_index] <= perpetual.event_time_ns[perpetual_index]);
        if (take_spot) {
            add_event(spot, spot_index, result.spot, start_ns, interval_ns);
            ++spot_index;
        } else {
            add_event(perpetual, perpetual_index, result.perpetual, start_ns, interval_ns);
            ++perpetual_index;
        }
    }
    forward_fill(result.spot.last_price);
    forward_fill(result.perpetual.last_price);
    return result;
}

}  // namespace spot_perp_lab
