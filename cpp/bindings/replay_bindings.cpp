#include "spot_perp_lab/replay.hpp"

#include <cstring>
#include <span>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using spot_perp_lab::MarketBars;
using spot_perp_lab::MarketView;
using spot_perp_lab::ReplayResult;

namespace {

template <typename T>
std::span<const T> as_span(const py::array_t<T, py::array::c_style | py::array::forcecast>& array) {
    return {array.data(), static_cast<std::size_t>(array.size())};
}

template <typename T>
py::array_t<T> copy_array(const std::vector<T>& values) {
    py::array_t<T> output(values.size());
    if (!values.empty()) {
        std::memcpy(output.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return output;
}

void add_market(py::dict& output, const std::string& prefix, const MarketBars& bars) {
    output[py::str(prefix + "_last_price")] = copy_array(bars.last_price);
    output[py::str(prefix + "_quantity")] = copy_array(bars.quantity);
    output[py::str(prefix + "_notional")] = copy_array(bars.notional);
    output[py::str(prefix + "_signed_quantity")] = copy_array(bars.signed_quantity);
    output[py::str(prefix + "_signed_notional")] = copy_array(bars.signed_notional);
    output[py::str(prefix + "_trade_count")] = copy_array(bars.trade_count);
    output[py::str(prefix + "_buyer_trade_count")] = copy_array(bars.buyer_trade_count);
    output[py::str(prefix + "_seller_trade_count")] = copy_array(bars.seller_trade_count);
}

}  // namespace

PYBIND11_MODULE(_replay, module) {
    module.doc() = "C++20 two-market fixed-grid replay kernel";
    module.def("compiler_version", []() { return std::string(__VERSION__); });
    module.def(
        "replay_two_markets",
        [](
            const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& spot_time,
            const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& spot_id,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& spot_price,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& spot_quantity,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& spot_notional,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& spot_signed_quantity,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& spot_signed_notional,
            const py::array_t<bool, py::array::c_style | py::array::forcecast>& spot_maker,
            const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& perp_time,
            const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>& perp_id,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& perp_price,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& perp_quantity,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& perp_notional,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& perp_signed_quantity,
            const py::array_t<double, py::array::c_style | py::array::forcecast>& perp_signed_notional,
            const py::array_t<bool, py::array::c_style | py::array::forcecast>& perp_maker,
            std::int64_t start_ns,
            std::int64_t end_ns,
            std::int64_t interval_ns
        ) {
            const MarketView spot{
                as_span(spot_time), as_span(spot_id), as_span(spot_price),
                as_span(spot_quantity), as_span(spot_notional), as_span(spot_signed_quantity),
                as_span(spot_signed_notional), as_span(spot_maker),
            };
            const MarketView perpetual{
                as_span(perp_time), as_span(perp_id), as_span(perp_price),
                as_span(perp_quantity), as_span(perp_notional), as_span(perp_signed_quantity),
                as_span(perp_signed_notional), as_span(perp_maker),
            };
            ReplayResult result = spot_perp_lab::replay_two_markets(
                spot, perpetual, start_ns, end_ns, interval_ns
            );
            py::dict output;
            output["decision_time_ns"] = copy_array(result.decision_time_ns);
            add_market(output, "spot", result.spot);
            add_market(output, "perpetual", result.perpetual);
            return output;
        },
        py::arg("spot_event_time_ns"), py::arg("spot_aggregate_trade_id"),
        py::arg("spot_price"), py::arg("spot_quantity"), py::arg("spot_notional"),
        py::arg("spot_signed_quantity"), py::arg("spot_signed_notional"),
        py::arg("spot_is_buyer_maker"), py::arg("perpetual_event_time_ns"),
        py::arg("perpetual_aggregate_trade_id"), py::arg("perpetual_price"),
        py::arg("perpetual_quantity"), py::arg("perpetual_notional"),
        py::arg("perpetual_signed_quantity"), py::arg("perpetual_signed_notional"),
        py::arg("perpetual_is_buyer_maker"), py::arg("start_ns"), py::arg("end_ns"),
        py::arg("interval_ns")
    );
}
