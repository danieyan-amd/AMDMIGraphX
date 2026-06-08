/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/problem_cache_aggregator.hpp>
#include <migraphx/gpu/json_problem_cache.hpp>
#include <migraphx/gpu/problem_cache.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace {

// One in-flight entry as the aggregator sees it: a logical merge key plus
// the raw value-typed solution (kept as a value so we can round-trip
// through the json_problem_cache backend without lossy stringification of
// non-string solutions).
struct merged_entry
{
    cache_device_key device_key{}; // structured device bucket id
    value problem_key{};           // serialized {name, problem} value
    value solution{};              // raw solution value
    std::size_t input_index = 0;
    std::string input_path;
};

// Minimal "logical entry" projected from the on-disk format. The aggregator
// works with these rather than with a backend-internal type, so it can be
// re-targeted at additional backends later without changing the merge
// logic.
struct logical_entry
{
    cache_device_key device_key{};
    value problem_key{}; // {name, problem}
    value solution{};
};

bool backend_enabled(const std::string& backend) { return backend == "json"; }

void require_backend_for_role(const std::string& backend, const char* role)
{
    if(not backend_enabled(backend))
    {
        MIGRAPHX_THROW(std::string{"Problem cache aggregator does not enable backend '"} + backend +
                       "' for " + role + ". Only 'json' is enabled in this layer.");
    }
}

void validate_json_input_file(const std::string& path)
{
    if(path.empty())
        MIGRAPHX_THROW("Problem cache input path is empty.");

    if(not fs::exists(path))
        MIGRAPHX_THROW("Problem cache input does not exist: " + path);

    auto content = read_string(path);
    if(content.empty())
        return;

    try
    {
        (void)from_json_string(content);
    }
    catch(const std::exception& e)
    {
        MIGRAPHX_THROW("Problem cache JSON parse failure for '" + path + "': " + e.what());
    }
}

std::string serialize_device_key(const cache_device_key& dk)
{
    return to_json_string(to_value(dk));
}

cache_merge_key make_merge_key(const cache_device_key& dk, const value& problem_key)
{
    cache_merge_key k;
    k.device_key = serialize_device_key(dk);

    // problem_key is a value of shape {{"name", string}, {"problem", value}};
    // both create_key writers in problem_cache.cpp / json_problem_cache.cpp
    // produce that shape. Defensive lookups so a malformed input throws a
    // clear message rather than crashing. .without_key() strips the entry's
    // own key so the JSON form is the bare value rather than ["problem",...].
    if(problem_key.contains("name"))
        k.name = problem_key.at("name").to<std::string>();
    if(problem_key.contains("problem"))
        k.problem = to_json_string(problem_key.at("problem").without_key());

    return k;
}

logical_entry apply_legacy_device_policy(const logical_entry& in,
                                         const cache_merge_options& options)
{
    logical_entry out = in;
    if(out.device_key == cache_device_key{})
    {
        if(options.empty_device_policy == legacy_device_policy::map_to_device)
        {
            if(options.mapped_device_key.empty())
                MIGRAPHX_THROW(
                    "legacy_device_policy::map_to_device requires mapped_device_key to be set.");
            // mapped_device_key is a serialized device key string. Stuff it
            // into device_name so the resulting bucket is non-empty and
            // deterministic; structured fields stay zero. The serialize()
            // round-trip stays stable because this is a pure assignment.
            out.device_key.device_name = options.mapped_device_key;
        }
    }
    return out;
}

bool is_empty_device_key(const cache_device_key& dk) { return dk == cache_device_key{}; }

std::vector<logical_entry> read_input_entries(const cache_input_spec& input)
{
    require_backend_for_role(input.backend, "input");

    if(input.backend == "json")
        validate_json_input_file(input.path);

    json_problem_cache backend;
    backend.load(input.path);

    std::vector<logical_entry> entries;
    for(const auto& bucket : backend.cache)
    {
        for(const auto& kv : bucket.second)
        {
            entries.push_back(logical_entry{bucket.first, kv.first, kv.second});
        }
    }
    return entries;
}

void write_output_entries(const cache_output_spec& output,
                          const std::vector<logical_entry>& entries)
{
    require_backend_for_role(output.backend, "output");

    json_problem_cache backend;
    for(const auto& e : entries)
    {
        if(e.solution.is_null())
            backend.mark(e.device_key, e.problem_key);
        else
            backend.insert(e.device_key, e.problem_key, e.solution);
    }
    backend.save(output.path);
}

} // namespace

cache_merge_report merge_problem_caches(const cache_merge_options& options)
{
    if(options.inputs.empty())
        MIGRAPHX_THROW("merge_problem_caches requires at least one input.");

    if(options.output.path.empty())
        MIGRAPHX_THROW("merge_problem_caches requires a non-empty output path.");

    require_backend_for_role(options.output.backend, "output");

    auto start = std::chrono::steady_clock::now();

    cache_merge_report report;
    report.total_input_files = options.inputs.size();
    report.output_path       = options.output.path;
    report.output_backend    = options.output.backend;

    std::unordered_map<cache_merge_key, merged_entry, cache_merge_key_hash> merged;

    for(std::size_t i = 0; i < options.inputs.size(); ++i)
    {
        const auto& input = options.inputs[i];
        auto entries      = read_input_entries(input);

        report.total_input_entries += entries.size();
        report.per_input_file_counts[input.path] += entries.size();
        report.per_input_backend_counts[input.backend] += entries.size();

        for(const auto& raw_entry : entries)
        {
            auto entry = apply_legacy_device_policy(raw_entry, options);
            if(is_empty_device_key(entry.device_key))
                report.legacy_empty_device_count++;

            auto key = make_merge_key(entry.device_key, entry.problem_key);
            auto it  = merged.find(key);
            if(it == merged.end())
            {
                merged.insert(
                    {key,
                     merged_entry{
                         entry.device_key, entry.problem_key, entry.solution, i, input.path}});
                continue;
            }

            auto existing_solution_str = to_json_string(it->second.solution);
            auto incoming_solution_str = to_json_string(entry.solution);
            if(existing_solution_str == incoming_solution_str)
            {
                report.duplicate_count++;
                continue;
            }

            report.conflict_count++;
            cache_conflict conflict;
            conflict.key                  = key;
            conflict.existing_solution    = existing_solution_str;
            conflict.incoming_solution    = incoming_solution_str;
            conflict.existing_input_index = it->second.input_index;
            conflict.incoming_input_index = i;
            conflict.existing_input_path  = it->second.input_path;
            conflict.incoming_input_path  = input.path;

            switch(options.conflict_policy)
            {
            case cache_conflict_policy::error_on_conflict:
                conflict.resolution = "error";
                report.conflicts.push_back(conflict);
                MIGRAPHX_THROW("Problem cache conflict for key device_key='" + key.device_key +
                               "', name='" + key.name + "', problem='" + key.problem + "'.");
            case cache_conflict_policy::first_wins:
                conflict.resolution = "kept-first";
                report.conflicts.push_back(conflict);
                break;
            case cache_conflict_policy::last_wins:
                conflict.resolution = "kept-last";
                report.conflicts.push_back(conflict);
                it->second = merged_entry{
                    entry.device_key, entry.problem_key, entry.solution, i, input.path};
                break;
            }
        }
    }

    std::vector<logical_entry> output_entries;
    output_entries.reserve(merged.size());
    for(const auto& item : merged)
    {
        output_entries.push_back(
            logical_entry{item.second.device_key, item.second.problem_key, item.second.solution});
        report.per_device_key_counts[item.first.device_key]++;
    }

    // Stable order: device, then problem-key JSON, then solution JSON. Even
    // though the on-disk format uses an unordered_map (so the file's key
    // order isn't observable), sorting before write keeps merge runs
    // bit-deterministic when the same input set is replayed.
    std::sort(output_entries.begin(),
              output_entries.end(),
              [](const logical_entry& a, const logical_entry& b) {
                  auto a_dk = serialize_device_key(a.device_key);
                  auto b_dk = serialize_device_key(b.device_key);
                  auto a_pk = to_json_string(a.problem_key);
                  auto b_pk = to_json_string(b.problem_key);
                  auto a_s  = to_json_string(a.solution);
                  auto b_s  = to_json_string(b.solution);
                  return std::tie(a_dk, a_pk, a_s) < std::tie(b_dk, b_pk, b_s);
              });

    write_output_entries(options.output, output_entries);

    report.total_output_entries = output_entries.size();
    if(options.report_metadata_loss)
    {
        // Current backend format carries no per-entry metadata beyond
        // (device_key, problem_key, solution); nothing to lose.
        report.metadata_loss_count = 0;
    }

    auto end          = std::chrono::steady_clock::now();
    report.elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return report;
}

cache_validation_report validate_problem_cache(const cache_input_spec& input,
                                               bool strict_device_key)
{
    cache_validation_report report;
    report.strict_device_key = strict_device_key;

    auto entries         = read_input_entries(input);
    report.total_entries = entries.size();

    std::unordered_map<cache_merge_key, std::string, cache_merge_key_hash> observed;
    for(const auto& entry : entries)
    {
        if(is_empty_device_key(entry.device_key))
            report.empty_device_key_count++;

        auto key     = make_merge_key(entry.device_key, entry.problem_key);
        auto sol_str = to_json_string(entry.solution);
        auto it      = observed.find(key);
        if(it == observed.end())
        {
            observed[key] = sol_str;
            continue;
        }

        if(it->second == sol_str)
            report.duplicate_solution_count++;
        else
            report.conflicting_solution_count++;
    }

    report.distinct_keys = observed.size();
    report.valid         = (report.conflicting_solution_count == 0);
    if(strict_device_key and report.empty_device_key_count > 0)
        report.valid = false;

    return report;
}

cache_merge_report convert_problem_cache(const cache_input_spec& input,
                                         const cache_output_spec& output,
                                         const cache_merge_options& options)
{
    cache_merge_options merge_opts = options;
    merge_opts.inputs              = {input};
    merge_opts.output              = output;
    return merge_problem_caches(merge_opts);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
