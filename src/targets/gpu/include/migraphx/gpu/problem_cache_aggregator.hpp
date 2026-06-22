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
#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_AGGREGATOR_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_AGGREGATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/problem_cache.hpp> // cache_device_key
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// ----------------------------------------------------------------------------
// Offline aggregator over problem-cache files.
//
// This module reads one or more on-disk problem caches (currently only the
// "json" backend, matching json_problem_cache's on-disk format), merges or
// validates them according to a small set of policies, and writes the
// result back out. It is deliberately an off-line tool surface: nothing
// here is on the migraphx::compile() hot path.
//
// The aggregator works against a logical entry shape -- (device_key bucket
// id, problem_key, solution_value) -- not against any particular backend's
// internal types. The currently-supported backend tag is "json"; the design
// reserves room for future backends ("sqlite", etc.) by carrying a
// cache_input_spec::backend / cache_output_spec::backend string, but
// supplying an unsupported backend throws.
// ----------------------------------------------------------------------------

enum class cache_conflict_policy
{
    error_on_conflict,
    first_wins,
    last_wins
};

enum class legacy_device_policy
{
    preserve_empty,
    map_to_device
};

struct cache_input_spec
{
    std::string path;
    std::string backend = "json";
};

struct cache_output_spec
{
    std::string path;
    std::string backend = "json";
};

// Identity tuple under which two cache entries are considered the "same
// problem". Two entries with the same merge key but different solutions are
// a conflict; same key + same solution is a duplicate.
struct cache_merge_key
{
    std::string device_key; // serialized device bucket (to_json_string)
    std::string name;       // entry name (e.g. "conv", "gemm")
    std::string problem;    // serialized problem descriptor

    bool operator==(const cache_merge_key& other) const
    {
        return device_key == other.device_key and name == other.name and problem == other.problem;
    }
};

struct cache_merge_key_hash
{
    std::size_t operator()(const cache_merge_key& key) const
    {
        auto h1 = std::hash<std::string>{}(key.device_key);
        auto h2 = std::hash<std::string>{}(key.name);
        auto h3 = std::hash<std::string>{}(key.problem);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

// One conflict record. Provenance fields point back at the input file that
// supplied each conflicting solution so an operator can reconcile manually.
struct cache_conflict
{
    cache_merge_key key;
    std::string existing_solution;
    std::string incoming_solution;
    std::size_t existing_input_index = 0;
    std::size_t incoming_input_index = 0;
    std::string existing_input_path;
    std::string incoming_input_path;
    std::string resolution; // "kept-first" | "kept-last" | "error"
};

struct cache_merge_options
{
    std::vector<cache_input_spec> inputs;
    cache_output_spec output;
    cache_conflict_policy conflict_policy    = cache_conflict_policy::error_on_conflict;
    legacy_device_policy empty_device_policy = legacy_device_policy::preserve_empty;
    std::string mapped_device_key;
    bool strict_device_key       = true;
    bool report_metadata_loss    = true;
    /// When true, remap GFX variant names (e.g. gfx1151 -> gfx1150) to
    /// canonical names during merge, deduplicating entries across compatible
    /// architectures. Uses get_canonical_gfx() from device_name.hpp.
    bool remap_gfx_to_canonical  = false;
};

struct cache_merge_report
{
    std::size_t total_input_files         = 0;
    std::size_t total_input_entries       = 0;
    std::size_t total_output_entries      = 0;
    std::size_t duplicate_count           = 0;
    std::size_t conflict_count            = 0;
    std::size_t legacy_empty_device_count = 0;
    std::size_t metadata_loss_count       = 0;
    std::map<std::string, std::size_t> per_device_key_counts;
    std::map<std::string, std::size_t> per_input_file_counts;
    std::map<std::string, std::size_t> per_input_backend_counts;
    std::vector<cache_conflict> conflicts;
    std::int64_t elapsed_ms = 0;
    std::string output_path;
    std::string output_backend;
};

struct cache_validation_report
{
    std::size_t total_entries              = 0;
    std::size_t distinct_keys              = 0;
    std::size_t duplicate_solution_count   = 0;
    std::size_t conflicting_solution_count = 0;
    std::size_t empty_device_key_count     = 0;
    bool strict_device_key                 = true;
    bool valid                             = true;
};

// Merge N inputs into a single output, applying conflict_policy and
// empty_device_policy along the way. Throws on:
//   * empty inputs list / empty output path
//   * unsupported backend (anything other than "json")
//   * missing input file
//   * corrupt input JSON
//   * conflict_policy == error_on_conflict and a conflict is encountered
//   * empty_device_policy == map_to_device and mapped_device_key is empty
MIGRAPHX_GPU_EXPORT cache_merge_report merge_problem_caches(const cache_merge_options& options);

// Read a single input and produce a validation report. Does not write any
// output. strict_device_key=true (the default) marks empty device keys as
// invalid; pass false to allow legacy unmigrated entries.
MIGRAPHX_GPU_EXPORT cache_validation_report validate_problem_cache(const cache_input_spec& input,
                                                                   bool strict_device_key = true);

// Convenience wrapper for a 1-input merge (effectively "convert backend X
// to backend Y, optionally rewriting empty device keys").
MIGRAPHX_GPU_EXPORT cache_merge_report
convert_problem_cache(const cache_input_spec& input,
                      const cache_output_spec& output,
                      const cache_merge_options& options = {});

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_AGGREGATOR_HPP
