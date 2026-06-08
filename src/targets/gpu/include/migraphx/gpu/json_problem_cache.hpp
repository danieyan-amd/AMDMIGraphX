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
 *
 */
#ifndef MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/problem_cache.hpp> // cache_device_key

#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Concrete problem_cache_backend that persists entries as a JSON file.
//
// The on-disk format matches the one already used by problem_cache::save():
// a JSON array of [cache_device_key, inner_map] pairs, where inner_map is
// itself an unordered_map<value, value>. This means a JSON file written by
// problem_cache today is loadable by json_problem_cache and vice versa --
// the integration in PR-B' can swap the in-place serialization for a
// backend call without altering the on-disk schema.
//
// Legacy flat-object format (a single JSON object whose keys are
// JSON-encoded {name, problem} strings) is migrated into the bucket
// identified by `migration_device_key`. Set the migration key before
// calling load() if you anticipate legacy files; otherwise legacy entries
// land in an anonymous (default-constructed) bucket.
//
// This class deliberately does NOT consult the MIGRAPHX_PROBLEM_CACHE
// environment variable. Path resolution is the caller's responsibility --
// keeping env-var policy out of the backend lets unit tests use arbitrary
// paths and lets the eventual aggregator route to multiple files.
struct MIGRAPHX_GPU_EXPORT json_problem_cache
{
    // Bucket used when migrating legacy flat-object files. Has no effect on
    // load() for the current array-of-pairs format. Defaults to an
    // anonymous (all-zero) key.
    void set_migration_device_key(cache_device_key key);

    // problem_cache_backend concept members:
    void load(const std::string& path);
    void save(const std::string& path) const;
    void insert(const cache_device_key& dk, const value& key, const value& solution);
    void mark(const cache_device_key& dk, const value& key);
    optional<value> get(const cache_device_key& dk, const value& key) const;
    bool has(const cache_device_key& dk, const value& key) const;

    // Exposed for tests and the future aggregator. Outer key is the device
    // bucket; inner is the {name, problem} -> solution map. Mirrors the
    // public `cache` member on problem_cache so PR-B' can adopt this
    // backend without changing the schema.
    std::unordered_map<cache_device_key, std::unordered_map<value, value>> cache;

    private:
    cache_device_key migration_device_key{};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP
