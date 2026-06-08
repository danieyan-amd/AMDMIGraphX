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
#ifndef MIGRAPHX_GUARD_GPU_SQLITE_PROBLEM_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_SQLITE_PROBLEM_CACHE_HPP

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

// Concrete problem_cache_backend that persists entries in a SQLite database.
//
// Schema (single table, deliberately minimal):
//
//   CREATE TABLE solutions (
//       device_key  TEXT NOT NULL, -- to_json_string(to_value(cache_device_key))
//       problem_key TEXT NOT NULL, -- to_json_string(value)  (the {name,problem} key)
//       solution    TEXT NOT NULL, -- to_json_string(value)  ("null" for marks)
//       PRIMARY KEY (device_key, problem_key)
//   ) WITHOUT ROWID;
//
// The Paul-compliant baseline carries no per-entry hardware metadata and no
// schema-version concept, so this backend deliberately does NOT replicate
// the columns the old (rejected) sqlite_cache_backend carried (op_kind,
// graphics_clock_mhz, regs_per_block, etc.). Cross-row identity is the
// composite (device_key, problem_key) primary key.
//
// Persistence model is store-and-forward, mirroring json_problem_cache:
//   - load(path)        : SELECT * into the in-memory `cache` map.
//   - save(path) const  : BEGIN; DELETE FROM solutions; INSERT every row
//                         currently in `cache`; COMMIT. (One transaction,
//                         file-atomic on POSIX/Windows journaling.)
//   - insert/mark/get/has: in-memory only. Persistence happens on save().
//
// This matches json_problem_cache's contract bit-for-bit (load+mutate+save,
// no eager I/O on each insert) and keeps the on-disk file portable: any
// SQLite reader can dump the table, and an external aggregator (PR-D')
// can simply walk all rows.
//
// Like json_problem_cache, this class deliberately does NOT consult the
// MIGRAPHX_PROBLEM_CACHE environment variable. Path resolution is the
// caller's responsibility.
struct MIGRAPHX_GPU_EXPORT sqlite_problem_cache
{
    sqlite_problem_cache() = default;

    // problem_cache_backend concept members:
    void load(const std::string& path);
    void save(const std::string& path) const;
    void insert(const cache_device_key& dk, const value& key, const value& solution);
    void mark(const cache_device_key& dk, const value& key);
    optional<value> get(const cache_device_key& dk, const value& key) const;
    bool has(const cache_device_key& dk, const value& key) const;

    // Exposed for tests and the future aggregator. Outer key is the device
    // bucket; inner is the {name, problem} -> solution map. Mirrors the
    // public `cache` member on json_problem_cache and problem_cache.
    std::unordered_map<cache_device_key, std::unordered_map<value, value>> cache;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_SQLITE_PROBLEM_CACHE_HPP
