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

#include <test.hpp>
#include <migraphx/gpu/problem_cache_aggregator.hpp>
#include <migraphx/gpu/json_problem_cache.hpp>
#include <migraphx/gpu/problem_cache.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/value.hpp>
#include <chrono>
#include <string>
#include <vector>

namespace {

using migraphx::gpu::cache_conflict_policy;
using migraphx::gpu::cache_device_key;
using migraphx::gpu::cache_input_spec;
using migraphx::gpu::cache_merge_options;
using migraphx::gpu::cache_output_spec;
using migraphx::gpu::convert_problem_cache;
using migraphx::gpu::json_problem_cache;
using migraphx::gpu::legacy_device_policy;
using migraphx::gpu::merge_problem_caches;
using migraphx::gpu::validate_problem_cache;

// ----------------------------------------------------------------------------
// Test helpers. All tests build entries via a small POD here, then serialize
// them to disk through a json_problem_cache instance. This mirrors the way
// the aggregator itself reads/writes (the only supported "json" backend in
// this layer) so the tests exercise the same code paths as production.
// ----------------------------------------------------------------------------

struct test_entry
{
    cache_device_key device_key;
    std::string name;
    std::string problem; // a string carried inside a value
    std::string solution;
};

cache_device_key dk_a()
{
    cache_device_key k;
    k.device_name    = "gfx1100";
    k.gfx_name       = "gfx1100";
    k.cu_count       = 48;
    k.wavefront_size = 32;
    return k;
}

cache_device_key dk_b()
{
    cache_device_key k;
    k.device_name    = "gfx1101";
    k.gfx_name       = "gfx1101";
    k.cu_count       = 64;
    k.wavefront_size = 32;
    return k;
}

cache_device_key dk_empty() { return cache_device_key{}; }

migraphx::fs::path make_test_dir(const std::string& tag)
{
    auto base = migraphx::fs::temp_directory_path() / "migraphx_problem_cache_aggregator_tests";
    auto now  = std::chrono::steady_clock::now().time_since_epoch().count();
    auto dir  = base / (tag + "_" + std::to_string(now));
    migraphx::fs::create_directories(dir);
    return dir;
}

void write_entries(const std::string& path, const std::vector<test_entry>& entries)
{
    json_problem_cache b;
    for(const auto& e : entries)
    {
        // {name, problem} key shape, matching json_problem_cache::create_key.
        migraphx::value pkey{{"name", e.name}, {"problem", e.problem}};
        if(e.solution.empty())
            b.mark(e.device_key, pkey);
        else
            b.insert(e.device_key, pkey, migraphx::value(e.solution));
    }
    b.save(path);
}

// Read every (device_key, problem_key, solution) triple from a JSON cache
// file and project to a comparable string tuple for set equality.
struct triple
{
    std::string device_key_json;
    std::string problem_key_json;
    std::string solution_json;

    friend bool operator<(const triple& a, const triple& b)
    {
        return std::tie(a.device_key_json, a.problem_key_json, a.solution_json) <
               std::tie(b.device_key_json, b.problem_key_json, b.solution_json);
    }
    friend bool operator==(const triple& a, const triple& b)
    {
        return a.device_key_json == b.device_key_json and
               a.problem_key_json == b.problem_key_json and a.solution_json == b.solution_json;
    }
};

std::vector<triple> read_triples(const std::string& path)
{
    json_problem_cache b;
    b.load(path);
    std::vector<triple> out;
    for(const auto& bucket : b.cache)
    {
        for(const auto& kv : bucket.second)
        {
            triple t;
            t.device_key_json  = migraphx::to_json_string(migraphx::to_value(bucket.first));
            t.problem_key_json = migraphx::to_json_string(kv.first);
            t.solution_json    = migraphx::to_json_string(kv.second);
            out.push_back(std::move(t));
        }
    }
    return out;
}

cache_merge_options
base_options(const std::string& in1, const std::string& in2, const std::string& out)
{
    cache_merge_options opts;
    opts.inputs = {{in1, "json"}, {in2, "json"}};
    opts.output = {out, "json"};
    return opts;
}

// ----------------------------------------------------------------------------
// Core merge behaviour
// ----------------------------------------------------------------------------

TEST_CASE(merge_two_json_caches_with_distinct_entries)
{
    auto dir = make_test_dir("distinct");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_a(), "gemm", "p2", "s2"}});

    auto report  = merge_problem_caches(base_options(a, b, out));
    auto triples = read_triples(out);

    EXPECT(report.total_output_entries == 2);
    EXPECT(triples.size() == 2);
}

TEST_CASE(deduplicate_same_key_same_solution)
{
    auto dir = make_test_dir("dedup");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_a(), "conv", "p1", "s1"}});

    auto report  = merge_problem_caches(base_options(a, b, out));
    auto triples = read_triples(out);

    EXPECT(report.duplicate_count == 1);
    EXPECT(report.total_output_entries == 1);
    EXPECT(triples.size() == 1);
}

TEST_CASE(conflict_same_key_different_solution_errors_by_default)
{
    auto dir = make_test_dir("conflict_error");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_a(), "conv", "p1", "s2"}});

    auto opts = base_options(a, b, out);
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(different_device_same_name_problem_keeps_both)
{
    auto dir = make_test_dir("different_device");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_b(), "conv", "p1", "s2"}});

    auto report  = merge_problem_caches(base_options(a, b, out));
    auto triples = read_triples(out);

    EXPECT(report.total_output_entries == 2);
    EXPECT(triples.size() == 2);
}

TEST_CASE(legacy_empty_device_preserve_empty)
{
    auto dir = make_test_dir("legacy_preserve");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_empty(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_empty(), "gemm", "p2", "s2"}});

    auto opts                = base_options(a, b, out);
    opts.empty_device_policy = legacy_device_policy::preserve_empty;

    auto report  = merge_problem_caches(opts);
    auto triples = read_triples(out);

    EXPECT(report.legacy_empty_device_count == 2);
    EXPECT(triples.size() == 2);
}

TEST_CASE(first_wins_reports_conflict)
{
    auto dir = make_test_dir("first_wins");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s_first"}});
    write_entries(b, {{dk_a(), "conv", "p1", "s_last"}});

    auto opts            = base_options(a, b, out);
    opts.conflict_policy = cache_conflict_policy::first_wins;

    auto report  = merge_problem_caches(opts);
    auto triples = read_triples(out);

    EXPECT(report.conflict_count == 1);
    EXPECT(not report.conflicts.empty());
    EXPECT(report.conflicts.front().resolution == "kept-first");
    EXPECT(triples.size() == 1);
    // The kept solution is "s_first" -- to_json_string on a string value
    // wraps it in quotes, hence the literal here.
    EXPECT(triples.front().solution_json == std::string{"\"s_first\""});
}

TEST_CASE(last_wins_reports_conflict)
{
    auto dir = make_test_dir("last_wins");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s_first"}});
    write_entries(b, {{dk_a(), "conv", "p1", "s_last"}});

    auto opts            = base_options(a, b, out);
    opts.conflict_policy = cache_conflict_policy::last_wins;

    auto report  = merge_problem_caches(opts);
    auto triples = read_triples(out);

    EXPECT(report.conflict_count == 1);
    EXPECT(not report.conflicts.empty());
    EXPECT(report.conflicts.front().resolution == "kept-last");
    EXPECT(triples.size() == 1);
    EXPECT(triples.front().solution_json == std::string{"\"s_last\""});
}

TEST_CASE(output_can_be_loaded_by_json_backend)
{
    auto dir = make_test_dir("loadable");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});
    write_entries(b, {{dk_a(), "gemm", "p2", "s2"}});

    (void)merge_problem_caches(base_options(a, b, out));

    json_problem_cache backend;
    backend.load(out);

    std::size_t total = 0;
    for(const auto& bucket : backend.cache)
        total += bucket.second.size();
    EXPECT(total == 2);
}

TEST_CASE(corrupt_input_and_unavailable_backend_errors)
{
    auto dir = make_test_dir("corrupt");
    auto bad = (dir / "bad.json").string();
    auto ok  = (dir / "ok.json").string();
    auto out = (dir / "out.json").string();

    migraphx::write_string(bad, "{not valid json");
    write_entries(ok, {{dk_a(), "gemm", "p2", "s2"}});

    auto opts = base_options(bad, ok, out);
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));

    opts.inputs = {{ok, "sqlite"}, {ok, "json"}};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(large_synthetic_merge_10k_entries)
{
    auto dir = make_test_dir("large10k");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    std::vector<test_entry> e1;
    std::vector<test_entry> e2;
    e1.reserve(5000);
    e2.reserve(5000);

    for(int i = 0; i < 5000; ++i)
    {
        e1.push_back(
            {dk_a(), "name_" + std::to_string(i), "problem_" + std::to_string(i), "sol_a"});
        e2.push_back({dk_a(),
                      "name_" + std::to_string(i + 5000),
                      "problem_" + std::to_string(i + 5000),
                      "sol_b"});
    }

    write_entries(a, e1);
    write_entries(b, e2);

    auto report  = merge_problem_caches(base_options(a, b, out));
    auto triples = read_triples(out);

    EXPECT(report.total_input_entries == 10000);
    EXPECT(report.total_output_entries == 10000);
    EXPECT(triples.size() == 10000);
}

// ----------------------------------------------------------------------------
// Validation surfaces (empty inputs, empty output, unsupported backends, etc.)
// ----------------------------------------------------------------------------

TEST_CASE(empty_inputs_list_throws)
{
    cache_merge_options opts;
    opts.output = {"unused.json", "json"};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(empty_output_path_throws)
{
    auto dir = make_test_dir("empty_out");
    auto a   = (dir / "a.json").string();
    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});

    cache_merge_options opts;
    opts.inputs = {{a, "json"}};
    opts.output = {"", "json"};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(unsupported_input_backend_throws)
{
    auto dir = make_test_dir("bad_in_backend");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.json").string();
    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});

    cache_merge_options opts;
    opts.inputs = {{a, "sqlite"}};
    opts.output = {out, "json"};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(unsupported_output_backend_throws)
{
    auto dir = make_test_dir("bad_out_backend");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.msg").string();
    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});

    cache_merge_options opts;
    opts.inputs = {{a, "json"}};
    opts.output = {out, "msgpack"};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(missing_input_file_throws)
{
    auto dir     = make_test_dir("missing_in");
    auto missing = (dir / "does_not_exist.json").string();
    auto out     = (dir / "out.json").string();

    cache_merge_options opts;
    opts.inputs = {{missing, "json"}};
    opts.output = {out, "json"};
    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(map_to_device_assigns_mapped_key)
{
    auto dir = make_test_dir("map_to_device");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_empty(), "conv", "p1", "s1"}, {dk_empty(), "gemm", "p2", "s2"}});

    cache_merge_options opts;
    opts.inputs              = {{a, "json"}};
    opts.output              = {out, "json"};
    opts.empty_device_policy = legacy_device_policy::map_to_device;
    opts.mapped_device_key   = "gfx1100|48|32";

    auto report  = merge_problem_caches(opts);
    auto triples = read_triples(out);

    EXPECT(report.legacy_empty_device_count == 0);
    EXPECT(triples.size() == 2);
    // Each entry's serialized device key contains the mapped string.
    for(const auto& t : triples)
        EXPECT(t.device_key_json.find("gfx1100|48|32") != std::string::npos);
}

TEST_CASE(map_to_device_without_mapped_key_throws)
{
    auto dir = make_test_dir("map_to_device_missing");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_empty(), "conv", "p1", "s1"}});

    cache_merge_options opts;
    opts.inputs              = {{a, "json"}};
    opts.output              = {out, "json"};
    opts.empty_device_policy = legacy_device_policy::map_to_device;
    // mapped_device_key intentionally left empty.

    EXPECT(test::throws([&] { merge_problem_caches(opts); }));
}

TEST_CASE(three_input_merge_with_per_file_counters)
{
    auto dir = make_test_dir("three_inputs");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto c   = (dir / "c.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s1"}, {dk_a(), "conv", "p2", "s2"}});
    write_entries(b, {{dk_a(), "gemm", "p3", "s3"}});
    write_entries(c, {{dk_b(), "conv", "p1", "s1"}, {dk_b(), "gemm", "p4", "s4"}});

    cache_merge_options opts;
    opts.inputs = {{a, "json"}, {b, "json"}, {c, "json"}};
    opts.output = {out, "json"};
    auto report = merge_problem_caches(opts);

    EXPECT(report.total_input_files == 3);
    EXPECT(report.total_input_entries == 5);
    EXPECT(report.total_output_entries == 5);
    EXPECT(report.per_input_file_counts[a] == 2);
    EXPECT(report.per_input_file_counts[b] == 1);
    EXPECT(report.per_input_file_counts[c] == 2);
    EXPECT(report.per_input_backend_counts["json"] == 5);

    auto dk_a_str = migraphx::to_json_string(migraphx::to_value(dk_a()));
    auto dk_b_str = migraphx::to_json_string(migraphx::to_value(dk_b()));
    EXPECT(report.per_device_key_counts[dk_a_str] == 3);
    EXPECT(report.per_device_key_counts[dk_b_str] == 2);
}

TEST_CASE(conflict_records_input_provenance)
{
    auto dir = make_test_dir("provenance");
    auto a   = (dir / "a.json").string();
    auto b   = (dir / "b.json").string();
    auto out = (dir / "out.json").string();

    write_entries(a, {{dk_a(), "conv", "p1", "s_a"}});
    write_entries(b, {{dk_a(), "conv", "p1", "s_b"}});

    cache_merge_options opts;
    opts.inputs          = {{a, "json"}, {b, "json"}};
    opts.output          = {out, "json"};
    opts.conflict_policy = cache_conflict_policy::last_wins;

    auto report = merge_problem_caches(opts);
    EXPECT(report.conflicts.size() == 1);
    auto& c = report.conflicts.front();
    EXPECT(c.existing_input_index == 0);
    EXPECT(c.incoming_input_index == 1);
    EXPECT(c.existing_input_path == a);
    EXPECT(c.incoming_input_path == b);
    EXPECT(c.existing_solution == std::string{"\"s_a\""});
    EXPECT(c.incoming_solution == std::string{"\"s_b\""});
    EXPECT(c.key.name == "conv");
    EXPECT(c.key.problem == std::string{"\"p1\""});
}

TEST_CASE(output_entries_round_trip_as_set)
{
    auto dir = make_test_dir("roundtrip_set");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.json").string();

    std::vector<test_entry> src = {{dk_b(), "gemm", "z", "s"},
                                   {dk_a(), "conv", "b", "s"},
                                   {dk_a(), "conv", "a", "s"},
                                   {dk_a(), "gemm", "m", "s"}};
    write_entries(a, src);

    cache_merge_options opts;
    opts.inputs = {{a, "json"}};
    opts.output = {out, "json"};
    (void)merge_problem_caches(opts);

    auto src_triples = read_triples(a);
    auto out_triples = read_triples(out);
    std::sort(src_triples.begin(), src_triples.end());
    std::sort(out_triples.begin(), out_triples.end());
    EXPECT(src_triples == out_triples);
}

TEST_CASE(report_elapsed_ms_is_populated)
{
    auto dir = make_test_dir("elapsed");
    auto a   = (dir / "a.json").string();
    auto out = (dir / "out.json").string();
    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});

    cache_merge_options opts;
    opts.inputs = {{a, "json"}};
    opts.output = {out, "json"};
    auto report = merge_problem_caches(opts);

    EXPECT(report.elapsed_ms >= 0);
    EXPECT(report.output_path == out);
    EXPECT(report.output_backend == "json");
}

// ----------------------------------------------------------------------------
// convert_problem_cache
// ----------------------------------------------------------------------------

TEST_CASE(convert_round_trips_entries)
{
    auto dir = make_test_dir("convert_roundtrip");
    auto in  = (dir / "in.json").string();
    auto out = (dir / "out.json").string();

    std::vector<test_entry> src = {
        {dk_a(), "conv", "p1", "s1"}, {dk_a(), "gemm", "p2", "s2"}, {dk_b(), "attn", "p3", "s3"}};
    write_entries(in, src);

    auto report  = convert_problem_cache({in, "json"}, {out, "json"});
    auto triples = read_triples(out);

    EXPECT(report.total_input_files == 1);
    EXPECT(report.total_input_entries == 3);
    EXPECT(report.total_output_entries == 3);
    EXPECT(report.conflict_count == 0);
    EXPECT(report.duplicate_count == 0);
    EXPECT(triples.size() == 3);
}

TEST_CASE(convert_with_map_to_device_assigns_key)
{
    auto dir = make_test_dir("convert_map");
    auto in  = (dir / "in.json").string();
    auto out = (dir / "out.json").string();

    write_entries(in, {{dk_empty(), "conv", "p1", "s1"}, {dk_empty(), "gemm", "p2", "s2"}});

    cache_merge_options opts;
    opts.empty_device_policy = legacy_device_policy::map_to_device;
    opts.mapped_device_key   = "gfx1100|48|32";

    auto report  = convert_problem_cache({in, "json"}, {out, "json"}, opts);
    auto triples = read_triples(out);

    EXPECT(report.legacy_empty_device_count == 0);
    EXPECT(triples.size() == 2);
    for(const auto& t : triples)
        EXPECT(t.device_key_json.find("gfx1100|48|32") != std::string::npos);
}

TEST_CASE(convert_unsupported_backend_throws)
{
    auto dir = make_test_dir("convert_bad_backend");
    auto in  = (dir / "in.json").string();
    auto out = (dir / "out.sqlite").string();
    write_entries(in, {{dk_a(), "conv", "p1", "s1"}});

    EXPECT(test::throws([&] { (void)convert_problem_cache({in, "json"}, {out, "sqlite"}); }));
}

// ----------------------------------------------------------------------------
// validate_problem_cache
// ----------------------------------------------------------------------------

TEST_CASE(validate_clean_cache_is_valid)
{
    auto dir = make_test_dir("validate_clean");
    auto a   = (dir / "a.json").string();
    write_entries(
        a,
        {{dk_a(), "conv", "p1", "s1"}, {dk_a(), "gemm", "p2", "s2"}, {dk_b(), "conv", "p1", "s1"}});

    auto report = validate_problem_cache({a, "json"});
    EXPECT(report.valid);
    EXPECT(report.total_entries == 3);
    EXPECT(report.distinct_keys == 3);
    EXPECT(report.conflicting_solution_count == 0);
    EXPECT(report.duplicate_solution_count == 0);
    EXPECT(report.empty_device_key_count == 0);
    EXPECT(report.strict_device_key);
}

TEST_CASE(validate_legacy_empty_device_strict_is_invalid)
{
    auto dir = make_test_dir("validate_legacy_strict");
    auto a   = (dir / "a.json").string();
    write_entries(a, {{dk_empty(), "conv", "p1", "s1"}, {dk_a(), "gemm", "p2", "s2"}});

    auto report = validate_problem_cache({a, "json"}, /*strict_device_key=*/true);
    EXPECT(not report.valid);
    EXPECT(report.empty_device_key_count == 1);
    EXPECT(report.conflicting_solution_count == 0);
}

TEST_CASE(validate_legacy_empty_device_non_strict_is_valid)
{
    auto dir = make_test_dir("validate_legacy_nonstrict");
    auto a   = (dir / "a.json").string();
    write_entries(a, {{dk_empty(), "conv", "p1", "s1"}, {dk_a(), "gemm", "p2", "s2"}});

    auto report = validate_problem_cache({a, "json"}, /*strict_device_key=*/false);
    EXPECT(report.valid);
    EXPECT(report.empty_device_key_count == 1);
    EXPECT(not report.strict_device_key);
}

TEST_CASE(validate_unsupported_backend_throws)
{
    auto dir = make_test_dir("validate_bad_backend");
    auto a   = (dir / "a.json").string();
    write_entries(a, {{dk_a(), "conv", "p1", "s1"}});

    EXPECT(test::throws([&] { (void)validate_problem_cache({a, "sqlite"}); }));
}

TEST_CASE(validate_missing_file_throws)
{
    auto dir     = make_test_dir("validate_missing");
    auto missing = (dir / "no.json").string();
    EXPECT(test::throws([&] { (void)validate_problem_cache({missing, "json"}); }));
}

} // namespace

int main(int argc, const char* argv[]) { test::run(argc, argv); }
