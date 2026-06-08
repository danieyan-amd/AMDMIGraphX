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
#include <migraphx/gpu/sqlite_problem_cache.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/logger.hpp>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#pragma clang diagnostic ignored "-Wreserved-identifier"
#include "third_party/sqlite3.h"
#pragma clang diagnostic pop

#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Compile-time confirmation that sqlite_problem_cache satisfies the backend
// concept. If a method signature drifts, this assertion fires at the
// definition site rather than at some far-away usage.
static_assert(std::is_constructible<problem_cache_backend, sqlite_problem_cache>{},
              "sqlite_problem_cache must satisfy the problem_cache_backend concept");

namespace {

constexpr const char* schema_sql = "CREATE TABLE IF NOT EXISTS solutions ("
                                   "  device_key  TEXT NOT NULL,"
                                   "  problem_key TEXT NOT NULL,"
                                   "  solution    TEXT NOT NULL,"
                                   "  PRIMARY KEY (device_key, problem_key)"
                                   ") WITHOUT ROWID;";

constexpr const char* select_all_sql = "SELECT device_key, problem_key, solution FROM solutions;";

constexpr const char* delete_all_sql = "DELETE FROM solutions;";

constexpr const char* insert_sql =
    "INSERT OR REPLACE INTO solutions(device_key, problem_key, solution) "
    "VALUES (?1, ?2, ?3);";

// SQLITE_TRANSIENT is a C-style cast macro from the sqlite3 amalgamation;
// hoist the cast to file scope so call sites stay clean and -Wold-style-cast
// fires only here. Cannot be constexpr because the underlying cast is
// reinterpret-style.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
const auto k_sqlite_transient = SQLITE_TRANSIENT;
#pragma clang diagnostic pop

// RAII wrapper for sqlite3* connection. Closes on destruction. Move/copy
// deleted: instances live as named locals and are never transferred.
struct sqlite_db
{
    sqlite3* db = nullptr;
    explicit sqlite_db(const std::string& path)
    {
        if(sqlite3_open(path.c_str(), &db) != SQLITE_OK)
        {
            std::string err = db != nullptr ? sqlite3_errmsg(db) : "sqlite3_open failed";
            sqlite3_close(db);
            db = nullptr;
            MIGRAPHX_THROW("sqlite_problem_cache: cannot open " + path + ": " + err);
        }
    }
    sqlite_db(const sqlite_db&)            = delete;
    sqlite_db& operator=(const sqlite_db&) = delete;
    sqlite_db(sqlite_db&&)                 = delete;
    sqlite_db& operator=(sqlite_db&&)      = delete;
    ~sqlite_db()
    {
        if(db != nullptr)
            sqlite3_close(db);
    }
    sqlite3* get() const { return db; }
};

// RAII wrapper for sqlite3_stmt*. Finalizes on destruction.
struct sqlite_stmt
{
    sqlite3_stmt* stmt = nullptr;
    sqlite_stmt(sqlite3* db, const char* sql)
    {
        if(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        {
            std::string err = sqlite3_errmsg(db);
            MIGRAPHX_THROW("sqlite_problem_cache: prepare failed: " + err + " (sql=" + sql + ")");
        }
    }
    sqlite_stmt(const sqlite_stmt&)            = delete;
    sqlite_stmt& operator=(const sqlite_stmt&) = delete;
    sqlite_stmt(sqlite_stmt&&)                 = delete;
    sqlite_stmt& operator=(sqlite_stmt&&)      = delete;
    ~sqlite_stmt()
    {
        if(stmt != nullptr)
            sqlite3_finalize(stmt);
    }
    sqlite3_stmt* get() const { return stmt; }
};

void exec_sql(sqlite3* db, const char* sql)
{
    char* errmsg = nullptr;
    if(sqlite3_exec(db, sql, nullptr, nullptr, &errmsg) != SQLITE_OK)
    {
        std::string err = errmsg != nullptr ? errmsg : "(no message)";
        sqlite3_free(errmsg);
        MIGRAPHX_THROW("sqlite_problem_cache: exec failed: " + err + " (sql=" + sql + ")");
    }
}

} // namespace

void sqlite_problem_cache::load(const std::string& path)
{
    if(path.empty())
        return;
    if(not fs::exists(path))
    {
        // Missing file is not an error: typical first-run case. Caller may
        // immediately follow with save() to create the file.
        return;
    }
    sqlite_db conn(path);
    exec_sql(conn.get(), schema_sql);

    sqlite_stmt sel(conn.get(), select_all_sql);
    while(true)
    {
        int rc = sqlite3_step(sel.get());
        if(rc == SQLITE_DONE)
            break;
        if(rc != SQLITE_ROW)
        {
            std::string err = sqlite3_errmsg(conn.get());
            MIGRAPHX_THROW("sqlite_problem_cache: SELECT failed: " + err);
        }
        const auto* dk_text =
            reinterpret_cast<const char*>(sqlite3_column_text(sel.get(), 0)); // NOLINT
        const auto* pk_text =
            reinterpret_cast<const char*>(sqlite3_column_text(sel.get(), 1)); // NOLINT
        const auto* sol_text =
            reinterpret_cast<const char*>(sqlite3_column_text(sel.get(), 2)); // NOLINT
        if(dk_text == nullptr or pk_text == nullptr or sol_text == nullptr)
        {
            log::warn() << "sqlite_problem_cache: skipping row with NULL column at " << path;
            continue;
        }
        cache_device_key dk;
        from_value(from_json_string(dk_text), dk);
        auto pk       = from_json_string(pk_text);
        auto sol      = from_json_string(sol_text);
        cache[dk][pk] = sol;
    }
}

void sqlite_problem_cache::save(const std::string& path) const
{
    if(path.empty())
        return;
    sqlite_db conn(path);
    exec_sql(conn.get(), schema_sql);
    exec_sql(conn.get(), "BEGIN IMMEDIATE TRANSACTION;");
    try
    {
        exec_sql(conn.get(), delete_all_sql);
        sqlite_stmt ins(conn.get(), insert_sql);
        for(const auto& bucket : cache)
        {
            auto dk_str = to_json_string(to_value(bucket.first));
            for(const auto& kv : bucket.second)
            {
                auto pk_str  = to_json_string(kv.first);
                auto sol_str = to_json_string(kv.second);
                sqlite3_reset(ins.get());
                sqlite3_clear_bindings(ins.get());
                sqlite3_bind_text(ins.get(), 1, dk_str.c_str(), -1, k_sqlite_transient);
                sqlite3_bind_text(ins.get(), 2, pk_str.c_str(), -1, k_sqlite_transient);
                sqlite3_bind_text(ins.get(), 3, sol_str.c_str(), -1, k_sqlite_transient);
                int rc = sqlite3_step(ins.get());
                if(rc != SQLITE_DONE)
                {
                    std::string err = sqlite3_errmsg(conn.get());
                    MIGRAPHX_THROW("sqlite_problem_cache: INSERT failed: " + err);
                }
            }
        }
        exec_sql(conn.get(), "COMMIT;");
    }
    catch(...)
    {
        // Best-effort rollback; ignore secondary errors.
        char* errmsg = nullptr;
        sqlite3_exec(conn.get(), "ROLLBACK;", nullptr, nullptr, &errmsg);
        sqlite3_free(errmsg);
        throw;
    }
}

void sqlite_problem_cache::insert(const cache_device_key& dk,
                                  const value& key,
                                  const value& solution)
{
    assert(not solution.is_null());
    cache[dk][key] = solution;
}

void sqlite_problem_cache::mark(const cache_device_key& dk, const value& key)
{
    cache[dk].insert(std::make_pair(key, value{}));
}

optional<value> sqlite_problem_cache::get(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return nullopt;
    auto it = bucket_it->second.find(key);
    if(it == bucket_it->second.end())
        return nullopt;
    return it->second;
}

bool sqlite_problem_cache::has(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return false;
    return contains(bucket_it->second, key);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
