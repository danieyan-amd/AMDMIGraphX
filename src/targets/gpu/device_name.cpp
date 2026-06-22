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
#include <migraphx/env.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

#include <iostream>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)

std::string get_gfx_name(const std::string& device_name)
{
    return trim(split_string(device_name, ':').front());
}

std::string get_canonical_gfx(const std::string& gfx_name)
{
    // Maps minor GFX architecture variants to their canonical major version.
    // GPUs in the same family produce identical tuning solutions when they
    // share the same ISA, so we collapse them to avoid redundant cache entries.
    //
    // This table is maintained manually. Add new entries as new GPUs ship.
    // Tom (Jun 17 meeting): "We have 11.0x for all Navi 31/32/33, then 11.5x
    // for RDNA 3.5... we should be nuanced about the GFX architecture."
    //
    // Format: {variant} -> {canonical}
    // Only add mappings where the ISA is confirmed identical.
    static const std::unordered_map<std::string, std::string> aliases = {
        // RDNA 3 family (gfx110x)
        {"gfx1031", "gfx1030"}, // Navi 22 -> Navi 21
        {"gfx1032", "gfx1030"}, // Navi 23 -> Navi 21
        {"gfx1033", "gfx1030"}, // Navi 24 -> Navi 21
        {"gfx1034", "gfx1030"}, // Phoenix  -> Navi 21
        {"gfx1035", "gfx1030"}, // Rembrandt-> Navi 21
        {"gfx1036", "gfx1030"}, // Raphael  -> Navi 21
        // RDNA 3 (gfx110x)
        {"gfx1101", "gfx1100"}, // Navi 32  -> Navi 31
        {"gfx1102", "gfx1100"}, // Navi 33  -> Navi 31
        {"gfx1103", "gfx1100"}, // Phoenix  -> Navi 31
        // RDNA 3.5 (gfx115x) — Strix family
        {"gfx1151", "gfx1150"}, // Strix Halo -> Strix
        {"gfx1152", "gfx1150"}, // Kraken     -> Strix
    };

    auto it = aliases.find(gfx_name);
    if(it != aliases.end())
        return it->second;
    return gfx_name;
}

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIGRAPHX_THROW("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties");
    return props.gcnArchName;
}

static bool gfx_has_fp8fnuz_intrinsics_impl(const std::string& gfx_name)
{
    return (starts_with(gfx_name, "gfx94"));
}

bool gfx_has_fp8fnuz_intrinsics()
{
    return gfx_has_fp8fnuz_intrinsics_impl(get_gfx_name(get_device_name()));
}

bool gfx_has_fp8fnuz_intrinsics(const context& ctx)
{
    return gfx_has_fp8fnuz_intrinsics_impl(
        get_gfx_name(ctx.get_current_device().get_device_name()));
}

static bool gfx_has_fp8ocp_intrinsics_impl(const std::string& gfx_name)
{
    bool is_navi_with_fp8ocp = starts_with(gfx_name, "gfx12") and gfx_name >= "gfx1200";
    bool is_mi_with_fp8ocp   = starts_with(gfx_name, "gfx9") and gfx_name >= "gfx950";
    return (is_navi_with_fp8ocp or is_mi_with_fp8ocp);
}

bool gfx_has_fp8ocp_intrinsics()
{
    return gfx_has_fp8ocp_intrinsics_impl(get_gfx_name(get_device_name()));
}

bool gfx_has_fp8ocp_intrinsics(const context& ctx)
{
    return gfx_has_fp8ocp_intrinsics_impl(get_gfx_name(ctx.get_current_device().get_device_name()));
}

static bool gfx_has_bf16_intrinsics_impl(const std::string& gfx_name)
{
    return not(starts_with(gfx_name, "gfx1030"));
}

bool gfx_has_bf16_intrinsics()
{
    return gfx_has_bf16_intrinsics_impl(get_gfx_name(get_device_name()));
}

bool gfx_has_bf16_intrinsics(const context& ctx)
{
    return gfx_has_bf16_intrinsics_impl(get_gfx_name(ctx.get_current_device().get_device_name()));
}

static bool gfx_has_mx_intrinsics_impl(const std::string& gfx_name)
{
    return starts_with(gfx_name, "gfx9") and gfx_name >= "gfx950";
}

bool gfx_has_mx_intrinsics() { return gfx_has_mx_intrinsics_impl(get_gfx_name(get_device_name())); }

bool gfx_has_mx_intrinsics(const context& ctx)
{
    return gfx_has_mx_intrinsics_impl(get_gfx_name(ctx.get_current_device().get_device_name()));
}

#if MIGRAPHX_USE_HIPBLASLT
static bool gfx_default_rocblas_impl(const std::string& gfx_name)
{
    return ((string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "hipblaslt")
                ? false
                : (gfx_name == "gfx90a"));
}

bool gfx_default_rocblas() { return gfx_default_rocblas_impl(get_gfx_name(get_device_name())); }

bool gfx_default_rocblas(const context& ctx)
{
    return gfx_default_rocblas_impl(get_gfx_name(ctx.get_current_device().get_device_name()));
}

static bool hipblaslt_supported_impl(const std::string& gfx_name)
{
    return (gfx_name == "gfx90a" or (starts_with(gfx_name, "gfx94") and gfx_name >= "gfx942") or
            (starts_with(gfx_name, "gfx95") and gfx_name >= "gfx950") or
            starts_with(gfx_name, "gfx110") or starts_with(gfx_name, "gfx120"));
}
#endif

bool hipblaslt_supported()
{
#if !MIGRAPHX_USE_HIPBLASLT
    return false;
#else
    return hipblaslt_supported_impl(get_gfx_name(get_device_name()));
#endif
}

bool hipblaslt_supported(const context& ctx)
{
#if !MIGRAPHX_USE_HIPBLASLT
    (void)ctx;
    return false;
#else
    return hipblaslt_supported_impl(get_gfx_name(ctx.get_current_device().get_device_name()));
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
