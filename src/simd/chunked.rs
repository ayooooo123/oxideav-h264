//! Hardware-SIMD (NEON + WASM SIMD128) and zero-zeroing chunked kernels
//! for H.264 §8.4.2.2.1 6-tap luma and §8.4.2.2.2 bilinear chroma motion compensation.

#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::inter_pred::{InterPredError, InterPredResult};
use core::mem::MaybeUninit;

#[inline(always)]
fn clip3(x: i32, y: i32, z: i32) -> i32 {
    if z < x {
        x
    } else if z > y {
        y
    } else {
        z
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
#[inline(always)]
fn clip1(x: i32, max_v: i32) -> i32 {
    x.clamp(0, max_v)
}

// ---------------------------------------------------------------------------
// 4-lane i32 hardware vector (AArch64 NEON / WASM32 SIMD128 / scalar fallback)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
mod v4 {
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct I32x4(int32x4_t);

    impl I32x4 {
        #[inline(always)]
        pub unsafe fn load(p: *const i32) -> Self {
            Self(unsafe { vld1q_s32(p) })
        }

        #[inline(always)]
        pub unsafe fn store(self, p: *mut i32) {
            unsafe { vst1q_s32(p, self.0) }
        }

        #[inline(always)]
        pub fn tap6(s0: Self, s1: Self, s2: Self, s3: Self, s4: Self, s5: Self) -> Self {
            unsafe {
                let outer = vaddq_s32(s0.0, s5.0);
                let mid = vaddq_s32(s1.0, s4.0);
                let inner = vaddq_s32(s2.0, s3.0);
                Self(vmlaq_n_s32(vmlsq_n_s32(outer, mid, 5), inner, 20))
            }
        }

        #[inline(always)]
        pub fn shr5_clip(self, max_v: i32) -> Self {
            unsafe {
                let r = vshrq_n_s32::<5>(vaddq_s32(self.0, vdupq_n_s32(16)));
                Self(vmaxq_s32(vdupq_n_s32(0), vminq_s32(vdupq_n_s32(max_v), r)))
            }
        }

        #[inline(always)]
        pub fn shr10_clip(self, max_v: i32) -> Self {
            unsafe {
                let r = vshrq_n_s32::<10>(vaddq_s32(self.0, vdupq_n_s32(512)));
                Self(vmaxq_s32(vdupq_n_s32(0), vminq_s32(vdupq_n_s32(max_v), r)))
            }
        }

        #[inline(always)]
        pub fn avg(self, other: Self) -> Self {
            unsafe { Self(vrhaddq_s32(self.0, other.0)) }
        }

        #[inline(always)]
        pub fn bilinear(
            sa: Self,
            sb: Self,
            sc: Self,
            sd: Self,
            c00: i32,
            c10: i32,
            c01: i32,
            c11: i32,
        ) -> Self {
            unsafe {
                let mut acc = vdupq_n_s32(32);
                acc = vmlaq_n_s32(acc, sa.0, c00);
                acc = vmlaq_n_s32(acc, sb.0, c10);
                acc = vmlaq_n_s32(acc, sc.0, c01);
                acc = vmlaq_n_s32(acc, sd.0, c11);
                Self(vshrq_n_s32::<6>(acc))
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod v4 {
    use core::arch::wasm32::*;

    #[derive(Clone, Copy)]
    pub struct I32x4(v128);

    impl I32x4 {
        #[inline(always)]
        pub unsafe fn load(p: *const i32) -> Self {
            Self(unsafe { v128_load(p.cast()) })
        }

        #[inline(always)]
        pub unsafe fn store(self, p: *mut i32) {
            unsafe { v128_store(p.cast(), self.0) }
        }

        #[inline(always)]
        pub fn tap6(s0: Self, s1: Self, s2: Self, s3: Self, s4: Self, s5: Self) -> Self {
            let outer = i32x4_add(s0.0, s5.0);
            let mid = i32x4_mul(i32x4_add(s1.0, s4.0), i32x4_splat(5));
            let inner = i32x4_mul(i32x4_add(s2.0, s3.0), i32x4_splat(20));
            Self(i32x4_add(i32x4_sub(outer, mid), inner))
        }

        #[inline(always)]
        pub fn shr5_clip(self, max_v: i32) -> Self {
            let r = i32x4_shr(i32x4_add(self.0, i32x4_splat(16)), 5);
            Self(i32x4_max(i32x4_splat(0), i32x4_min(i32x4_splat(max_v), r)))
        }

        #[inline(always)]
        pub fn shr10_clip(self, max_v: i32) -> Self {
            let r = i32x4_shr(i32x4_add(self.0, i32x4_splat(512)), 10);
            Self(i32x4_max(i32x4_splat(0), i32x4_min(i32x4_splat(max_v), r)))
        }

        #[inline(always)]
        pub fn avg(self, other: Self) -> Self {
            Self(i32x4_shr(
                i32x4_add(i32x4_add(self.0, other.0), i32x4_splat(1)),
                1,
            ))
        }

        #[inline(always)]
        pub fn bilinear(
            sa: Self,
            sb: Self,
            sc: Self,
            sd: Self,
            c00: i32,
            c10: i32,
            c01: i32,
            c11: i32,
        ) -> Self {
            let a = i32x4_add(
                i32x4_mul(sa.0, i32x4_splat(c00)),
                i32x4_mul(sb.0, i32x4_splat(c10)),
            );
            let b = i32x4_add(
                i32x4_mul(sc.0, i32x4_splat(c01)),
                i32x4_mul(sd.0, i32x4_splat(c11)),
            );
            Self(i32x4_shr(i32x4_add(i32x4_add(a, b), i32x4_splat(32)), 6))
        }
    }
}

#[cfg(not(any(
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
)))]
mod v4 {
    #[derive(Clone, Copy)]
    pub struct I32x4([i32; 4]);

    impl I32x4 {
        #[inline(always)]
        pub unsafe fn load(p: *const i32) -> Self {
            Self(unsafe { *(p as *const [i32; 4]) })
        }

        #[inline(always)]
        pub unsafe fn store(self, p: *mut i32) {
            unsafe { *(p as *mut [i32; 4]) = self.0 }
        }

        #[inline(always)]
        pub fn tap6(s0: Self, s1: Self, s2: Self, s3: Self, s4: Self, s5: Self) -> Self {
            Self(core::array::from_fn(|k| {
                (s0.0[k] + s5.0[k]) - 5 * (s1.0[k] + s4.0[k]) + 20 * (s2.0[k] + s3.0[k])
            }))
        }

        #[inline(always)]
        pub fn shr5_clip(self, max_v: i32) -> Self {
            Self(self.0.map(|v| super::clip1((v + 16) >> 5, max_v)))
        }

        #[inline(always)]
        pub fn shr10_clip(self, max_v: i32) -> Self {
            Self(self.0.map(|v| super::clip1((v + 512) >> 10, max_v)))
        }

        #[inline(always)]
        pub fn avg(self, other: Self) -> Self {
            Self(core::array::from_fn(|k| (self.0[k] + other.0[k] + 1) >> 1))
        }

        #[inline(always)]
        pub fn bilinear(
            sa: Self,
            sb: Self,
            sc: Self,
            sd: Self,
            c00: i32,
            c10: i32,
            c01: i32,
            c11: i32,
        ) -> Self {
            Self(core::array::from_fn(|k| {
                (c00 * sa.0[k] + c10 * sb.0[k] + c01 * sc.0[k] + c11 * sd.0[k] + 32) >> 6
            }))
        }
    }
}

use v4::I32x4;

const MAX_H: usize = 16 + 5;
const MAX_W: usize = 16;
const MAX_ROW_INT: usize = MAX_W + 5;

/// Materialize `n` integer samples starting at `(int_x - 2, iy)` into `out[..n]` with edge clamping.
#[inline(always)]
fn fill_row_halo(
    out: &mut [i32; MAX_ROW_INT],
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    iy: usize,
    x_start: i32,
    n: usize,
) {
    let row_base = iy * src_stride;
    let xmax = src_w as i32 - 1;
    if x_start >= 0 && x_start + n as i32 - 1 <= xmax {
        let start = row_base + x_start as usize;
        out[..n].copy_from_slice(&src[start..start + n]);
    } else {
        for k in 0..n {
            let cx = clip3(0, xmax, x_start + k as i32) as usize;
            out[k] = src[row_base + cx];
        }
    }
}

#[inline(always)]
unsafe fn h_tap6_ptr(p: *const i32) -> I32x4 {
    unsafe {
        I32x4::tap6(
            I32x4::load(p),
            I32x4::load(p.add(1)),
            I32x4::load(p.add(2)),
            I32x4::load(p.add(3)),
            I32x4::load(p.add(4)),
            I32x4::load(p.add(5)),
        )
    }
}

#[inline(always)]
unsafe fn v_tap6_strided(p: *const i32, stride: usize) -> I32x4 {
    unsafe {
        I32x4::tap6(
            I32x4::load(p),
            I32x4::load(p.add(stride)),
            I32x4::load(p.add(2 * stride)),
            I32x4::load(p.add(3 * stride)),
            I32x4::load(p.add(4 * stride)),
            I32x4::load(p.add(5 * stride)),
        )
    }
}

/// Build `(h + 5)` rows of `w` horizontal 6-tap intermediates (`b1` values, unrounded) without zero-init waste.
#[inline]
fn build_h_strip(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    int_x: i32,
    int_y: i32,
    w: usize,
    h: usize,
    h_strip: &mut [[MaybeUninit<i32>; MAX_W]; MAX_H],
) {
    let nrows = h + 5;
    let x0 = int_x - 2;
    let x_inside = x0 >= 0 && (x0 + (w + 5) as i32) <= src_w as i32;
    let mut row_buf = [0i32; MAX_ROW_INT];
    for r in 0..nrows {
        let iy = clip3(0, src_h as i32 - 1, int_y + r as i32 - 2) as usize;
        let out_ptr = h_strip[r].as_mut_ptr().cast::<i32>();
        if x_inside {
            let sp = unsafe { src.as_ptr().add(iy * src_stride + x0 as usize) };
            let mut x = 0;
            while x < w {
                unsafe { h_tap6_ptr(sp.add(x)).store(out_ptr.add(x)) };
                x += 4;
            }
        } else {
            fill_row_halo(&mut row_buf, src, src_stride, src_w, iy, x0, w + 5);
            let sp = row_buf.as_ptr();
            let mut x = 0;
            while x < w {
                unsafe { h_tap6_ptr(sp.add(x)).store(out_ptr.add(x)) };
                x += 4;
            }
        }
    }
}

/// Build `(h + 5)` rows of `w` clamped integer samples at `x_base .. x_base + w`.
#[inline]
fn build_col_strip(
    src: &[i32],
    src_stride: usize,
    src_w: usize,
    src_h: usize,
    x_base: i32,
    int_y: i32,
    w: usize,
    h: usize,
    col_strip: &mut [[MaybeUninit<i32>; MAX_W]; MAX_H],
) {
    let nrows = h + 5;
    let xmax = src_w as i32 - 1;
    let inside_x = x_base >= 0 && x_base + w as i32 - 1 <= xmax;
    for r in 0..nrows {
        let iy = clip3(0, src_h as i32 - 1, int_y + r as i32 - 2) as usize;
        let row_base = iy * src_stride;
        let dst_ptr = col_strip[r].as_mut_ptr().cast::<i32>();
        if inside_x {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    src.as_ptr().add(row_base + x_base as usize),
                    dst_ptr,
                    w,
                );
            }
        } else {
            for i in 0..w {
                let cx = clip3(0, xmax, x_base + i as i32) as usize;
                unsafe { *dst_ptr.add(i) = src[row_base + cx] };
            }
        }
    }
}

/// §8.4.2.2.1 — luma fractional-sample interpolation.
pub fn interpolate_luma(
    src: &[i32],
    src_stride: usize,
    src_width: usize,
    src_height: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) -> InterPredResult<()> {
    if x_frac > 3 {
        return Err(InterPredError::LumaFracOutOfRange(x_frac));
    }
    if y_frac > 3 {
        return Err(InterPredError::LumaFracOutOfRange(y_frac));
    }
    if w == 0 || h == 0 || !w.is_multiple_of(4) || !h.is_multiple_of(4) {
        return Err(InterPredError::InvalidBlockSize { w, h });
    }
    if src_width == 0 || src_height == 0 {
        return Err(InterPredError::ZeroDimRef {
            src_width,
            src_height,
        });
    }
    let wu = w as usize;
    let hu = h as usize;
    if wu > MAX_W || hu > 16 {
        return super::scalar::interpolate_luma(
            src, src_stride, src_width, src_height, int_x, int_y, x_frac, y_frac, w, h, bit_depth,
            dst, dst_stride,
        );
    }
    let max_v = (1i32 << bit_depth) - 1;
    let xmax = src_width as i32 - 1;
    let ymax = src_height as i32 - 1;

    // 1. Pure integer (0, 0).
    if x_frac == 0 && y_frac == 0 {
        let inside_x = int_x >= 0 && int_x + wu as i32 - 1 <= xmax;
        for yl in 0..hu {
            let iy = clip3(0, ymax, int_y + yl as i32) as usize;
            let row_base = iy * src_stride;
            let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
            if inside_x {
                drow.copy_from_slice(&src[row_base + int_x as usize..row_base + int_x as usize + wu]);
            } else {
                for (i, slot) in drow.iter_mut().enumerate() {
                    *slot = src[row_base + clip3(0, xmax, int_x + i as i32) as usize];
                }
            }
        }
        return Ok(());
    }

    // 2. Pure horizontal positions: (2, 0), (1, 0), (3, 0) — zero strip memory needed!
    if y_frac == 0 {
        let x0 = int_x - 2;
        let inside_x = x0 >= 0 && (x0 + (wu + 5) as i32) <= src_width as i32;
        let mut row_buf = [0i32; MAX_ROW_INT];
        let g_off = if x_frac == 3 { 3usize } else { 2usize }; // offset inside [int_x - 2 ..]
        for yl in 0..hu {
            let iy = clip3(0, ymax, int_y + yl as i32) as usize;
            let drow_ptr = dst[yl * dst_stride..yl * dst_stride + wu].as_mut_ptr();
            let sp = if inside_x {
                unsafe { src.as_ptr().add(iy * src_stride + x0 as usize) }
            } else {
                fill_row_halo(&mut row_buf, src, src_stride, src_width, iy, x0, wu + 5);
                row_buf.as_ptr()
            };
            let mut x = 0;
            while x < wu {
                unsafe {
                    let b_s = h_tap6_ptr(sp.add(x)).shr5_clip(max_v);
                    let out = if x_frac == 2 {
                        b_s
                    } else {
                        let g = I32x4::load(sp.add(x + g_off));
                        b_s.avg(g)
                    };
                    out.store(drow_ptr.add(x));
                }
                x += 4;
            }
        }
        return Ok(());
    }

    // 3. Pure vertical positions: (0, 2), (0, 1), (0, 3) — direct in-place SIMD when inside!
    if x_frac == 0 {
        let inside = int_x >= 0
            && int_x + wu as i32 - 1 <= xmax
            && int_y >= 2
            && int_y + hu as i32 + 2 <= ymax
            && src.len() >= (int_y as usize + hu + 3 - 1) * src_stride + int_x as usize + wu;
        let g_row_off = if y_frac == 3 { 3usize } else { 2usize };
        if inside {
            for yl in 0..hu {
                let top_y = (int_y as usize + yl) - 2;
                let base_ptr = unsafe { src.as_ptr().add(top_y * src_stride + int_x as usize) };
                let drow_ptr = dst[yl * dst_stride..yl * dst_stride + wu].as_mut_ptr();
                let mut x = 0;
                while x < wu {
                    unsafe {
                        let p = base_ptr.add(x);
                        let h_s = v_tap6_strided(p, src_stride).shr5_clip(max_v);
                        let out = if y_frac == 2 {
                            h_s
                        } else {
                            let g = I32x4::load(p.add(g_row_off * src_stride));
                            h_s.avg(g)
                        };
                        out.store(drow_ptr.add(x));
                    }
                    x += 4;
                }
            }
            return Ok(());
        }
        let mut col_strip = [[MaybeUninit::<i32>::uninit(); MAX_W]; MAX_H];
        build_col_strip(
            src,
            src_stride,
            src_width,
            src_height,
            int_x,
            int_y,
            wu,
            hu,
            &mut col_strip,
        );
        let strip_ptr = col_strip.as_ptr().cast::<i32>();
        for yl in 0..hu {
            let drow_ptr = dst[yl * dst_stride..yl * dst_stride + wu].as_mut_ptr();
            let mut x = 0;
            while x < wu {
                unsafe {
                    let p = strip_ptr.add(yl * MAX_W + x);
                    let h_s = v_tap6_strided(p, MAX_W).shr5_clip(max_v);
                    let out = if y_frac == 2 {
                        h_s
                    } else {
                        let g = I32x4::load(p.add(g_row_off * MAX_W));
                        h_s.avg(g)
                    };
                    out.store(drow_ptr.add(x));
                }
                x += 4;
            }
        }
        return Ok(());
    }

    // 4. All remaining positions need the horizontal 6-tap strip (no zero-init overhead).
    let mut h_strip = [[MaybeUninit::<i32>::uninit(); MAX_W]; MAX_H];
    build_h_strip(
        src,
        src_stride,
        src_width,
        src_height,
        int_x,
        int_y,
        wu,
        hu,
        &mut h_strip,
    );
    let hs_ptr = h_strip.as_ptr().cast::<i32>();

    // (2, 2) = j, (2, 1) = avg(b, j), (2, 3) = avg(j, s): derived purely from h_strip!
    if x_frac == 2 {
        for yl in 0..hu {
            let drow_ptr = dst[yl * dst_stride..yl * dst_stride + wu].as_mut_ptr();
            let mut x = 0;
            while x < wu {
                unsafe {
                    let p = hs_ptr.add(yl * MAX_W + x);
                    let j_s = v_tap6_strided(p, MAX_W).shr10_clip(max_v);
                    let out = match y_frac {
                        2 => j_s,
                        1 => {
                            let b_s = I32x4::load(p.add(2 * MAX_W)).shr5_clip(max_v);
                            b_s.avg(j_s)
                        }
                        _ => {
                            let s_s = I32x4::load(p.add(3 * MAX_W)).shr5_clip(max_v);
                            j_s.avg(s_s)
                        }
                    };
                    out.store(drow_ptr.add(x));
                }
                x += 4;
            }
        }
        return Ok(());
    }

    // Remaining 6 quarter-pel positions: x_frac in {1, 3}, y_frac in {1, 2, 3}.
    // Need vertical 6-tap at `col_x = int_x` (when x_frac == 1 -> `h`) or `int_x + 1` (when x_frac == 3 -> `m`).
    let col_x = int_x + i32::from(x_frac == 3);
    let inside_col = col_x >= 0
        && col_x + wu as i32 - 1 <= xmax
        && int_y >= 2
        && int_y + hu as i32 + 2 <= ymax
        && src.len() >= (int_y as usize + hu + 3 - 1) * src_stride + col_x as usize + wu;

    let mut col_strip = [[MaybeUninit::<i32>::uninit(); MAX_W]; MAX_H];
    if !inside_col {
        build_col_strip(
            src,
            src_stride,
            src_width,
            src_height,
            col_x,
            int_y,
            wu,
            hu,
            &mut col_strip,
        );
    }
    let cs_ptr = col_strip.as_ptr().cast::<i32>();

    for yl in 0..hu {
        let drow_ptr = dst[yl * dst_stride..yl * dst_stride + wu].as_mut_ptr();
        let top_y = if inside_col {
            (int_y as usize + yl) - 2
        } else {
            0
        };
        let mut x = 0;
        while x < wu {
            unsafe {
                let hm_s = if inside_col {
                    v_tap6_strided(
                        src.as_ptr().add(top_y * src_stride + col_x as usize + x),
                        src_stride,
                    )
                    .shr5_clip(max_v)
                } else {
                    v_tap6_strided(cs_ptr.add(yl * MAX_W + x), MAX_W).shr5_clip(max_v)
                };
                let hp = hs_ptr.add(yl * MAX_W + x);
                let other = match y_frac {
                    1 => I32x4::load(hp.add(2 * MAX_W)).shr5_clip(max_v), // b
                    2 => v_tap6_strided(hp, MAX_W).shr10_clip(max_v),     // j
                    _ => I32x4::load(hp.add(3 * MAX_W)).shr5_clip(max_v), // s
                };
                hm_s.avg(other).store(drow_ptr.add(x));
            }
            x += 4;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// §8.4.2.2.2 — chroma bilinear interpolation
// ---------------------------------------------------------------------------

pub fn interpolate_chroma(
    src: &[i32],
    src_stride: usize,
    src_width: usize,
    src_height: usize,
    int_x: i32,
    int_y: i32,
    x_frac: u8,
    y_frac: u8,
    w: u32,
    h: u32,
    _bit_depth: u32,
    dst: &mut [i32],
    dst_stride: usize,
) -> InterPredResult<()> {
    if x_frac > 7 {
        return Err(InterPredError::ChromaFracOutOfRange(x_frac));
    }
    if y_frac > 7 {
        return Err(InterPredError::ChromaFracOutOfRange(y_frac));
    }
    if w == 0 || h == 0 {
        return Err(InterPredError::InvalidBlockSize { w, h });
    }
    if src_width == 0 || src_height == 0 {
        return Err(InterPredError::ZeroDimRef {
            src_width,
            src_height,
        });
    }
    let wu = w as usize;
    let hu = h as usize;
    let xf = x_frac as i32;
    let yf = y_frac as i32;
    let w8mxf = 8 - xf;
    let w8myf = 8 - yf;
    let xmax = src_width as i32 - 1;
    let ymax = src_height as i32 - 1;

    if xf == 0 && yf == 0 && int_x >= 0 && int_y >= 0 && int_x + wu as i32 - 1 <= xmax && int_y + hu as i32 - 1 <= ymax {
        for yl in 0..hu {
            let iy = (int_y + yl as i32) as usize;
            let row_base = iy * src_stride + int_x as usize;
            dst[yl * dst_stride..yl * dst_stride + wu]
                .copy_from_slice(&src[row_base..row_base + wu]);
        }
        return Ok(());
    }

    let inside = int_x >= 0
        && int_y >= 0
        && (int_x + wu as i32) <= xmax
        && (int_y + hu as i32) <= ymax
        && src.len() >= (int_y as usize + hu) * src_stride + int_x as usize + wu + 1;

    if inside {
        let c00 = w8mxf * w8myf;
        let c10 = xf * w8myf;
        let c01 = w8mxf * yf;
        let c11 = xf * yf;
        let end4 = wu & !3;
        for yl in 0..hu {
            let iy = (int_y + yl as i32) as usize;
            let row_a = iy * src_stride + int_x as usize;
            let row_c = (iy + 1) * src_stride + int_x as usize;
            let drow = &mut dst[yl * dst_stride..yl * dst_stride + wu];
            let mut x = 0;
            while x < end4 {
                unsafe {
                    let pa = src.as_ptr().add(row_a + x);
                    let pc = src.as_ptr().add(row_c + x);
                    I32x4::bilinear(
                        I32x4::load(pa),
                        I32x4::load(pa.add(1)),
                        I32x4::load(pc),
                        I32x4::load(pc.add(1)),
                        c00,
                        c10,
                        c01,
                        c11,
                    )
                    .store(drow.as_mut_ptr().add(x));
                }
                x += 4;
            }
            for i in end4..wu {
                let sa = src[row_a + i];
                let sb = src[row_a + i + 1];
                let sc = src[row_c + i];
                let sd = src[row_c + i + 1];
                drow[i] = (c00 * sa + c10 * sb + c01 * sc + c11 * sd + 32) >> 6;
            }
        }
        return Ok(());
    }

    for yl in 0..hu {
        for xl in 0..wu {
            let ix = int_x + xl as i32;
            let iy = int_y + yl as i32;
            let cxa = clip3(0, xmax, ix) as usize;
            let cxb = clip3(0, xmax, ix + 1) as usize;
            let cya = clip3(0, ymax, iy) as usize;
            let cyc = clip3(0, ymax, iy + 1) as usize;
            let sa = src[cya * src_stride + cxa];
            let sb = src[cya * src_stride + cxb];
            let sc = src[cyc * src_stride + cxa];
            let sd = src[cyc * src_stride + cxb];
            dst[yl * dst_stride + xl] =
                (w8mxf * w8myf * sa + xf * w8myf * sb + w8mxf * yf * sc + xf * yf * sd + 32) >> 6;
        }
    }
    Ok(())
}
