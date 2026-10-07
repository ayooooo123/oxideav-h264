// SPDX-License-Identifier: LGPL-2.1-or-later
// Port of FFmpeg 2da55bf libavcodec/h264_slice.c (h264_field_start's
// recovery point and IDR marking, h264_select_output_frame's reorder
// depth estimate and output gate), libavcodec/h264_refs.c
// (ff_h264_execute_ref_pic_marking's unmarked random access point
// heuristic and its MMCO 5 reset) and libavcodec/h264dec.c
// (decode_nal_units' per-packet SEI reset and has_recovery_point,
// send_next_delayed_frame, ff_h264_flush_change, h264_decode_flush).
// Copyright (c) 2003 Michael Niedermayer <michaelni@gmx.at>
//
// This file is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 2.1, or (at your option) any later version.
// It is distributed WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See LICENSE-LGPL.

//! Which decoded pictures FFmpeg's H.264 decoder outputs.
//!
//! A stream entered anywhere but an IDR picture (a transport stream cut
//! mid-GOP, a seek) decodes pictures whose references were never seen.
//! FFmpeg marks a picture *recovered* at an IDR picture, at the frame a
//! recovery point SEI names, or at an I picture whose reference state
//! makes it an unmarked random access point, and outputs nothing before
//! the first recovered picture (unless asked to with `-flags
//! output_corrupt` or `-flags2 showall`, which the player never does).
//!
//! The decoder drives this state at the points FFmpeg does: per packet,
//! per SEI, per IDR slice, at each picture's first slice, after a
//! reference picture's marking, and at each picture's turn in output
//! order. FFmpeg updates `frame_recovered` when *its* decoder outputs a
//! picture; this decoder outputs in the same order, but its §C.4 output
//! window can be longer, so a picture decoded between FFmpeg's output of
//! a heuristic random access point and this decoder's can be judged
//! differently when the stream has B pictures. IDR and recovery point
//! decisions do not depend on that timing.

use crate::ref_list::{DpbEntry, MmcoOp, RefMarking};

/// The picture is an IDR picture, or follows one in decoding order
/// (`FRAME_RECOVERED_IDR`).
pub(crate) const RECOVERED_IDR: u8 = 1 << 0;
/// The picture is at, or displayed after, the frame a recovery point SEI
/// names (`FRAME_RECOVERED_SEI`).
pub(crate) const RECOVERED_SEI: u8 = 1 << 1;
/// The picture is an unmarked random access point, or was decoded after
/// one was output (`FRAME_RECOVERED_HEURISTIC`).
pub(crate) const RECOVERED_HEURISTIC: u8 = 1 << 2;

/// `recovery_frame_cnt` values FFmpeg rejects (h264_sei.c:153,
/// `1 << MAX_LOG2_MAX_FRAME_NUM`).
const MAX_RECOVERY_FRAME_CNT: u32 = 1 << 16;

/// `H264_MAX_DPB_FRAMES`: the length of FFmpeg's POC history.
const MAX_DPB_FRAMES: usize = 16;

/// Whether FFmpeg's marking of a picture with these MMCOs fails
/// (h264_refs.c:637-648): an MMCO 1 or 3 whose short-term picture
/// `find_short` cannot find in the reference state before the marking,
/// as when the stream was entered after that picture. FFmpeg's
/// `short_pic_num` is `(CurrPicNum − difference_of_pic_nums_minus1 − 1)
/// mod MaxPicNum` (h264_refs.c:858), the picture it names has that
/// FrameNum (half of it for a field), and a field form unreferences one
/// field, leaving the picture until both are gone.
pub(crate) fn mmco_target_missing(
    dpb: &[DpbEntry],
    ops: &[MmcoOp],
    field: bool,
    current_bottom: bool,
    frame_num: u32,
    log2_max_frame_num: u32,
) -> bool {
    let max_frame_num = 1u32 << log2_max_frame_num.min(16);
    let (curr_pic_num, max_pic_num) = if field {
        (2 * frame_num + 1, 2 * max_frame_num)
    } else {
        (frame_num, max_frame_num)
    };
    // (FrameNum, parities unreferenced so far: bit 0 top, bit 1 bottom).
    let mut unreferenced: Vec<(u32, u8)> = Vec::new();
    for op in ops {
        let (difference, long_term_frame_idx) = match *op {
            MmcoOp::MarkShortTermUnused(difference) => (difference, None),
            MmcoOp::AssignLongTerm(difference, idx) => (difference, Some(idx)),
            MmcoOp::MarkAllUnused => return false,
            _ => continue,
        };
        let pic_num = curr_pic_num.wrapping_sub(difference).wrapping_sub(1) & (max_pic_num - 1);
        let target = if field { pic_num >> 1 } else { pic_num };
        let parities = if !field || long_term_frame_idx.is_some() {
            // A frame, or MMCO 3, which moves the whole picture to the
            // long-term list.
            0b11
        } else {
            // An even PicNum names the field of the other parity.
            let bottom = current_bottom == (pic_num & 1 == 1);
            if bottom {
                0b10
            } else {
                0b01
            }
        };
        let gone = unreferenced.iter().any(|&(f, p)| f == target && p == 0b11);
        let found = !gone && dpb.iter().any(|e| e.frame_num == target && e.any_field_is(RefMarking::ShortTerm));
        if found {
            match unreferenced.iter_mut().find(|(f, _)| *f == target) {
                Some((_, p)) => *p |= parities,
                None => unreferenced.push((target, parities)),
            }
            continue;
        }
        // MMCO 3 naming a picture already at that long-term index is
        // not a failure.
        let already_long = long_term_frame_idx.is_some_and(|idx| {
            dpb.iter()
                .any(|e| e.frame_num == target && e.long_term_frame_idx == idx && e.any_field_is(RefMarking::LongTerm))
        });
        if !already_long {
            return true;
        }
    }
    false
}

/// What the unmarked random access point heuristic reads after a
/// reference picture's marking (h264_refs.c:815-826).
pub(crate) struct MarkedPicture {
    /// Reference frames (or field pairs) marked short-term, the current
    /// picture included.
    pub short_refs: usize,
    /// Reference frames (or field pairs) marked long-term.
    pub long_refs: usize,
    /// The largest `num_ref_idx_l0_default_active_minus1 + 1` and
    /// `num_ref_idx_l1_default_active_minus1 + 1` of every stored PPS.
    pub pps_ref_count: [u32; 2],
    /// The current picture is a field.
    pub field: bool,
    /// The first slice of the current frame is an I slice (not SI).
    pub intra: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Recovery {
    /// `h->sei.recovery_point.recovery_frame_cnt` (`None` for -1): the
    /// recovery point SEI of the current packet.
    sei_recovery_frame_cnt: Option<u32>,
    /// `h->recovery_frame`: the `frame_num` whose reference picture is
    /// recovered.
    recovery_frame: Option<u32>,
    /// `h->valid_recovery_point`.
    valid_recovery_point: bool,
    /// `h->frame_recovered`.
    frame_recovered: u8,
    /// `h->has_recovery_point`: an IDR slice or a recovery point SEI was
    /// seen.
    has_recovery_point: bool,
    /// `avctx->has_b_frames`: FFmpeg's reorder depth estimate.
    has_b_frames: u32,
    /// `h->last_pocs`, ascending; `i32::MIN` where unset.
    last_pocs: [i32; MAX_DPB_FRAMES],
}

impl Default for Recovery {
    fn default() -> Self {
        Self {
            sei_recovery_frame_cnt: None,
            recovery_frame: None,
            valid_recovery_point: false,
            frame_recovered: 0,
            has_recovery_point: false,
            has_b_frames: 0,
            last_pocs: [i32::MIN; MAX_DPB_FRAMES],
        }
    }
}

impl Recovery {
    /// decode_nal_units (h264dec.c:617-623): a packet starts without the
    /// previous packet's SEI unless it completes a field pair.
    pub(crate) fn packet_start(&mut self, awaiting_second_field: bool) {
        if !awaiting_second_field {
            self.sei_recovery_frame_cnt = None;
        }
    }

    /// decode_recovery_point (h264_sei.c:149-162) and h264dec.c:745.
    pub(crate) fn recovery_point_sei(&mut self, recovery_frame_cnt: u32) {
        if recovery_frame_cnt < MAX_RECOVERY_FRAME_CNT {
            self.sei_recovery_frame_cnt = Some(recovery_frame_cnt);
        }
        self.has_recovery_point |= self.sei_recovery_frame_cnt.is_some();
    }

    /// h264dec.c:672: every IDR slice NAL unit.
    pub(crate) fn idr_slice(&mut self) {
        self.has_recovery_point = true;
    }

    /// `avcodec_parameters_to_context` (`has_b_frames = video_delay`):
    /// FFmpeg's decoder starts from the reorder depth its demuxer's
    /// `avformat_find_stream_info` measured.
    pub(crate) fn set_container_reorder_depth(&mut self, video_delay: u32) {
        self.has_b_frames = video_delay.min(MAX_DPB_FRAMES as u32);
    }

    /// h264_field_start (h264_slice.c:1429-1432), at every picture or
    /// field: an SPS bitstream restriction raises the reorder depth.
    pub(crate) fn field_start(&mut self, num_reorder_frames: Option<u32>) {
        if let Some(n) = num_reorder_frames {
            self.has_b_frames = self.has_b_frames.max(n);
        }
    }

    /// h264_field_start (h264_slice.c:1677-1707): the recovery flags of
    /// the picture or field opened by a slice with this `frame_num`
    /// (`intra`: an I or SI slice). A field pair's frame carries the
    /// union of its fields' flags.
    pub(crate) fn picture_start(
        &mut self,
        frame_num: u32,
        log2_max_frame_num: u32,
        intra: bool,
        idr: bool,
        reference: bool,
    ) -> u8 {
        let mask = (1u32 << log2_max_frame_num.min(16)) - 1;
        if let Some(count) = self.sei_recovery_frame_cnt {
            if frame_num != count || !intra {
                self.valid_recovery_point = true;
            }
            let replace = match self.recovery_frame {
                None => true,
                Some(frame) => (frame.wrapping_sub(frame_num) & mask) > count,
            };
            if replace {
                self.recovery_frame = Some(if self.valid_recovery_point {
                    frame_num.wrapping_add(count) & mask
                } else {
                    frame_num
                });
            }
        }
        let mut recovered = 0;
        if idr {
            recovered |= RECOVERED_IDR;
            self.frame_recovered |= RECOVERED_IDR;
        }
        if self.recovery_frame == Some(frame_num) && reference {
            self.recovery_frame = None;
            recovered |= RECOVERED_SEI;
        }
        recovered | self.frame_recovered
    }

    /// The reorder depth estimate of h264_field_start (h264_slice.c:
    /// 1429-1432) and h264_select_output_frame (:1323-1351), once per
    /// frame (at a field pair's second field): `poc` is the frame's
    /// PicOrderCnt, `b_frame` its first slice is a B slice,
    /// `num_reorder_frames` the SPS's VUI value when it carries a
    /// bitstream restriction.
    pub(crate) fn frame_output_order(&mut self, poc: i32, b_frame: bool, num_reorder_frames: Option<u32>) {
        if let Some(n) = num_reorder_frames {
            self.has_b_frames = self.has_b_frames.max(n);
        }
        let mut i = 0;
        loop {
            if i == MAX_DPB_FRAMES || poc < self.last_pocs[i] {
                if i > 0 {
                    self.last_pocs[i - 1] = poc;
                }
                break;
            } else if i > 0 {
                self.last_pocs[i - 1] = self.last_pocs[i];
            }
            i += 1;
        }
        let mut out_of_order = MAX_DPB_FRAMES - i;
        let last = self.last_pocs[MAX_DPB_FRAMES - 1] as i64;
        let before_last = self.last_pocs[MAX_DPB_FRAMES - 2];
        if b_frame || (before_last > i32::MIN && last - before_last as i64 > 2) {
            out_of_order = out_of_order.max(1);
        }
        if out_of_order == MAX_DPB_FRAMES {
            self.reset_poc_history();
            self.last_pocs[0] = poc;
        } else if (self.has_b_frames as usize) < out_of_order && num_reorder_frames.is_none() {
            self.has_b_frames = out_of_order as u32;
        }
    }

    /// h264_refs.c:729-730 (MMCO 5) and the other places FFmpeg forgets
    /// its POC history.
    pub(crate) fn reset_poc_history(&mut self) {
        self.last_pocs = [i32::MIN; MAX_DPB_FRAMES];
    }

    /// ff_h264_execute_ref_pic_marking (h264_refs.c:815-826), after a
    /// reference picture's marking: the recovery flag an unmarked random
    /// access point earns.
    pub(crate) fn marked(&mut self, pic: &MarkedPicture) -> u8 {
        let field = u32::from(pic.field);
        let [ref0, ref1] = pic.pps_ref_count;
        let few_refs = pic.short_refs <= 2
            || (ref0 <= 2 && ref1 <= 1 && self.has_b_frames > 0)
            || (ref0 <= 1 + field && ref1 <= 1);
        if pic.long_refs == 0
            && few_refs
            && ref0 <= 2 + field + 2 * u32::from(!self.has_recovery_point)
            && pic.intra
        {
            if self.has_b_frames == 0 {
                self.frame_recovered |= RECOVERED_HEURISTIC;
            }
            return RECOVERED_HEURISTIC;
        }
        0
    }

    /// h264_select_output_frame (h264_slice.c:1389-1401) and
    /// send_next_delayed_frame (h264dec.c:1057-1058): whether a picture
    /// with these flags is output at its turn in output order.
    pub(crate) fn output(&mut self, recovered: u8) -> bool {
        self.frame_recovered |= recovered;
        recovered | (self.frame_recovered & RECOVERED_SEI) != 0
    }

    /// ff_h264_flush_change (h264dec.c:470-472): a decoder flush, or a
    /// new SPS that reinitialises the decoder (h264_slice.c:1167-1169).
    pub(crate) fn flush_change(&mut self) {
        self.recovery_frame = None;
        self.frame_recovered = 0;
        self.reset_poc_history();
    }

    /// h264_decode_flush (h264dec.c:480-486): a seek.
    pub(crate) fn seek(&mut self) {
        self.flush_change();
        self.sei_recovery_frame_cnt = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idr_recovers_every_later_picture() {
        let mut r = Recovery::default();
        assert_eq!(r.picture_start(3, 4, false, false, true), 0);
        assert!(!r.output(0));
        let idr = r.picture_start(0, 4, true, true, true);
        assert_eq!(idr, RECOVERED_IDR);
        assert_eq!(r.picture_start(1, 4, false, false, true), RECOVERED_IDR);
    }

    #[test]
    fn recovery_point_recovers_from_its_frame_on() {
        let mut r = Recovery::default();
        r.packet_start(false);
        r.recovery_point_sei(2);
        // frame_num 5 + 2: frame 7 is the recovery frame.
        assert_eq!(r.picture_start(5, 4, false, false, true), 0);
        r.packet_start(false);
        assert_eq!(r.picture_start(6, 4, false, false, true), 0);
        assert!(!r.output(0));
        r.packet_start(false);
        let recovery = r.picture_start(7, 4, false, false, true);
        assert_eq!(recovery, RECOVERED_SEI);
        // A picture decoded before the recovery frame but output after
        // it is recovered at its turn.
        assert!(r.output(recovery));
        assert!(r.output(0));
    }

    #[test]
    fn recovery_frame_wraps_with_max_frame_num() {
        let mut r = Recovery::default();
        r.recovery_point_sei(3);
        // log2_max_frame_num 4: frame_num 14 + 3 wraps to 1.
        assert_eq!(r.picture_start(14, 4, false, false, true), 0);
        r.packet_start(false);
        assert_eq!(r.picture_start(15, 4, false, false, true), 0);
        assert_eq!(r.picture_start(0, 4, false, false, true), 0);
        assert_eq!(r.picture_start(1, 4, false, false, true), RECOVERED_SEI);
    }

    #[test]
    fn a_recovery_point_counting_to_its_own_i_frame_recovers_it() {
        // recovery_frame_cnt equal to the I frame's frame_num: FFmpeg
        // keeps the I frame itself as the recovery frame.
        let mut r = Recovery::default();
        r.recovery_point_sei(4);
        assert_eq!(r.picture_start(4, 4, true, false, true), RECOVERED_SEI);
    }

    #[test]
    fn heuristic_needs_an_i_picture_with_few_references() {
        let pic = |short_refs, intra| MarkedPicture {
            short_refs,
            long_refs: 0,
            pps_ref_count: [1, 1],
            field: false,
            intra,
        };
        let mut r = Recovery::default();
        assert_eq!(r.marked(&pic(1, false)), 0);
        assert_eq!(r.marked(&pic(1, true)), RECOVERED_HEURISTIC);
        // Without reordering the flag recovers later pictures at once.
        assert_eq!(r.picture_start(1, 4, false, false, true), RECOVERED_HEURISTIC);
        let mut r = Recovery::default();
        assert_eq!(
            r.marked(&MarkedPicture { long_refs: 1, ..pic(1, true) }),
            0,
            "a long-term reference rules the heuristic out"
        );
        assert_eq!(
            r.marked(&MarkedPicture { pps_ref_count: [5, 1], ..pic(1, true) }),
            0,
            "a PPS allowing many references rules it out"
        );
    }

    #[test]
    fn reorder_estimate_follows_b_pictures_and_poc_jumps() {
        let mut r = Recovery::default();
        r.frame_output_order(0, false, None);
        r.frame_output_order(2, false, None);
        assert_eq!(r.has_b_frames, 0);
        r.frame_output_order(8, false, None);
        assert_eq!(r.has_b_frames, 1, "a POC step above 2 implies reordering");
        let mut r = Recovery::default();
        r.frame_output_order(0, false, Some(2));
        assert_eq!(r.has_b_frames, 2);
        r.frame_output_order(2, true, Some(2));
        assert_eq!(r.has_b_frames, 2, "a bitstream restriction caps the estimate");
    }

    #[test]
    fn flush_forgets_recovery() {
        let mut r = Recovery::default();
        r.picture_start(0, 4, true, true, true);
        r.flush_change();
        assert_eq!(r.picture_start(1, 4, false, false, true), 0);
    }
}
