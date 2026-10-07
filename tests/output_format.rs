//! `Decoder::output_video_dimensions` and `Decoder::output_pixel_format`
//! describe the frame `receive_frame` returned last: its visible size
//! after the SPS frame cropping, and its plane layout, across an SPS that
//! changes both mid-stream. Before the first frame they report the SPS
//! of the picture being decoded.

use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase};
use oxideav_h264::nal::AnnexBSplitter;

/// x264 (through FFmpeg): three 128x96 4:2:0 frames, then a second SPS
/// and three 100x60 4:2:2 frames coded as 112x64 with frame cropping
/// (right 12, bottom 4).
const STREAM: &[u8] = include_bytes!("fixtures/size_change_crop.h264");

/// FNV-1a 64 of each frame of FFmpeg 2da55bf's rawvideo decode of the
/// two halves (`ffmpeg -i <half>.h264 -f rawvideo -pix_fmt yuv420p|yuv422p`).
const FFMPEG_FRAMES: [u64; 6] = [
    0xdb4f0eaadc2c6cb0,
    0xdd9c8b3b93d2901d,
    0x78a9594960b41968,
    0xc67e17057e6ac7ee,
    0xc5e535a06b874a9d,
    0xe3a4dc606b60dd6c,
];

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    bytes.fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn decoder() -> Box<dyn Decoder> {
    oxideav_h264::h264_decoder::make_decoder(&CodecParameters::video(CodecId::new("h264"))).unwrap()
}

fn packet(data: Vec<u8>) -> Packet {
    Packet::new(0, TimeBase::new(1, 25), data)
}

#[test]
fn every_frame_reports_its_own_size_and_layout() {
    let mut dec = decoder();
    // One packet: by the time the first frames come out the decoder has
    // already parsed the second SPS, which must not leak into them.
    dec.send_packet(&packet(STREAM.to_vec())).unwrap();
    dec.flush().unwrap();
    let mut frames = Vec::new();
    loop {
        let frame = match dec.receive_frame() {
            Ok(Frame::Video(frame)) => frame,
            Ok(other) => panic!("expected video, got {other:?}"),
            Err(Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        };
        let (width, height) = dec.output_video_dimensions().expect("a returned frame has a size");
        let format = dec.output_pixel_format().expect("8-bit 4:2:0 and 4:2:2 have formats");
        let chroma_rows = match format {
            PixelFormat::Yuv420P => height / 2,
            PixelFormat::Yuv422P => height,
            other => panic!("unexpected {other:?}"),
        };
        let planes = frame.image_planes();
        assert_eq!(planes.len(), 3);
        // The planes hold exactly the visible picture.
        assert_eq!((planes[0].stride, planes[0].data.len()), (width as usize, (width * height) as usize));
        for chroma in &planes[1..] {
            assert_eq!((chroma.stride, chroma.data.len()), ((width / 2) as usize, (width / 2 * chroma_rows) as usize));
        }
        let hash = fnv1a(planes.iter().flat_map(|p| p.data.iter().copied()));
        frames.push((width, height, format, hash));
    }
    let want: Vec<_> = FFMPEG_FRAMES
        .iter()
        .enumerate()
        .map(|(i, &hash)| match i {
            0..=2 => (128, 96, PixelFormat::Yuv420P, hash),
            _ => (100, 60, PixelFormat::Yuv422P, hash),
        })
        .collect();
    assert_eq!(frames, want);
}

#[test]
fn before_the_first_frame_the_sps_is_reported() {
    let mut dec = decoder();
    assert_eq!(dec.output_video_dimensions(), None);
    assert_eq!(dec.output_pixel_format(), None);

    // The first access unit only: SPS, PPS, SEI and the IDR picture.
    let nals: Vec<&[u8]> = AnnexBSplitter::new(STREAM).collect();
    let second_picture = nals
        .iter()
        .enumerate()
        .filter(|(_, nal)| matches!(nal[0] & 0x1f, 1 | 5))
        .nth(1)
        .map(|(i, _)| i)
        .unwrap();
    let first_unit: Vec<u8> = nals[..second_picture].iter().flat_map(|nal| [&[0, 0, 0, 1][..], nal].concat()).collect();
    dec.send_packet(&packet(first_unit)).unwrap();
    assert!(matches!(dec.receive_frame(), Err(Error::NeedMore)));
    assert_eq!(dec.output_video_dimensions(), Some((128, 96)));
    assert_eq!(dec.output_pixel_format(), Some(PixelFormat::Yuv420P));

    // After a seek nothing has been returned yet.
    dec.reset().unwrap();
    assert_eq!(dec.output_video_dimensions(), None);
}
