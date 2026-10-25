use oxideav_core::{CodecParameters, Decoder, Error, Frame, Packet, TimeBase};

fn drain(decoder: &mut dyn Decoder, frames: &mut Vec<Vec<u8>>) {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(frame)) => frames.push(frame.planes.into_iter().flat_map(|plane| plane.data).collect()),
            Ok(_) => panic!("unexpected non-video frame"),
            Err(Error::NeedMore | Error::Eof) => break,
            Err(error) => panic!("receive: {error}"),
        }
    }
}

#[test]
fn reset_keeps_the_avcc_parameter_sets() {
    let bytes = include_bytes!("fixtures/x264_testsrc_128x96_25f.h264");
    let mut starts: Vec<_> = bytes.windows(3).enumerate().filter_map(|(i, b)| (b == [0, 0, 1]).then_some(i + 3)).collect();
    starts.push(bytes.len() + 3);
    let nals: Vec<_> = starts.windows(2).map(|pair| {
        let mut end = pair[1] - 3;
        while end > pair[0] && bytes[end - 1] == 0 { end -= 1; }
        &bytes[pair[0]..end]
    }).collect();
    let sps = nals.iter().find(|n| n[0] & 31 == 7).unwrap();
    let pps = nals.iter().find(|n| n[0] & 31 == 8).unwrap();
    let mut avcc = vec![1, sps[1], sps[2], sps[3], 0xff, 0xe1];
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps);
    avcc.push(1);
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps);
    let packets: Vec<_> = nals.into_iter().filter(|n| !matches!(n[0] & 31, 7 | 8)).map(|nal| {
        let mut packet = (nal.len() as u32).to_be_bytes().to_vec();
        packet.extend_from_slice(nal);
        packet
    }).collect();
    let mut params = CodecParameters::video("h264".into());
    params.extradata = avcc;
    let mut decoder = oxideav_h264::h264_decoder::make_decoder(&params).unwrap();
    let mut expected = Vec::new();
    for packet in &packets {
        decoder.send_packet(&Packet::new(0, TimeBase::new(1, 25), packet.clone())).unwrap();
        drain(&mut *decoder, &mut expected);
    }
    decoder.flush().unwrap();
    drain(&mut *decoder, &mut expected);
    assert_eq!(expected.len(), 25);
    decoder.reset().unwrap();
    for packet in &packets[..3] {
        decoder.send_packet(&Packet::new(0, TimeBase::new(1, 25), packet.clone())).unwrap();
        drain(&mut *decoder, &mut Vec::new());
    }
    decoder.reset().unwrap();
    let mut actual = Vec::new();
    for packet in &packets {
        decoder.send_packet(&Packet::new(0, TimeBase::new(1, 25), packet.clone())).unwrap();
        drain(&mut *decoder, &mut actual);
    }
    decoder.flush().unwrap();
    drain(&mut *decoder, &mut actual);
    assert_eq!(actual, expected, "every complete frame after repeated seeks");
}
