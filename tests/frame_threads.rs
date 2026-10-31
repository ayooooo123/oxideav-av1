#![cfg(not(target_arch = "wasm32"))]

use oxideav_av1::decoder::frame_driver::frame_threads::{FrameThreadDecoder, MAX_LOOKAHEAD_BYTES, MAX_LOOKAHEAD_FRAMES};
use oxideav_av1::decoder::SpecDecodeSession;
use oxideav_av1::encoder::ivf::IvfReader;
use oxideav_av1::registry::{make_decoder, CODEC_ID_STR};
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, ExecutionContext, Frame, Packet, PixelFormat, TimeBase};

#[derive(Debug, PartialEq, Eq)]
struct Picture {
    pts: Option<i64>,
    dimensions: Option<(u32, u32)>,
    format: Option<PixelFormat>,
    planes: Vec<(usize, Vec<u8>)>,
}

fn drain(decoder: &mut dyn Decoder, out: &mut Vec<Picture>) -> Result<(), Error> {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(frame)) => out.push(Picture {
                pts: frame.pts,
                dimensions: decoder.output_video_dimensions(),
                format: decoder.output_pixel_format(),
                planes: frame.planes.into_iter().map(|p| (p.stride, p.data)).collect(),
            }),
            Ok(_) => panic!("AV1 returned a non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return Ok(()),
            Err(error) => return Err(error),
        }
    }
}

fn packets(ivf: &[u8], split: bool) -> Vec<Packet> {
    if !split {
        let mut packet = Packet::new(0, TimeBase::new(1, 1000), ivf.to_vec());
        packet.pts = Some(17);
        return vec![packet];
    }
    IvfReader::new(ivf).unwrap().read_all().unwrap().into_iter().enumerate().map(|(i, record)| {
        let mut packet = Packet::new(0, TimeBase::new(1, 1000), record.payload);
        // Distinct packet timestamps also cover hidden + shown frames sharing
        // a temporal unit and the show-existing output's packet ownership.
        packet.pts = Some(i as i64 * 37 - 113);
        packet
    }).collect()
}

fn decode(ivf: &[u8], context: &ExecutionContext, split: bool) -> Result<Vec<Picture>, Error> {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut decoder = make_decoder(&params)?;
    decoder.set_execution_context(context);
    let mut out = Vec::new();
    for packet in packets(ivf, split) {
        decoder.send_packet(&packet)?;
        drain(decoder.as_mut(), &mut out)?;
    }
    decoder.flush()?;
    drain(decoder.as_mut(), &mut out)?;
    assert!(matches!(decoder.receive_frame(), Err(Error::Eof)));
    Ok(out)
}

fn corpus() -> Vec<(String, Vec<u8>)> {
    // Reuse the existing corpus's hex declarations instead of copying its
    // encoded streams into a second fixture collection. Only *_IVF concat
    // literals are selected; expected-pixel/hash literals are never inputs.
    let mut inputs = Vec::new();
    for declaration in include_str!("fixture_conformance.rs").split("const ").skip(1) {
        let Some((name, value)) = declaration.split_once(": &str = concat!(") else { continue; };
        if !name.ends_with("_IVF") { continue; }
        let body = value.split_once(");").expect("complete corpus concat literal").0;
        let hex: String = body.split('"').skip(1).step_by(2).collect();
        assert_eq!(hex.len() % 2, 0, "{name}");
        let bytes: Vec<_> = hex.as_bytes().chunks_exact(2).map(|pair| {
            u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()
        }).collect();
        assert!(bytes.starts_with(b"DKIF"), "{name}");
        inputs.push((name.into(), bytes));
    }
    for (name, bytes) in [
        ("odd_420", include_bytes!("decoded_format/a_420_33x17.ivf").as_slice()),
        ("odd_420_10bit", include_bytes!("decoded_format/b_420p10_47x31.ivf").as_slice()),
        ("odd_gray", include_bytes!("decoded_format/c_gray_34x18.ivf").as_slice()),
        ("odd_444", include_bytes!("decoded_format/d_444_35x19.ivf").as_slice()),
    ] { inputs.push((name.into(), bytes.to_vec())); }
    assert!(inputs.len() > 50, "reuse the full conformance corpus, not a token sample");
    inputs
}

#[test]
fn deterministic_frame_thread_stress_matches_every_serial_picture() {
    let budgets = [ExecutionContext::with_threads(2), ExecutionContext::with_threads(4),
        ExecutionContext::with_threads(8), ExecutionContext::auto()];
    let inputs = corpus();
    let mut comparisons = 0;
    for (name, ivf) in &inputs {
        let serial_whole = decode(ivf, &ExecutionContext::serial(), false).unwrap_or_else(|e| panic!("{name}: serial whole {e}"));
        let serial_split = decode(ivf, &ExecutionContext::serial(), true).unwrap_or_else(|e| panic!("{name}: serial split {e}"));
        for repeat in 0..8 {
            let split = repeat % 2 != 0;
            let expected = if split { &serial_split } else { &serial_whole };
            for budget in &budgets {
                let actual = decode(ivf, budget, split).unwrap_or_else(|e| panic!("{name}, repeat {repeat}, budget {}: {e}", budget.threads));
                assert_eq!(&actual, expected, "{name}, repeat {repeat}, budget {}, split {split}", budget.threads);
                comparisons += actual.len();
            }
        }
    }
    eprintln!("frame-thread stress: {} streams, 8 repetitions, budgets 2/4/8/auto, {comparisons} byte-exact picture comparisons", inputs.len());
}

#[test]
fn lookahead_and_worker_retention_are_bounded() {
    let inputs = corpus();
    let (_, ivf) = inputs.iter().find(|(name, _)| name == "SELF_PYR_64X64_Q60_LEN17_DEEP_IVF").expect("deep-pyramid corpus input");
    for threads in [2, 4, 8, ExecutionContext::auto().threads] {
        let mut decoder = FrameThreadDecoder::new(&SpecDecodeSession::new(), &ExecutionContext::with_threads(threads));
        decoder.send_packet(ivf, Some(41)).unwrap();
        while decoder.receive_frame().unwrap().is_some() {}
        decoder.flush().unwrap();
        while decoder.receive_frame().unwrap().is_some() {}
        let stats = decoder.stats();
        assert!(stats.peak_active_frames <= threads.min(4));
        assert!(stats.peak_queued_frames <= MAX_LOOKAHEAD_FRAMES);
        assert!(stats.peak_queued_bytes <= MAX_LOOKAHEAD_BYTES);
        assert!(stats.peak_retained_frames <= 8 * (MAX_LOOKAHEAD_FRAMES + 4) + MAX_LOOKAHEAD_FRAMES + 4);
    }
}

#[test]
fn changing_budgets_preserves_pending_output_and_reference_state() {
    let (_, ivf) = corpus().into_iter().find(|(name, _)| name == "SELF_PYR_64X64_Q60_LEN17_DEEP_IVF").unwrap();
    let expected = decode(&ivf, &ExecutionContext::serial(), true).unwrap();
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut decoder = make_decoder(&params).unwrap();
    decoder.set_execution_context(&ExecutionContext::with_threads(8));
    let mut out = Vec::new();
    for (i, packet) in packets(&ivf, true).into_iter().enumerate() {
        decoder.send_packet(&packet).unwrap();
        if i % 4 == 1 { decoder.set_execution_context(&ExecutionContext::serial()); }
        if i % 4 == 3 { decoder.set_execution_context(&ExecutionContext::with_threads(4)); }
        drain(decoder.as_mut(), &mut out).unwrap();
    }
    decoder.flush().unwrap();
    drain(decoder.as_mut(), &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn reset_discards_inflight_output_and_keeps_sequence_configuration() {
    let (_, ivf) = corpus().into_iter().find(|(name, _)| name == "SELF_PYR_64X64_Q60_LEN17_DEEP_IVF").unwrap();
    let expected = decode(&ivf, &ExecutionContext::serial(), true).unwrap();
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut decoder = make_decoder(&params).unwrap();
    decoder.set_execution_context(&ExecutionContext::with_threads(8));
    for packet in packets(&ivf, true).into_iter().take(4) { decoder.send_packet(&packet).unwrap(); }
    decoder.reset().unwrap();
    let mut out = Vec::new();
    for packet in packets(&ivf, true) {
        decoder.send_packet(&packet).unwrap();
        drain(decoder.as_mut(), &mut out).unwrap();
    }
    decoder.flush().unwrap();
    drain(decoder.as_mut(), &mut out).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn malformed_input_returns_an_error_without_stranding_dependents() {
    let (_, ivf) = corpus().into_iter().find(|(name, _)| name == "SELF_PYR_64X64_Q60_LEN17_DEEP_IVF").unwrap();
    let mut units = packets(&ivf, true);
    // Keep the header and most tile bytes, but truncate an arithmetic payload.
    let cut = units[0].data.len().saturating_sub(3);
    units[0].data.truncate(cut);
    for threads in [2, 4, 8] {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut decoder = make_decoder(&params).unwrap();
        decoder.set_execution_context(&ExecutionContext::with_threads(threads));
        let mut failed = false;
        let mut out = Vec::new();
        for packet in &units {
            if decoder.send_packet(packet).is_err() || drain(decoder.as_mut(), &mut out).is_err() {
                failed = true;
                break;
            }
        }
        if !failed { failed = decoder.flush().is_err() || drain(decoder.as_mut(), &mut out).is_err(); }
        assert!(failed, "truncated coded frame must fail at budget {threads}");
    }
}

#[test]
fn show_existing_key_refresh_preserves_snapshot_and_packet_timestamps() {
    let (_, ivf) = corpus().into_iter().find(|(name, _)| name == "TINY_I_ONLY_16X16_IVF").unwrap();
    let mut input = packets(&ivf, true);
    for pts in [1001, 1002] {
        // OBU_FRAME_HEADER, one payload byte: show_existing_frame=1,
        // frame_to_show_map_idx=0, then trailing_bits(). A shown KEY
        // reloads its saved state and aliases that same version to all slots.
        let mut packet = Packet::new(0, TimeBase::new(1, 1000), vec![0x1a, 0x01, 0x88]);
        packet.pts = Some(pts);
        input.push(packet);
    }
    let mut expected = None;
    for context in [ExecutionContext::serial(), ExecutionContext::with_threads(2),
        ExecutionContext::with_threads(4), ExecutionContext::with_threads(8), ExecutionContext::auto()] {
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let mut decoder = make_decoder(&params).unwrap();
        decoder.set_execution_context(&context);
        let mut output = Vec::new();
        for packet in &input {
            decoder.send_packet(packet).unwrap();
            assert_eq!(decoder.output_video_dimensions(), Some((16, 16)));
            drain(decoder.as_mut(), &mut output).unwrap();
        }
        decoder.flush().unwrap();
        drain(decoder.as_mut(), &mut output).unwrap();
        assert_eq!(output.len(), 3);
        assert_eq!(output.iter().map(|frame| frame.pts).collect::<Vec<_>>(),
            vec![input[0].pts, Some(1001), Some(1002)]);
        assert_eq!(output[0].planes, output[1].planes);
        assert_eq!(output[1].planes, output[2].planes);
        if let Some(expected) = &expected {
            assert_eq!(&output, expected);
        } else {
            expected = Some(output);
        }
    }
}
