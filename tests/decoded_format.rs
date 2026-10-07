//! The registry decoder reports the size and pixel layout of the frame it
//! last returned (oxideav-core `Decoder::output_video_dimensions` /
//! `output_pixel_format`): odd sizes, 10-bit, monochrome and 4:4:4. A new
//! size or layout is reported with the first frame that has it, not when
//! its sequence header arrives.
//!
//! `decoded_format/*.ivf` are two-frame libaom encodes of FFmpeg
//! `testsrc`, decoded one after the other by one decoder.

use oxideav_core::{
    CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};

/// A frame's width, height and pixel format.
type Layout = (u32, u32, PixelFormat);

/// The temporal units of every stream, in order, and the layout of each
/// frame they decode to (one shown frame per unit).
fn units() -> Vec<(Vec<u8>, Layout)> {
    use PixelFormat::*;
    let streams: [(&[u8], Layout); 4] = [
        (
            include_bytes!("decoded_format/a_420_33x17.ivf"),
            (33, 17, Yuv420P),
        ),
        (
            include_bytes!("decoded_format/b_420p10_47x31.ivf"),
            (47, 31, Yuv420P10Le),
        ),
        (
            include_bytes!("decoded_format/c_gray_34x18.ivf"),
            (34, 18, Gray8),
        ),
        (
            include_bytes!("decoded_format/d_444_35x19.ivf"),
            (35, 19, Yuv444P),
        ),
    ];
    let mut units = Vec::new();
    for (file, layout) in streams {
        // IVF: a file header of the length at bytes 6..8, then a 12-byte
        // header (size, pts) before each temporal unit.
        assert_eq!(&file[..4], b"DKIF");
        let mut at = usize::from(u16::from_le_bytes([file[6], file[7]]));
        let mut count = 0;
        while at + 12 <= file.len() {
            let size = u32::from_le_bytes(file[at..at + 4].try_into().unwrap()) as usize;
            units.push((file[at + 12..at + 12 + size].to_vec(), layout));
            at += 12 + size;
            count += 1;
        }
        assert_eq!(count, 2, "{layout:?}: temporal units");
    }
    units
}

/// Asserts that `frame` holds a `w`×`h` picture of `format`: the plane
/// count, and each plane's row bytes and rows (this decoder packs the
/// frame without padding).
fn assert_geometry(frame: &VideoFrame, (w, h, format): Layout, at: usize) {
    use PixelFormat::*;
    let (bytes, sub_x, sub_y, count) = match format {
        Yuv420P => (1, 2, 2, 3),
        Yuv420P10Le => (2, 2, 2, 3),
        Yuv444P => (1, 1, 1, 3),
        Gray8 => (1, 1, 1, 1),
        other => panic!("frame {at}: no plane layout for {other:?} in this test"),
    };
    let planes = frame.image_planes();
    assert_eq!(planes.len(), count, "frame {at}: plane count");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = if i == 0 {
            (w, h)
        } else {
            (w.div_ceil(sub_x), h.div_ceil(sub_y))
        };
        assert_eq!(
            plane.stride,
            pw as usize * bytes,
            "frame {at}: plane {i} stride"
        );
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "frame {at}: plane {i} rows"
        );
    }
}

fn decoder() -> Box<dyn Decoder> {
    oxideav_av1::registry::make_decoder(&CodecParameters::video("av1".into())).expect("decoder")
}

fn report(dec: &dyn Decoder) -> Option<Layout> {
    let (w, h) = dec.output_video_dimensions()?;
    Some((w, h, dec.output_pixel_format()?))
}

fn packet(unit: Vec<u8>) -> Packet {
    Packet::new(0, TimeBase::new(1, 25), unit)
}

/// Receives every available frame, checking each against the next
/// expected layout right after it is returned.
fn receive_all(dec: &mut dyn Decoder, expected: &[Layout], seen: &mut usize) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let want = expected[*seen];
                assert_eq!(report(dec), Some(want), "report after frame {seen}");
                assert_geometry(&frame, want, *seen);
                *seen += 1;
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// Every stream sent before any frame is received: all four sequence
/// headers are parsed while the first stream's frames wait in the queue,
/// so the reports must follow the frames.
#[test]
fn each_frame_reports_its_own_size_and_layout() {
    let (units, expected): (Vec<_>, Vec<_>) = units().into_iter().unzip();
    let mut dec = decoder();
    for unit in units {
        dec.send_packet(&packet(unit)).expect("send");
    }
    // Before any frame is returned, the next frame in output order (the
    // first stream's), not the newest sequence header.
    assert_eq!(
        report(&*dec),
        Some(expected[0]),
        "report before the first frame"
    );
    let mut seen = 0;
    receive_all(&mut *dec, &expected, &mut seen);
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
    // After the end, the last frame returned.
    assert_eq!(report(&*dec), expected.last().copied());
}

/// One temporal unit per packet, receiving after each, as a demuxer
/// feeds it.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let (units, expected): (Vec<_>, Vec<_>) = units().into_iter().unzip();
    let mut dec = decoder();
    let mut seen = 0;
    for unit in units {
        dec.send_packet(&packet(unit)).expect("send");
        receive_all(&mut *dec, &expected, &mut seen);
    }
    dec.flush().expect("flush");
    receive_all(&mut *dec, &expected, &mut seen);
    assert_eq!(seen, expected.len(), "frames");
}
