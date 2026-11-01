//! Feed complete decoded plane bytes to the Node hashing host, before any display conversion.
use oxideav_av1::encoder::ivf::IvfReader;
use oxideav_av1::registry::{make_decoder, CODEC_ID_STR};
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "picture_hash")]
extern "C" {
    fn plane(ptr: *const u8, len: usize);
    fn end_picture();
}

fn drain(decoder: &mut dyn Decoder) {
    loop {
        match decoder.receive_frame() {
            Ok(Frame::Video(frame)) => {
                #[cfg(target_arch = "wasm32")]
                // SAFETY: the host consumes each live plane synchronously, without retaining
                // its pointer; every supplied range belongs to the decoded plane allocation.
                unsafe {
                    for p in &frame.planes {
                        plane(p.data.as_ptr(), p.data.len());
                    }
                    end_picture();
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use std::io::Write;
                    let mut out = std::io::stdout().lock();
                    let len: usize = frame.planes.iter().map(|p| p.data.len()).sum();
                    out.write_all(&(len as u64).to_le_bytes()).unwrap();
                    for p in &frame.planes {
                        out.write_all(&p.data).unwrap();
                    }
                }
            }
            Err(Error::NeedMore | Error::Eof) => break,
            Err(e) => panic!("decode failed: {e}"),
            _ => panic!("non-video output"),
        }
    }
}

fn main() {
    let file = std::env::args().nth(1).expect("IVF input path");
    let input = std::fs::read(file).unwrap();
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut decoder = make_decoder(&params).unwrap();
    if let Some(threads) = std::env::args().nth(2) {
        let threads = threads.parse::<usize>().expect("positive decoder thread count");
        assert!(threads > 0, "decoder thread count must be positive");
        decoder.set_execution_context(&oxideav_core::ExecutionContext::with_threads(threads));
    }
    for record in IvfReader::new(&input).unwrap().read_all().unwrap() {
        decoder
            .send_packet(&Packet::new(0, TimeBase::new(1, 30), record.payload))
            .unwrap();
        drain(decoder.as_mut());
    }
    decoder.flush().unwrap();
    drain(decoder.as_mut());
}
