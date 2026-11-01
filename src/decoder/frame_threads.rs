//! Whole-frame lookahead over immutable reference-slot versions.
//!
//! This module is a child of `frame_driver`. Only the coordinator changes the
//! logical slot map; workers publish into their own once-only result cells.
//! Parsed work and ordered results are bounded. Input packets are owned lazily:
//! a whole IVF packet is retained once, not expanded into every frame at send.

use super::*;
use std::collections::{HashSet, VecDeque};
use std::mem::size_of;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex, OnceLock};
use std::thread::JoinHandle;

/// Maximum parsed jobs waiting for a worker or an earlier output.
pub const MAX_LOOKAHEAD_FRAMES: usize = 16;
/// Compressed parsed-work budget; one oversized frame is admitted on its own.
pub const MAX_LOOKAHEAD_BYTES: usize = 32 * 1024 * 1024;
const MAX_ACTIVE_FRAMES: usize = 4;
type Layout = (u32, u32, u8, bool, bool, bool);
type Version = Arc<ReferenceVersion>;
type Slots = [Option<Version>; NUM_REF_FRAMES as usize];

/// Peaks since creation/reset. Byte counters account for owned buffer capacities
/// and fixed payloads, not allocator bookkeeping or transient walker scratch.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrameThreadStats {
    pub peak_retained_frames: usize,
    pub peak_retained_bytes: usize,
    pub peak_active_frames: usize,
    pub peak_queued_frames: usize,
    pub peak_queued_bytes: usize,
}

#[derive(Debug)]
struct ReferenceVersion {
    layout: Layout,
    result: OnceLock<Result<Arc<CompletedFrame>, Error>>,
}

#[derive(Debug)]
struct CompletedFrame {
    slot: Option<Arc<SpecRefSlot>>,
    // Taking an ordered output releases its independent display copy even when
    // the version's grain-free reference payload remains in the slot map.
    output: Mutex<Option<Arc<SpecFrame>>>,
}

#[derive(Debug)]
struct Job {
    id: usize,
    seq: Arc<SequenceHeader>,
    decode: FrameDecode,
    fh: FrameHeader,
    info: RefInfo,
    slots: Slots,
    needed: [bool; NUM_REF_FRAMES as usize],
    tiles: Vec<TilePayload>,
    version: Version,
    replay: bool,
}

#[derive(Debug)]
struct Running {
    id: usize,
    version: Version,
    slots: Slots,
    seq: Arc<SequenceHeader>,
    bytes: usize,
    handle: Option<JoinHandle<()>>,
}

#[derive(Debug)]
struct Ordered {
    version: Version,
    shown: bool,
    pts: Option<i64>,
}

#[derive(Debug)]
struct PendingHeader {
    fh: FrameHeader,
    original: Vec<u8>,
    tiles: Vec<TilePayload>,
    next_tile: u32,
}

#[derive(Debug, Clone, Copy)]
enum ParseStep { Progress, Done, Blocked }

#[derive(Debug)]
struct Input {
    bytes: Vec<u8>,
    cursor: usize,
    tu_end: Option<usize>,
    ivf: bool,
    pts: Option<i64>,
    pending: Option<PendingHeader>,
}

/// Hidden registry-facing whole-frame decoder. Callers should receive between
/// sends: accepting unlimited sends without receiving necessarily retains their
/// compressed input. A single whole-IVF input can also exceed the parsed-work
/// byte budget; both are included honestly in `peak_queued_bytes`.
#[derive(Debug)]
pub struct FrameThreadDecoder {
    seq: Option<Arc<SequenceHeader>>,
    decode: FrameDecode,
    info: RefInfo,
    slots: Slots,
    chosen_op: u8,
    op_idc: u16,
    max_picture_size: u32,
    exec: oxideav_core::ExecutionContext,
    inputs: VecDeque<Input>,
    pending: VecDeque<Job>,
    ordered: VecDeque<Ordered>,
    running: Vec<Running>,
    sender: mpsc::Sender<usize>,
    receiver: mpsc::Receiver<usize>,
    budget: Arc<super::worker_budget::Budget>,
    next_id: usize,
    draining: bool,
    input_blocked: bool,
    failure: Option<Error>,
    peaks: FrameThreadStats,
}

impl FrameThreadDecoder {
    pub fn new(session: &SpecDecodeSession, exec: &oxideav_core::ExecutionContext) -> Self {
        let (sender, receiver) = mpsc::channel();
        let budget = super::worker_budget::Budget::new(exec.effective_workers(usize::MAX), sender.clone());
        let mut seeded: Vec<(Arc<SpecRefSlot>, Version)> = Vec::new();
        let cc = session.seq.as_ref().map(|s| &s.color_config);
        let slots = core::array::from_fn(|i| {
            let slot = session.refs.slots[i].as_ref()?;
            if let Some((_, version)) = seeded.iter().find(|(old, _)| Arc::ptr_eq(old, slot)) {
                return Some(version.clone());
            }
            let &(w, h) = slot.plane_dims.first()?;
            let layout = (w, h, slot.bit_depth, slot.planes.len() == 1,
                cc.is_some_and(|c| c.subsampling_x), cc.is_some_and(|c| c.subsampling_y));
            let version = Arc::new(ReferenceVersion { layout, result: OnceLock::new() });
            let _ = version.result.set(Ok(Arc::new(CompletedFrame {
                slot: Some(slot.clone()), output: Mutex::new(None),
            })));
            seeded.push((slot.clone(), version.clone()));
            Some(version)
        });
        let mut decoder = Self {
            seq: session.seq.clone().map(Arc::new), info: session.refs.info.clone(), slots,
            decode: session.refs.decode,
            chosen_op: session.chosen_op, op_idc: session.op_idc,
            max_picture_size: session.max_picture_size, exec: exec.clone(),
            inputs: VecDeque::new(), pending: VecDeque::new(), ordered: VecDeque::new(),
            running: Vec::new(), sender, receiver, next_id: 0, draining: false,
            input_blocked: false,
            budget,
            failure: None, peaks: FrameThreadStats::default(),
        };
        decoder.record_peaks();
        decoder
    }

    pub fn send_packet(&mut self, data: &[u8], pts: Option<i64>) -> Result<(), Error> {
        if let Some(error) = self.failure { return Err(error); }
        let ivf = data.starts_with(b"DKIF");
        if ivf { IvfReader::new(data).map_err(|_| Error::UnexpectedEnd)?; }
        self.draining = false;
        self.inputs.push_back(Input {
            bytes: data.to_vec(), cursor: if ivf { 32 } else { 0 },
            tu_end: if ivf { None } else { Some(data.len()) }, ivf, pts, pending: None,
        });
        self.pump()
    }

    /// `None` means need-more input, or true exhaustion after `flush`. Once
    /// flushed, completion notifications are awaited until an output or EOF.
    pub fn receive_frame(&mut self) -> Result<Option<(Arc<SpecFrame>, Option<i64>)>, Error> {
        loop {
            self.pump()?;
            let mut retired = false;
            while let Some(front) = self.ordered.front() {
                let Some(result) = front.version.result.get() else { break; };
                let completed = result.as_ref().map_err(|&e| e)?;
                let output = if front.shown {
                    Some(completed.output.lock().map_err(|_| Error::PartitionWalkOutOfRange)?
                        .take().ok_or(Error::PartitionWalkOutOfRange)?)
                } else { None };
                let pts = front.pts;
                self.ordered.pop_front();
                retired = true;
                if let Some(output) = output { return Ok(Some((output, pts))); }
            }
            if retired { continue; }
            let outstanding = !self.inputs.is_empty() || !self.pending.is_empty()
                || !self.running.is_empty() || !self.ordered.is_empty();
            if !outstanding { return Ok(None); }
            let full = self.ordered.len() >= MAX_LOOKAHEAD_FRAMES + self.active_limit()
                || self.pending.len() >= MAX_LOOKAHEAD_FRAMES
                || self.work_bytes() >= MAX_LOOKAHEAD_BYTES || self.input_blocked;
            if !self.draining && !full { return Ok(None); }
            if self.running.is_empty() {
                return self.fail(Error::PartitionWalkOutOfRange);
            }
            if let Err(error) = self.collect(true) { return self.fail(error); }
        }
    }

    pub fn next_frame_layout(&self) -> Option<Layout> {
        self.ordered.iter().find(|entry| entry.shown).map(|entry| entry.version.layout)
    }

    pub fn flush(&mut self) -> Result<(), Error> {
        self.draining = true;
        self.pump()
    }

    pub fn reset(&mut self) {
        self.join_all();
        self.inputs.clear(); self.pending.clear(); self.ordered.clear();
        self.slots = Default::default(); self.info = RefInfo::default();
        while self.receiver.try_recv().is_ok() {}
        self.failure = None; self.draining = false;
        self.peaks = FrameThreadStats::default();
    }

    pub fn stats(&self) -> FrameThreadStats { self.peaks }

    pub fn set_execution_context(&mut self, ctx: &oxideav_core::ExecutionContext) {
        // Existing jobs retain their grant until joined; a smaller new budget
        // must not overlap them with newly granted workers.
        self.join_all();
        self.exec = ctx.clone();
        self.budget = super::worker_budget::Budget::new(ctx.effective_workers(usize::MAX), self.sender.clone());
    }

    pub fn finish_into_session(mut self) -> Result<(SpecDecodeSession, Vec<(Arc<SpecFrame>, Option<i64>)>), Error> {
        self.flush()?;
        let mut output = Vec::new();
        while let Some(frame) = self.receive_frame()? { output.push(frame); }
        self.join_all();
        let slots = core::array::from_fn(|i| self.slots[i].as_ref().and_then(|version| {
            version.result.get().and_then(|result| result.as_ref().ok())
                .and_then(|completed| completed.slot.clone())
        }));
        let session = SpecDecodeSession {
            seq: self.seq.as_ref().map(|seq| (**seq).clone()),
            refs: SpecRefState { info: self.info.clone(), decode: self.decode, slots },
            chosen_op: self.chosen_op, op_idc: self.op_idc,
            max_picture_size: self.max_picture_size, exec: self.exec.clone(),
        };
        Ok((session, output))
    }

    fn active_limit(&self) -> usize { self.exec.effective_workers(MAX_ACTIVE_FRAMES).max(1) }

    fn fail<T>(&mut self, error: Error) -> Result<T, Error> {
        self.failure = Some(error);
        Err(error)
    }

    fn pump(&mut self) -> Result<(), Error> {
        if let Some(error) = self.failure { return Err(error); }
        let result = self.pump_inner();
        self.record_peaks();
        if let Err(error) = result { self.failure = Some(error); }
        result
    }

    fn pump_inner(&mut self) -> Result<(), Error> {
        self.collect(false)?;
        loop {
            let mut changed = self.launch_ready()?;
            if !self.inputs.is_empty()
                && self.pending.len() < MAX_LOOKAHEAD_FRAMES
                && self.ordered.len() < MAX_LOOKAHEAD_FRAMES + self.active_limit()
                && (self.work_bytes() < MAX_LOOKAHEAD_BYTES || self.ordered.is_empty())
            {
                changed |= self.parse_one()?;
            }
            if !changed { break; }
        }
        Ok(())
    }

    fn collect(&mut self, block: bool) -> Result<(), Error> {
        let first = if block {
            Some(self.receiver.recv().map_err(|_| Error::PartitionWalkOutOfRange)?)
        } else { None };
        let mut message = first;
        loop {
            let id = match message.take().or_else(|| self.receiver.try_recv().ok()) {
                Some(id) => id, None => break,
            };
            if let Some(index) = self.running.iter().position(|running| running.id == id) {
                let mut running = self.running.swap_remove(index);
                if running.handle.take().is_some_and(|handle| handle.join().is_err()) {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                match running.version.result.get() {
                    Some(Ok(_)) => {}, Some(Err(error)) => return Err(*error),
                    None => return Err(Error::PartitionWalkOutOfRange),
                }
            }
        }
        Ok(())
    }

    fn launch_ready(&mut self) -> Result<bool, Error> {
        let mut launched = false;
        let mut index = 0;
        while self.running.len() < self.active_limit() && index < self.pending.len() {
            let job = &self.pending[index];
            let mut ready = true;
            for (slot, &needed) in job.needed.iter().enumerate() {
                if !needed { continue; }
                let version = job.slots[slot].as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                match version.result.get() {
                    Some(Ok(completed)) if completed.slot.is_some() => {},
                    Some(Err(error)) => return Err(*error),
                    Some(Ok(_)) => return Err(Error::PartitionWalkOutOfRange),
                    None => ready = false,
                }
            }
            if !ready { index += 1; continue; }
            let Some(frame_lease) = self.budget.try_frame() else { break; };
            let job = self.pending.remove(index).expect("indexed queued job");
            let id = job.id;
            let version = job.version.clone();
            let slots = job.slots.clone();
            let seq = job.seq.clone();
            let bytes = job_bytes(&job);
            let sender = self.sender.clone();
            let exec = self.exec.clone();
            let budget = self.budget.clone();
            let destination = version.clone();
            let run = move || {
                let result = catch_unwind(AssertUnwindSafe(|| {
                    super::worker_budget::within(budget, || run_job(job, &exec))
                }))
                    .unwrap_or(Err(Error::PartitionWalkOutOfRange));
                let _ = destination.result.set(result.map(Arc::new));
                drop(frame_lease);
                let _ = sender.send(id);
            };
            #[cfg(target_arch = "wasm32")]
            let handle = { run(); None };
            #[cfg(not(target_arch = "wasm32"))]
            let handle = Some(std::thread::Builder::new().name(format!("av1-frame-{id}"))
                .spawn(run).map_err(|_| Error::PartitionWalkOutOfRange)?);
            self.running.push(Running { id, version, slots, seq, bytes, handle });
            launched = true;
        }
        Ok(launched)
    }

    fn parse_one(&mut self) -> Result<bool, Error> {
        let mut input = self.inputs.pop_front().expect("nonempty input queue");
        let result = self.parse_input_step(&mut input);
        self.input_blocked = matches!(result, Ok(ParseStep::Blocked));
        if !matches!(result, Ok(ParseStep::Done)) { self.inputs.push_front(input); }
        result.map(|step| !matches!(step, ParseStep::Blocked))
    }

    fn parse_input_step(&mut self, input: &mut Input) -> Result<ParseStep, Error> {
        if input.tu_end == Some(input.cursor) {
            if input.pending.as_ref().is_some_and(|pending| !pending.tiles.is_empty()) {
                return Err(Error::TileGroupInvalid);
            }
            input.pending = None;
            input.tu_end = None;
            if !input.ivf { return Ok(ParseStep::Done); }
        }
        if input.tu_end.is_none() {
            if input.cursor == input.bytes.len() { return Ok(ParseStep::Done); }
            let header_end = input.cursor.checked_add(12).ok_or(Error::UnexpectedEnd)?;
            let header = input.bytes.get(input.cursor..header_end).ok_or(Error::UnexpectedEnd)?;
            let size = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
            let end = header_end.checked_add(size).ok_or(Error::UnexpectedEnd)?;
            if end > input.bytes.len() { return Err(Error::UnexpectedEnd); }
            input.cursor = header_end;
            input.tu_end = Some(end);
            if size == 0 { return Ok(ParseStep::Progress); }
        }
        let end = input.tu_end.ok_or(Error::UnexpectedEnd)?;
        let mut iter = ObuIter::new(&input.bytes[input.cursor..end]);
        let desc = iter.next().ok_or(Error::UnexpectedEnd)??;
        let partial_bytes = input.pending.as_ref().map_or(0, |pending| {
            pending.original.capacity() + tile_bytes(&pending.tiles)
        });
        // Do not copy an oversized OBU into parsed work while earlier work is
        // retained. With no earlier job, a single frame may exceed the budget;
        // its remaining tile groups must still be admitted to make progress.
        if !self.ordered.is_empty()
            && matches!(desc.obu_type, ObuType::Frame | ObuType::FrameHeader | ObuType::TileGroup)
            && self.work_bytes().saturating_add(partial_bytes).saturating_add(desc.payload_len)
                > MAX_LOOKAHEAD_BYTES
        {
            return Ok(ParseStep::Blocked);
        }
        input.cursor += iter.position();
        if self.op_idc != 0 && desc.extension_flag
            && !matches!(desc.obu_type, ObuType::SequenceHeader | ObuType::TemporalDelimiter)
            && (((self.op_idc >> desc.temporal_id) & 1 == 0)
                || ((self.op_idc >> (desc.spatial_id + 8)) & 1 == 0))
        { return Ok(ParseStep::Progress); }
        match desc.obu_type {
            ObuType::TemporalDelimiter | ObuType::Padding | ObuType::Metadata | ObuType::TileList => {},
            ObuType::SequenceHeader => {
                let seq = parse_sequence_header(desc.payload)?;
                let op = seq.operating_points.get(usize::from(self.chosen_op))
                    .ok_or(Error::OperatingPointOutOfRange)?;
                self.op_idc = op.operating_point_idc;
                self.decode = frame_decoder(seq.color_config.bit_depth);
                self.seq = Some(Arc::new(seq));
            }
            ObuType::FrameHeader => {
                if input.pending.as_ref().is_some_and(|pending| !pending.tiles.is_empty()) {
                    return Err(Error::TileGroupInvalid);
                }
                let seq = self.seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = parse_frame_header_with_refs(desc.payload, seq, &self.info)?;
                check_picture_size(&fh, self.max_picture_size)?;
                if fh.show_existing_frame {
                    input.pending = None;
                    self.submit(fh, Vec::new(), input.pts, true)?;
                } else {
                    input.pending = Some(PendingHeader { fh, original: desc.payload.to_vec(),
                        tiles: Vec::new(), next_tile: 0 });
                }
            }
            ObuType::RedundantFrameHeader => {
                let pending = input.pending.as_ref().ok_or(Error::TileGroupInvalid)?;
                if desc.payload != pending.original.as_slice() { return Err(Error::TileGroupInvalid); }
            }
            ObuType::TileGroup => {
                self.seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let pending = input.pending.as_mut().ok_or(Error::TileGroupInvalid)?;
                let ti = pending.fh.tile_info.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let num_tiles = ti.tile_cols * ti.tile_rows;
                let parsed = parse_tile_group_body_with_ti(ti, desc.payload)?;
                if parsed.tg_start != pending.next_tile { return Err(Error::TileGroupInvalid); }
                pending.next_tile = parsed.tg_end + 1;
                pending.tiles.extend(parsed.tiles);
                if pending.next_tile == num_tiles {
                    let pending = input.pending.take().expect("checked frame header");
                    self.submit(pending.fh, pending.tiles, input.pts, false)?;
                }
            }
            ObuType::Frame => {
                if input.pending.as_ref().is_some_and(|pending| !pending.tiles.is_empty()) {
                    return Err(Error::TileGroupInvalid);
                }
                input.pending = None;
                let seq = self.seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = parse_frame_header_with_refs(desc.payload, seq, &self.info)?;
                check_picture_size(&fh, self.max_picture_size)?;
                let body = desc.payload.get(fh.bits_consumed.div_ceil(8)..).ok_or(Error::UnexpectedEnd)?;
                let ti = fh.tile_info.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let tiles = parse_whole_frame_tile_group(ti, body, true)?;
                self.submit(fh, tiles, input.pts, false)?;
            }
            _ => return Err(Error::PartitionWalkOutOfRange),
        }
        Ok(ParseStep::Progress)
    }

    fn submit(&mut self, fh: FrameHeader, tiles: Vec<TilePayload>, pts: Option<i64>, replay: bool) -> Result<(), Error> {
        let seq = self.seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?.clone();
        let cc = &seq.color_config;
        let layout = if replay {
            let slot = usize::from(fh.frame_to_show_map_idx.ok_or(Error::PartitionWalkOutOfRange)?);
            self.slots.get(slot).and_then(|slot| slot.as_ref())
                .ok_or(Error::PartitionWalkOutOfRange)?.layout
        } else {
            let fs = fh.frame_size.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
            (fs.upscaled_width, fs.frame_height, cc.bit_depth, cc.mono_chrome,
                cc.subsampling_x, cc.subsampling_y)
        };
        let version = Arc::new(ReferenceVersion { layout, result: OnceLock::new() });
        let mut needed = [false; NUM_REF_FRAMES as usize];
        if replay {
            needed[usize::from(fh.frame_to_show_map_idx.ok_or(Error::PartitionWalkOutOfRange)?)] = true;
        } else {
            if let Some(refs) = fh.inter_refs.as_ref() {
                for &slot in &refs.ref_frame_idx {
                    *needed.get_mut(usize::from(slot)).ok_or(Error::PartitionWalkOutOfRange)? = true;
                }
                if fh.primary_ref_frame != PRIMARY_REF_NONE {
                    needed[usize::from(refs.ref_frame_idx[usize::from(fh.primary_ref_frame)])] = true;
                }
            }
            if let Some(grain) = fh.film_grain_params.as_ref().filter(|grain| grain.apply_grain && grain.predicted) {
                *needed.get_mut(usize::from(grain.film_grain_params_ref_idx))
                    .ok_or(Error::PartitionWalkOutOfRange)? = true;
            }
        }
        let job = Job { id: self.next_id, seq, fh, info: self.info.clone(), slots: self.slots.clone(),
            decode: self.decode, needed, tiles, version: version.clone(), replay };
        self.next_id = self.next_id.wrapping_add(1);
        self.publish_header(&job)?;
        self.ordered.push_back(Ordered { version, shown: replay || job.fh.show_frame, pts });
        self.pending.push_back(job);
        Ok(())
    }

    fn publish_header(&mut self, job: &Job) -> Result<(), Error> {
        let fh = &job.fh;
        if job.replay {
            let source = usize::from(fh.frame_to_show_map_idx.ok_or(Error::PartitionWalkOutOfRange)?);
            for slot in 0..NUM_REF_FRAMES as usize {
                if fh.refresh_frame_flags & (1 << slot) == 0 { continue; }
                self.info.valid[slot] = true;
                self.info.order_hint[slot] = job.info.order_hint[source];
                self.info.frame_id[slot] = job.info.frame_id[source];
                self.info.upscaled_width[slot] = job.info.upscaled_width[source];
                self.info.frame_height[slot] = job.info.frame_height[source];
                self.info.render_width[slot] = job.info.render_width[source];
                self.info.render_height[slot] = job.info.render_height[source];
                self.info.frame_type_is_key[slot] = job.info.frame_type_is_key[source];
                self.info.saved_gm_params[slot] = job.info.saved_gm_params[source];
                self.info.saved_lf_ref_deltas[slot] = job.info.saved_lf_ref_deltas[source];
                self.info.saved_lf_mode_deltas[slot] = job.info.saved_lf_mode_deltas[source];
                self.info.saved_seg_feature_active[slot] = job.info.saved_seg_feature_active[source];
                self.info.saved_seg_feature_data[slot] = job.info.saved_seg_feature_data[source];
                self.slots[slot] = job.slots[source].clone();
            }
            return Ok(());
        }
        let fs = fh.frame_size.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
        let gm = fh.global_motion_params.as_ref().map(|gm| gm.gm_params)
            .unwrap_or_else(crate::uncompressed_header_tail::prev_gm_params_default);
        let (lf_ref, lf_mode) = fh.loop_filter_params.as_ref()
            .map(|lf| (lf.loop_filter_ref_deltas, lf.loop_filter_mode_deltas))
            .unwrap_or((crate::uncompressed_header_tail::LOOP_FILTER_REF_DELTAS_DEFAULT, [0; 2]));
        let disabled = SegmentationParams::disabled();
        let seg = fh.segmentation_params.as_ref().unwrap_or(&disabled);
        for slot in 0..NUM_REF_FRAMES as usize {
            if fh.refresh_frame_flags & (1 << slot) == 0 { continue; }
            self.info.valid[slot] = true;
            self.info.order_hint[slot] = fh.order_hint;
            self.info.frame_id[slot] = fh.current_frame_id;
            self.info.upscaled_width[slot] = fs.upscaled_width;
            self.info.frame_height[slot] = fs.frame_height;
            self.info.render_width[slot] = fs.render_width;
            self.info.render_height[slot] = fs.render_height;
            self.info.frame_type_is_key[slot] = matches!(fh.frame_type, crate::frame_header::FrameType::Key);
            self.info.saved_gm_params[slot] = gm;
            self.info.saved_lf_ref_deltas[slot] = lf_ref;
            self.info.saved_lf_mode_deltas[slot] = lf_mode;
            self.info.saved_seg_feature_active[slot] = seg.segment_feature_active;
            self.info.saved_seg_feature_data[slot] = seg.segment_feature_data;
            self.slots[slot] = Some(job.version.clone());
        }
        Ok(())
    }

    fn parsed_bytes(&self) -> usize {
        self.pending.iter().map(job_bytes).sum::<usize>()
            + self.inputs.iter().filter_map(|input| input.pending.as_ref())
                .map(|pending| pending.original.capacity() + tile_bytes(&pending.tiles)).sum::<usize>()
    }

    fn work_bytes(&self) -> usize {
        self.parsed_bytes() + self.running.iter().map(|job| job.bytes).sum::<usize>()
    }

    fn record_peaks(&mut self) {
        let mut versions = HashSet::new();
        let mut payloads = HashSet::new();
        let mut outputs = HashSet::new();
        let mut sequences = HashSet::new();
        let mut retained_frames = 0;
        let mut retained_bytes = 0;
        let mut visit = |version: &Version| {
            if !versions.insert(Arc::as_ptr(version)) { return; }
            if let Some(Ok(completed)) = version.result.get() {
                let output = completed.output.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                if completed.slot.is_some() || output.is_some() { retained_frames += 1; }
                if let Some(slot) = completed.slot.as_ref() {
                    if payloads.insert(Arc::as_ptr(slot)) { retained_bytes += slot_bytes(slot); }
                }
                if let Some(frame) = output.as_ref() {
                    if outputs.insert(Arc::as_ptr(frame)) { retained_bytes += frame_bytes(frame); }
                }
            }
        };
        for version in self.slots.iter().flatten() { visit(version); }
        for job in &self.pending {
            visit(&job.version);
            for version in job.slots.iter().flatten() { visit(version); }
        }
        for job in &self.running {
            visit(&job.version);
            for version in job.slots.iter().flatten() { visit(version); }
        }
        for entry in &self.ordered { visit(&entry.version); }
        let mut queued_bytes = self.parsed_bytes() + self.running.iter().map(|job| job.bytes).sum::<usize>();
        queued_bytes += self.inputs.iter().map(|input| input.bytes.capacity()).sum::<usize>();
        for seq in self.seq.iter().chain(self.pending.iter().map(|job| &job.seq))
            .chain(self.running.iter().map(|job| &job.seq))
        {
            if sequences.insert(Arc::as_ptr(seq)) {
                queued_bytes += size_of::<SequenceHeader>()
                    + seq.operating_points.capacity() * size_of::<crate::sequence_header::OperatingPoint>();
            }
        }
        self.peaks.peak_retained_frames = self.peaks.peak_retained_frames.max(retained_frames);
        self.peaks.peak_retained_bytes = self.peaks.peak_retained_bytes.max(retained_bytes);
        self.peaks.peak_active_frames = self.peaks.peak_active_frames.max(self.running.len());
        self.peaks.peak_queued_frames = self.peaks.peak_queued_frames.max(self.pending.len());
        self.peaks.peak_queued_bytes = self.peaks.peak_queued_bytes.max(queued_bytes);
    }

    fn join_all(&mut self) {
        for mut running in self.running.drain(..) {
            if running.handle.take().is_some_and(|handle| handle.join().is_err()) {
                self.failure = Some(Error::PartitionWalkOutOfRange);
            }
            if let Some(Err(error)) = running.version.result.get() { self.failure = Some(*error); }
        }
    }
}

impl Drop for FrameThreadDecoder {
    fn drop(&mut self) { self.join_all(); }
}

fn run_job(job: Job, exec: &oxideav_core::ExecutionContext) -> Result<CompletedFrame, Error> {
    let mut refs = SpecRefState {
        info: job.info,
        decode: job.decode,
        slots: core::array::from_fn(|i| {
            if !job.needed[i] { return None; }
            job.slots[i].as_ref().and_then(|version| version.result.get())
                .and_then(|result| result.as_ref().ok()).and_then(|completed| completed.slot.clone())
        }),
    };
    if job.replay {
        let frame = output_existing_frame(&refs, &job.fh, &job.seq)?;
        return Ok(CompletedFrame { slot: None, output: Mutex::new(Some(Arc::new(frame))) });
    }
    let decoded = decode_frame_spec_tiles(&job.seq, &job.fh, &job.tiles, Some(&refs), exec)?;
    let frame = reference_frame_update(&mut refs, &job.fh, decoded)?;
    let slot = if job.fh.refresh_frame_flags == 0 { None }
        else { refs.slots[job.fh.refresh_frame_flags.trailing_zeros() as usize].clone() };
    let output = if job.fh.show_frame { Some(Arc::new(frame)) } else { None };
    Ok(CompletedFrame { slot, output: Mutex::new(output) })
}

fn tile_bytes(tiles: &Vec<TilePayload>) -> usize {
    tiles.capacity() * size_of::<TilePayload>() + tiles.iter().map(|tile| tile.bytes.capacity()).sum::<usize>()
}

fn job_bytes(job: &Job) -> usize {
    size_of::<Job>() + tile_bytes(&job.tiles) + job.fh.tile_info.as_ref().map_or(0, |ti| {
        (ti.mi_col_starts.capacity() + ti.mi_row_starts.capacity()) * size_of::<u32>()
    })
}

fn slot_bytes(slot: &SpecRefSlot) -> usize {
    size_of::<SpecRefSlot>() + size_of::<TileCdfContext>()
        + slot.planes.retained_bytes()
        + slot.plane_dims.capacity() * size_of::<(u32, u32)>()
        + slot.mf_mvs.capacity() * size_of::<i16>() + slot.mf_ref_frames.capacity()
        + slot.segment_ids.capacity() * size_of::<i32>()
}

fn frame_bytes(frame: &SpecFrame) -> usize {
    size_of::<SpecFrame>() + frame.planes.capacity() * size_of::<Vec<u8>>()
        + frame.planes.iter().map(|plane| plane.capacity()).sum::<usize>()
        + frame.plane_dims.capacity() * size_of::<(u32, u32)>()
}
