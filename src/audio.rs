//! PCM in, 16 kHz mono f32 out.
//!
//! A chunk of input is whatever the client sent; a sample (or a frame of
//! several channels) may straddle two chunks, so the bytes left over are kept
//! for the next one. Channels are averaged; any rate is resampled to 16 kHz
//! (rubato's FFT resampler, in fixed blocks).

use rubato::{FftFixedIn, Resampler};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// What the engines take.
pub const RATE: u32 = 16_000;

/// The input's sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Signed 16-bit little-endian (`pw-record --format s16`, `arecord -f S16_LE`).
    #[default]
    S16le,
    /// 32-bit float little-endian.
    F32le,
}

impl Format {
    fn width(self) -> usize {
        match self {
            Format::S16le => 2,
            Format::F32le => 4,
        }
    }
}

/// The input's shape, as `listen` was told it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub rate: u32,
    pub format: Format,
    pub channels: u16,
}

impl Shape {
    pub fn check(&self) -> Result<(), String> {
        if !(4_000..=384_000).contains(&self.rate) {
            return Err(format!("--rate {} is outside 4000..=384000", self.rate));
        }
        if !(1..=32).contains(&self.channels) {
            return Err(format!("--channels {} is outside 1..=32", self.channels));
        }
        Ok(())
    }
}

/// Bytes of one shape in, 16 kHz mono out.
pub struct Converter {
    shape: Shape,
    carry: Vec<u8>,
    resampler: Option<(FftFixedIn<f32>, Vec<f32>)>,
}

const BLOCK: usize = 1024;

impl Converter {
    pub fn new(shape: Shape) -> Result<Self, String> {
        shape.check()?;
        let resampler = if shape.rate == RATE {
            None
        } else {
            let r = FftFixedIn::<f32>::new(shape.rate as usize, RATE as usize, BLOCK, 2, 1).map_err(|e| format!("resampler: {e}"))?;
            Some((r, Vec::new()))
        };
        Ok(Self { shape, carry: Vec::new(), resampler })
    }

    /// Converts `bytes`, keeping a partial frame for the next call.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<f32> {
        let frame = self.shape.format.width() * self.shape.channels as usize;
        self.carry.extend_from_slice(bytes);
        let whole = self.carry.len() / frame * frame;
        let mono = downmix(&self.carry[..whole], self.shape);
        self.carry.drain(..whole);
        self.resample(mono, false)
    }

    /// The rest: the resampler's tail (a partial frame is dropped).
    pub fn finish(&mut self) -> Vec<f32> {
        self.resample(Vec::new(), true)
    }

    fn resample(&mut self, mono: Vec<f32>, last: bool) -> Vec<f32> {
        let Some((r, pending)) = self.resampler.as_mut() else { return mono };
        pending.extend(mono);
        let mut out = Vec::new();
        while pending.len() >= r.input_frames_next() {
            let n = r.input_frames_next();
            if let Ok(mut o) = r.process(&[&pending[..n]], None) {
                out.append(&mut o[0]);
            }
            pending.drain(..n);
        }
        if last && !pending.is_empty() {
            if let Ok(mut o) = r.process_partial(Some(&[&pending[..]]), None) {
                out.append(&mut o[0]);
            }
            pending.clear();
            // The delay line's tail, so the last words are not cut.
            if let Ok(mut o) = r.process_partial::<&[f32]>(None, None) {
                out.append(&mut o[0]);
            }
        }
        out
    }
}

fn downmix(bytes: &[u8], shape: Shape) -> Vec<f32> {
    let ch = shape.channels as usize;
    let sample = |i: usize| -> f32 {
        match shape.format {
            Format::S16le => i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]) as f32 / 32768.0,
            Format::F32le => {
                let v = f32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
                if v.is_finite() {
                    v.clamp(-1.0, 1.0)
                } else {
                    0.0
                }
            }
        }
    };
    let n = bytes.len() / shape.format.width();
    (0..n / ch).map(|f| (0..ch).map(|c| sample(f * ch + c)).sum::<f32>() / ch as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s16(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn a_sample_split_across_chunks_is_kept() {
        let mut c = Converter::new(Shape { rate: RATE, format: Format::S16le, channels: 1 }).unwrap();
        let b = s16(&[16384, -16384, 8192]);
        let mut out = c.push(&b[..3]);
        out.extend(c.push(&b[3..]));
        assert_eq!(out, vec![0.5, -0.5, 0.25]);
    }

    #[test]
    fn stereo_is_averaged() {
        let mut c = Converter::new(Shape { rate: RATE, format: Format::S16le, channels: 2 }).unwrap();
        assert_eq!(c.push(&s16(&[16384, 0, -16384, -16384])), vec![0.25, -0.5]);
        let mut f = Converter::new(Shape { rate: RATE, format: Format::F32le, channels: 1 }).unwrap();
        let b: Vec<u8> = [0.5f32, f32::NAN, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(f.push(&b), vec![0.5, 0.0, 1.0]);
    }

    #[test]
    fn forty_eight_k_becomes_sixteen_k() {
        let mut c = Converter::new(Shape { rate: 48_000, format: Format::S16le, channels: 1 }).unwrap();
        // One second of a 440 Hz tone, sent in odd-sized chunks.
        let tone: Vec<i16> = (0..48_000).map(|i| ((i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 10_000.0) as i16).collect();
        let bytes = s16(&tone);
        let mut out = Vec::new();
        for ch in bytes.chunks(1001) {
            out.extend(c.push(ch));
        }
        out.extend(c.finish());
        assert!((out.len() as i64 - 16_000).abs() < 1100, "{}", out.len());
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / out.len() as f32).sqrt();
        assert!(rms > 0.15 && rms < 0.25, "{rms}");
    }

    #[test]
    fn shapes_are_checked() {
        assert!(Converter::new(Shape { rate: 100, format: Format::S16le, channels: 1 }).is_err());
        assert!(Converter::new(Shape { rate: RATE, format: Format::S16le, channels: 0 }).is_err());
    }
}
