//! Time one sage-v1 policy's inference per decision, outside the guest, with
//! the same tract version and the same loading steps as guest/src/step.rs:
//! inputs by name (obs [1, 78], state [1, S] carried from step to step,
//! truth [1, 7], cwnd [1, 1], cc_info [1, C] for agent_cubic's 5 or
//! agent_rate's 7 statistics), optimized and made runnable once, then run
//! once per decision. The inputs are fixed values, so this measures the
//! graph's cost, not a run's.
//!
//! Usage: policy-bench POLICY.onnx [CALLS]   (prints median, p90, p99 in us)

use std::time::Instant;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::infer::Factoid;

const OBS: usize = 78;
const TRUTH: usize = 7;

fn main() -> TractResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .expect("usage: policy-bench POLICY.onnx [CALLS]");
    let calls: usize = args.get(2).map_or(20_000, |n| n.parse().expect("CALLS"));
    let mut model = tract_onnx::onnx().model_for_path(path)?;
    let names: Vec<String> = model
        .inputs
        .iter()
        .map(|o| model.node(o.node).name.clone())
        .collect();
    let mut widths = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let w = match name.as_str() {
            "obs" => OBS,
            "truth" => TRUTH,
            "cwnd" => 1,
            "state" | "cc_info" => {
                let fact = model.input_fact(i)?;
                let dims = fact
                    .shape
                    .dims()
                    .last()
                    .and_then(|d| d.concretize())
                    .and_then(|d| d.to_i64().ok())
                    .expect("fixed width");
                usize::try_from(dims).expect("input width")
            }
            other => panic!("input {other:?} is not obs, state, truth, cwnd or cc_info"),
        };
        model = model.with_input_fact(i, f32::fact([1, w]).into())?;
        widths.push(w);
    }
    let plan = model.into_optimized()?.into_runnable()?;
    let row = |w: usize, v: f32| -> TValue {
        tract_ndarray::Array2::from_elem((1, w), v)
            .into_tensor()
            .into()
    };
    let mut state: Option<TValue> = None;
    let mut times = Vec::with_capacity(calls);
    for k in 0..calls + 1000 {
        let inputs: TVec<TValue> = names
            .iter()
            .zip(&widths)
            .map(|(n, &w)| match n.as_str() {
                "obs" => row(w, 0.5),
                "truth" => row(w, 1.0),
                "cwnd" => row(w, 40.0),
                // A typical period: 20 RTT samples, 10 Mbit/s, no loss, 20 ms
                // long, 20 ms min RTT (agent_cubic); 20 ms long, 20 sent and
                // delivered, 20 samples of 20 ms mean, no rise (agent_rate).
                "cc_info" => {
                    let v: &[f32] = if w == 5 {
                        &[20.0, 1.25e6, 0.0, 20_000.0, 20_000.0]
                    } else {
                        &[20_000.0, 20.0, 20.0, 0.0, 20.0, 20_000.0, 0.0]
                    };
                    tract_ndarray::Array2::from_shape_vec((1, w), v.to_vec())
                        .expect("cc_info is 5 or 7 wide")
                        .into_tensor()
                        .into()
                }
                _ => state.clone().unwrap_or_else(|| row(w, 0.0)),
            })
            .collect();
        let t = Instant::now();
        let mut out = plan.run(inputs)?;
        let dt = t.elapsed();
        if out.len() > 1 {
            state = Some(out.remove(1));
        }
        if k >= 1000 {
            times.push(dt.as_secs_f64() * 1e6); // the first 1000 calls warm up
        }
    }
    times.sort_by(f64::total_cmp);
    let q = |p: f64| times[((times.len() - 1) as f64 * p) as usize];
    println!(
        "{path}: inputs {names:?}; {calls} calls: median {:.1} us, p90 {:.1} us, p99 {:.1} us",
        q(0.5),
        q(0.9),
        q(0.99)
    );
    Ok(())
}
