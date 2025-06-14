use crate::flat_kernel_cache::FlatKernelCache;

pub fn select_working_set_wss2_flat_cache(
    alphas: &[f64],
    y: &[f64],
    grad: &[f64],
    c: f64,
    kernel_cache: &mut FlatKernelCache,
    active_indices: &[usize],
) -> Option<((usize, usize), f64)> {
    let mut max_violation = 0.0;
    let mut i_opt = None;

    for (idx_pos, &i) in active_indices.iter().enumerate() {
        let violation = y[i] * grad[i];
        if ((alphas[i] < c && violation < -1e-3) || (alphas[i] > 0.0 && violation > 1e-3))
            && violation.abs() > max_violation
        {
            max_violation = violation.abs();
            i_opt = Some(idx_pos);
        }
    }
    let ii = i_opt?;
    let i = active_indices[ii];
    let gi = grad[i];
    let kii = kernel_cache.get(i, i);

    let mut max_gain = -1.0;
    let mut j_opt = None;

    for (idx_pos, &j) in active_indices.iter().enumerate() {
        if j == i { continue; }
        let kjj = kernel_cache.get(j, j);
        let kij = kernel_cache.get(i, j);
        let eta = 2.0 * kij - kii - kjj;
        if eta >= 0.0 { continue; }
        let gain = ((gi - grad[j]).powi(2)) / (-eta);
        if gain > max_gain {
            max_gain = gain;
            j_opt = Some(idx_pos);
        }
    }
    let jj = j_opt?;

    Some(((ii, jj), max_violation))
}
