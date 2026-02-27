# Adapted from vmdpy (MIT License)
# Original source: https://github.com/vrcarva/vmdpy

import torch



def vmd_batch(
        signals: torch.Tensor,
        alpha: float,
        tau: float,
        K: int,
        DC: bool,
        init: int,
        tol: float,
        max_N: int = 2500,
        device: str = "cuda",
):


    signals = signals.float()

    actual_device = signals.device
    B, C, T = signals.shape
    half = T // 2
    f_mirror = torch.zeros((B, C, 2 * T), dtype=signals.dtype, device=actual_device)

    f_mirror[:, :, :half] = torch.flip(signals[:, :, :half], dims=[-1])  # Left flip
    f_mirror[:, :, half: half + T] = signals  # Middle original signal
    f_mirror[:, :, half + T: 2 * T] = torch.flip(signals[:, :, half:], dims=[-1])  # Right flip

    T_mirr = f_mirror.shape[-1]
    t_lin = torch.arange(1, T_mirr + 1, device=actual_device, dtype=signals.dtype) / T_mirr
    freqs = t_lin - 0.5 - (1.0 / T_mirr)

    f_hat = torch.fft.fft(f_mirror, dim=-1)
    f_hat = torch.fft.fftshift(f_hat, dim=-1)

    f_hat_plus = f_hat.clone()
    f_hat_plus[..., : (T_mirr // 2)] = 0.0

    Alpha = alpha * torch.ones(K, device=actual_device, dtype=torch.cfloat)

    u_hat_plus_old = torch.zeros((B, T_mirr, K, C), dtype=torch.cfloat, device=actual_device)
    u_hat_plus_new = torch.zeros_like(u_hat_plus_old)

    lamda_hat_old = torch.zeros((B, T_mirr, C), dtype=torch.cfloat, device=actual_device)
    lamda_hat_new = torch.zeros_like(lamda_hat_old)

    omega_old = torch.zeros((B, K), dtype=torch.float, device=actual_device)
    omega_new = torch.zeros_like(omega_old)

    if init == 1:
        for i in range(K):
            omega_old[:, i] = (0.5 / K) * i
    elif init == 2:
        rnd = torch.rand((B, K), device=actual_device, dtype=signals.dtype)
        ex = torch.exp(torch.log(torch.tensor(1.0, device=actual_device, dtype=signals.dtype)) + (
                torch.log(torch.tensor(0.5, device=actual_device, dtype=signals.dtype)) - torch.log(
            torch.tensor(1.0, device=actual_device, dtype=signals.dtype))) * rnd)
        ex_sorted, _ = torch.sort(ex, dim=1)
        omega_old[:] = ex_sorted
    else:
        omega_old[:] = 0.0

    if DC:
        omega_old[:, 0] = 0.0

    uDiff = tol + 1e-14
    n = 0

    sum_uk = torch.zeros((B, T_mirr, C), dtype=torch.cfloat, device=actual_device)

    f_hat_plus_bt = f_hat_plus.transpose(1, 2)

    while (uDiff > tol) and (n < max_N - 1):
        # Update mode k=0
        sum_uk = sum_uk + (u_hat_plus_old[:, :, K - 1, :] - u_hat_plus_old[:, :, 0, :])

        freq_diff_0 = freqs.unsqueeze(0) - omega_old[:, 0].unsqueeze(1)
        denom_0 = 1.0 + Alpha[0] * (freq_diff_0 ** 2)

        numerator_0 = f_hat_plus_bt - sum_uk - lamda_hat_old / 2.0
        u_hat_plus_new[:, :, 0, :] = numerator_0 / denom_0.unsqueeze(-1)

        if not DC:
            half_mirr = T_mirr // 2
            u_slice = torch.abs(u_hat_plus_new[:, half_mirr:, 0, :]) ** 2
            power_sum = u_slice.sum(dim=(1, 2))
            freq_mat = freqs[half_mirr:].unsqueeze(0)
            weighted = (freq_mat.unsqueeze(-1) * u_slice).sum(dim=(1, 2))

            mask = power_sum > 1e-14
            omega_new[:, 0] = torch.where(
                mask,
                weighted / power_sum,
                torch.zeros_like(power_sum, dtype=torch.float)
            )
        else:
            omega_new[:, 0] = 0.0

        # Update mode k=1..K-1
        for k_i in range(1, K):
            sum_uk = sum_uk + (u_hat_plus_new[:, :, k_i - 1, :] - u_hat_plus_old[:, :, k_i, :])

            freq_diff_k = freqs.unsqueeze(0) - omega_old[:, k_i].unsqueeze(1)
            denom_k = 1.0 + Alpha[k_i] * (freq_diff_k ** 2)

            numerator_k = f_hat_plus_bt - sum_uk - lamda_hat_old / 2.0
            u_hat_plus_new[:, :, k_i, :] = numerator_k / denom_k.unsqueeze(-1)

            u_slice = torch.abs(u_hat_plus_new[:, half_mirr:, k_i, :]) ** 2
            power_sum = u_slice.sum(dim=(1, 2))
            freq_mat = freqs[half_mirr:].unsqueeze(0)
            weighted = (freq_mat.unsqueeze(-1) * u_slice).sum(dim=(1, 2))

            mask = power_sum > 1e-14
            omega_new[:, k_i] = torch.where(
                mask,
                weighted / power_sum,
                torch.zeros_like(power_sum, dtype=torch.float)
            )

        # Dual ascent
        sum_modes = u_hat_plus_new.sum(dim=2)
        lamda_hat_new = lamda_hat_old + tau * (sum_modes - f_hat_plus_bt)

        # Measure convergence
        diff = (u_hat_plus_new - u_hat_plus_old).abs() ** 2
        diff_sum = diff.sum(dim=(1, 2, 3)) / T_mirr
        uDiff = diff_sum.mean()

        # Rotate old/new
        u_hat_plus_old, u_hat_plus_new = u_hat_plus_new, u_hat_plus_old
        omega_old, omega_new = omega_new, omega_old
        lamda_hat_old, lamda_hat_new = lamda_hat_new, lamda_hat_old

        n += 1


    # Post-processing
    final_u_hat = torch.zeros_like(u_hat_plus_old)
    half_mirr = T_mirr // 2

    final_u_hat[:, half_mirr:, :, :] = u_hat_plus_old[:, half_mirr:, :, :]

    idxs = torch.arange(1, half_mirr + 1, device=actual_device)
    idxs = torch.flip(idxs, dims=[0])

    final_u_hat[:, idxs, :, :] = torch.conj(u_hat_plus_old[:, half_mirr:, :, :])

    final_u_hat[:, 0, :, :] = torch.conj(final_u_hat[:, -1, :, :])

    # IFFT to time domain
    u_comp = torch.zeros((B, K, T_mirr, C), dtype=torch.cfloat, device=actual_device)

    for k_i in range(K):
        for c_i in range(C):
            spec_ifftshift = torch.fft.ifftshift(final_u_hat[:, :, k_i, c_i], dim=1)
            u_comp[:, k_i, :, c_i] = torch.fft.ifft(spec_ifftshift, dim=1)

    u_comp = u_comp.real

    # Trim mirrored sections
    start = T_mirr // 4
    end = 3 * T_mirr // 4

    u_final = u_comp[:, :, start:end, :].detach().to(actual_device)

    # Recalculate spectrum
    T_out = u_final.shape[2]
    final_spec = torch.zeros((B, T_out, K, C), dtype=torch.cfloat, device=actual_device)

    for k_i in range(K):
        for c_i in range(C):
            spec_ = torch.fft.fft(u_final[:, k_i, :, c_i], dim=1)
            spec_shifted = torch.fft.fftshift(spec_, dim=1)
            final_spec[:, :, k_i, c_i] = spec_shifted.to(actual_device)

    omega_final = omega_old.real.detach().to(actual_device)

    return u_final, final_spec, omega_final