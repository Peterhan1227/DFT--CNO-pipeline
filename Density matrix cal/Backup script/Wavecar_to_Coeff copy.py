import numpy as np
from pathlib import Path
from vaspwfc import vaspwfc

wavecar_path = Path(__file__).resolve().parent / "Data" / "WAVECAR"
output_dir = wavecar_path.parent
rho_file = output_dir / "density_matrix.npy"
wfc = vaspwfc(str(wavecar_path))

print(f"Loaded WAVECAR: {wavecar_path}")
print(
    f"nkpts={wfc._nkpts}, nbands={wfc._nbands}, "
    f"ngrid={tuple(wfc._ngrid)}, encut={wfc._encut}"
)

ispin = 1
occ_tol = 1e-6

# FFT grid from vaspwfc
ngrid = wfc._ngrid
Nx, Ny, Nz = ngrid
Nr = Nx * Ny * Nz

rho = np.zeros((Nr, Nr), dtype=np.complex128)

for ik in range(1, wfc._nkpts + 1):
    # occupied bands at this k
    occ_all = wfc._occs[ispin - 1, ik - 1, :]

    # occ_all = np.ones_like(occ_all)

    bands = np.where(occ_all > occ_tol)[0] + 1
    occ = occ_all[bands - 1]

    if np.max(occ) > 1.5:
        occ = occ / 2.0
    
    # G-vectors for this k
    gvec = wfc.gvectors(ik)  # shape: (nG, 3)

    # read C_{nk}(G), only loop over bands
    Ck = np.stack([
        wfc.readBandCoeff(
            ispin=ispin,
            ikpt=ik,
            iband=ib,
            norm=True
        )
        for ib in bands
    ])
    # Ck shape: (n_occ_bands, nG)

    # put coefficients onto FFT grid
    coeff_grid = np.zeros((len(bands), Nx, Ny, Nz), dtype=np.complex128)

    gx = gvec[:, 0] % Nx
    gy = gvec[:, 1] % Ny
    gz = gvec[:, 2] % Nz

    coeff_grid[:, gx, gy, gz] = Ck

    # batched inverse FFT: sum_G C_{nk}(G) exp(i G.r)
    u_r = np.fft.ifftn(coeff_grid, axes=(1, 2, 3)) * np.sqrt(Nr)

    # flatten real-space grid
    Psi = u_r.reshape(len(bands), Nr)

    # if ik == 1:
    #     norms = np.sum(np.abs(Psi)**2, axis=1)
    #     print("First 10 real-space norms:", norms[:10])
    #     print("Mean norm:", norms.mean())

    #     S = Psi @ Psi.conj().T
    #     print("Top-left overlap matrix abs:")
    #     print(np.abs(S[:8, :8]))

    # add band occupations
    weighted_Psi = occ[:, None] * Psi

    # C_k(r,r') = sum_n f_nk psi_nk(r) psi_nk*(r')
    rho += Psi.T @ weighted_Psi.conj()

    if ik == 1 or ik % 20 == 0 or ik == wfc._nkpts:
        print(f"Processed k-point {ik}/{wfc._nkpts} with {len(bands)} occupied bands")

rho /= wfc._nkpts

finite_ok = np.isfinite(rho).all()
hermitian_error = np.max(np.abs(rho - rho.conj().T))
trace_rho = np.trace(rho).real

print(f"rho shape: {rho.shape}")
print(f"Finite entries: {finite_ok}")
print(f"Max Hermitian deviation: {hermitian_error:.3e}")
print(f"Trace(rho): {trace_rho:.8f}")

print( (np.sort(np.linalg.eigvalsh(rho))[::-1])[:10]/trace_rho)

# np.save(rho_file, rho)
# file_size_mb = rho_file.stat().st_size / (1024 ** 2)
# print(f"Saved density matrix to: {rho_file}")
# print(f"Saved file size: {file_size_mb:.2f} MB")
