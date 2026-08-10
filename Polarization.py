"""Polarization feature construction and physics-consistent augmentation.

Pure tensor math over the 4 raw DoFP channels -- no Lightning, no model. Extracted
from Classifier so it can be tested directly: these functions have analytic ground
truth (unpolarized light must give DoLP == 0, a mirrored tile must negate AoLP),
and they produce the campaign's only measured win (DoLP as a 5th input channel,
-2.22 miss@FA5 at the same parameter count).

Channel order is [I90, I45, I0, I135] -- see CH_* below. The polar_flip / polar_rot
augmentations permute those channels to stay physically consistent, which is why
they live next to the Stokes math rather than in a generic augmentation module.
"""

import torch

# Input channel order of the raw DoFP tile.
CH_I90, CH_I45, CH_I0, CH_I135 = 0, 1, 2, 3


def calculate_stokes(x):
        """
        Compute the Stokes parameters from the 4 linear polarization channels.

        Input channel order is [I90, I45, I0, I135] -- see CH_* above.

        Stokes formulas (linear only; a DoFP sensor measures 0/45/90/135, so the
        circular component S3 is not observable and is returned as zero rather
        than being faked from one of the linear channels):
          S0 = I(0°) + I(90°)     total intensity
          S1 = I(0°) - I(90°)
          S2 = I(45°) - I(135°)
          S3 = 0

        Matches the annotator's reference implementation in
        magician_grabber_annotator/visualizeData.py:convertPolarCVMATToRGB
        (ways 9/10), which is what the operator sees when labelling.

        Args:
            x: Tensor of shape (batch_size, 4, height, width).

        Returns:
            Stokes tensor of shape (batch_size, 4, height, width).
        """
        I90  = x[:, CH_I90,  :, :]
        I45  = x[:, CH_I45,  :, :]
        I0   = x[:, CH_I0,   :, :]
        I135 = x[:, CH_I135, :, :]
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135
        S3 = torch.zeros_like(S0)
        return torch.stack((S0, S1, S2, S3), dim=1)

def calculate_DoLP(x):
        """
        Compute the Degree of Linear Polarization from Stokes parameters.
        DoLP = sqrt(S1^2 + S2^2) / S0, representing the fraction of light that
        is linearly polarized. Value range: [0, 1].

        Args:
            x: Stokes tensor of shape (batch_size, 4, height, width).

        Returns:
            DoLP tensor of shape (batch_size, height, width).
        """
        S0 = x[:, 0, :, :]
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        #S3 = x[:, 3, :, :]  # Not used in DoLP
        # Clamped like the annotator's reference implementation: S0 -> 0 on a dark
        # tile would otherwise emit an unbounded value into the network.
        DoLP = torch.sqrt(S1**2 + S2**2) / (S0 + 1e-6)
        return torch.clamp(DoLP, 0.0, 1.0)

def calculate_AoLP(x):
        """
        Compute the Angle of Linear Polarization from Stokes parameters.
        AoLP = 0.5 * atan2(S2, S1), representing the orientation angle of the
        electric field oscillation. Value range: [-pi/2, pi/2].

        Args:
            x: Stokes tensor of shape (batch_size, 4, height, width).

        Returns:
            AoLP tensor of shape (batch_size, height, width).
        """
        S1 = x[:, 1, :, :]
        S2 = x[:, 2, :, :]
        AoLP = 0.5 * torch.atan2(S2, S1)
        return AoLP

def build_input_features(x, *, in_channels, monochrome=False, AoLP=False, DoLP=False,
                         Unpolarized=False, MaxPolarization=False, MinPolarization=False,
                         RangePolarization=False):
        """
        Build model input by appending derived channels.
        All polarization-derived features are computed from the original 4 channels only.
        Expects x shape: [B, >=4, H, W] at input (normally [B,4,H,W]).
        Returns x shape: [B, in_channels, H, W]

        Accepts uint8 input (0-255) and normalises to float32 [0,1] on the GPU.
        This keeps PCIe transfer bandwidth 4× lower than sending float32 tensors.
        """
        # Dequantize uint8 → float32 on the device where x already lives.
        # The multiplication is a single fused kernel; cost is negligible vs. the
        # conv ops that follow.  float() preserves the current device and layout.
        if x.dtype == torch.uint8:
            x = x.float() * (1.0 / 255.0)

        if x.shape[1] < 4:
            raise ValueError(f"Expected at least 4 channels for polarization input, got {x.shape[1]}")

        if monochrome:
            # simulate a regular monochrome camera: intensity only, no polarization
            x = x[:, 0:4, :, :].mean(dim=1, keepdim=True).expand(-1, 4, -1, -1).contiguous()

        pol = x[:, 0:4, :, :]  # original polarization channels only

        # AoLP / DoLP from Stokes computed on the original channels
        if AoLP or DoLP:
            stokes = calculate_stokes(pol)

            if DoLP:
                dolp = calculate_DoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, dolp), dim=1)

            if AoLP:
                aolp = calculate_AoLP(stokes).unsqueeze(1)  # [B,1,H,W]
                x = torch.cat((x, aolp), dim=1)

        # Unpolarized = mean over original 4 channels
        if Unpolarized:
            mon = pol.mean(dim=1, keepdim=True)
            x = torch.cat((x, mon), dim=1)

        # Max / Min / Range over original 4 channels
        if MaxPolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            x = torch.cat((x, max_pol), dim=1)

        if MinPolarization:
            min_pol = pol.min(dim=1, keepdim=True)[0]
            x = torch.cat((x, min_pol), dim=1)

        if RangePolarization:
            max_pol = pol.max(dim=1, keepdim=True)[0]
            min_pol = pol.min(dim=1, keepdim=True)[0]
            range_pol = max_pol - min_pol
            x = torch.cat((x, range_pol), dim=1)

        # Final sanity: ensure model sees the expected channel count
        if x.shape[1] != in_channels:
            raise ValueError(f"Feature builder produced {x.shape[1]} channels, expected {in_channels}. "
                             f"(Flags: DoLP={DoLP}, AoLP={AoLP}, Unpolarized={Unpolarized}, "
                             f"MaxPolarization={MaxPolarization}, MinPolarization={MinPolarization}, "
                             f"RangePolarization={RangePolarization})")
        return x

def augment_train_batch(x, *, training=True, gain_jitter=0.0, polar_flip=False,
                        polar_rot=False, channel_jitter=0.0):
        """
        Training-only augmentation on the raw 4-channel polarization batch.

        gain_jitter: per-sample multiplicative gain, log-uniform in
        [1/(1+j), 1+j], identical on all 4 channels — emulates the exposure
        differences between recording sessions (dataset names carry exposures
        150..4500) without breaking polarization ratios.

        polar_flip: random horizontal/vertical mirror. A mirror maps the angle
        of linear polarization theta -> -theta, i.e. S2 = I45-I135 negates, so
        the 45deg and 135deg channels (1 and 3) MUST be swapped along with the
        pixel flip. Verified empirically on the static-camera
        measure65mmheight_* sets: corr(S2, mirror(S2)) = -0.86..-0.92 while the
        invariant S1 control stays positive.

        Returns float32 in [0,1] (dequantizes uint8 first so downstream
        build_input_features skips its own dequantization consistently).
        """
        if not training or (gain_jitter <= 0.0 and not polar_flip
                                 and channel_jitter <= 0.0 and not polar_rot):
            return x
        if x.dtype == torch.uint8:
            x = x.float() * (1.0 / 255.0)

        if polar_flip:
            swap = [0, 3, 2, 1]  # 45deg <-> 135deg
            for dim in (-1, -2):  # horizontal, then vertical mirror
                sel = torch.rand(x.shape[0], device=x.device) < 0.5
                if sel.any():
                    x[sel] = x[sel].flip(dim)[:, swap, :, :]

        if polar_rot:
            # +/-90deg rotations complete the dihedral group the mirrors start
            # (180deg = H+V mirror is already covered). Stokes rotation by 90deg
            # negates S1 and S2 -> swap I0<->I90 AND I45<->I135. In the H5/model
            # channel order [p90, p45, p0, p135] that is the permutation [2,3,0,1].
            rot_swap = [2, 3, 0, 1]
            for k in (1, 3):  # 90 and 270 degrees, each on a random 1/4 of the batch
                sel = torch.rand(x.shape[0], device=x.device) < 0.25
                if sel.any():
                    x[sel] = torch.rot90(x[sel], k, dims=(-2, -1))[:, rot_swap, :, :]

        if gain_jitter > 0.0:
            j = float(gain_jitter)
            lo = torch.log(torch.tensor(1.0 / (1.0 + j)))
            hi = torch.log(torch.tensor(1.0 + j))
            g = torch.exp(torch.empty(x.shape[0], 1, 1, 1, device=x.device)
                          .uniform_(float(lo), float(hi)))
            x = torch.clamp(x * g, 0.0, 1.0)

        if channel_jitter > 0.0:
            # INDEPENDENT per-channel gains: emulates the 6 strobed scene lights,
            # whose changes shift the polarization channel proportions (measured
            # signature swing between lights: L1 up to ~0.30 on the normalized
            # 4-vector, i.e. per-channel relative changes up to ~+/-40%). This is
            # the augmentation counterpart of the annotator's canonical-light
            # remap: instead of normalizing lights away at inference, train the
            # model to be invariant to them.
            j = float(channel_jitter)
            lo = torch.log(torch.tensor(1.0 / (1.0 + j)))
            hi = torch.log(torch.tensor(1.0 + j))
            g = torch.exp(torch.empty(x.shape[0], x.shape[1], 1, 1, device=x.device)
                          .uniform_(float(lo), float(hi)))
            x = torch.clamp(x * g, 0.0, 1.0)

        return x

def add_input_noise(x, *, training=True, noise_std=0.0, noise_clip=None):
        """
        Add Gaussian noise to the input during training for regularization.
        Only active when training is True and noise_std > 0.

        Args:
            x: Input tensor.

        Returns:
            Input tensor with noise added.
        """
        if training and noise_std > 0.0:
            noise = torch.randn_like(x) * noise_std
            if noise_clip is not None:
                noise = torch.clamp(noise, -noise_clip, noise_clip)
            x = x + noise
        return x
