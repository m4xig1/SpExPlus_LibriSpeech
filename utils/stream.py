import torch
from torchaudio.transforms import Fade


def separate_sources(
    model,
    mix,
    ref,
    ref_len,
    device="cuda" if torch.cuda.is_available else "cpu",
    segment_len=0.3,
    overlap=0.01,
    fade_shape="linear",
    sample_rate=16000,
):
    """
    Makes predictions on batch by splitting it by chunks with `segment_len` size with `overlap`
    """
    if mix.dim() == 1:
        mix = torch.unsqueeze(mix, 0)
        
    batch_size = mix.shape[0]
    pred_shape = mix.shape[-1]

    chunk_len = int(sample_rate * segment_len * (1 + overlap))
    overlap_frames = sample_rate * overlap
    start, end = 0, chunk_len

    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape=fade_shape)

    pred = {
        "short": torch.zeros(batch_size, pred_shape, device=device),
        "mid": torch.zeros(batch_size, pred_shape, device=device),
        "long": torch.zeros(batch_size, pred_shape, device=device),
        "logits": None,
    }
    torch.zeros(batch_size, pred_shape, device=device)

    while start < pred_shape - overlap_frames:
        chunk = mix[:, start:end]
        # we need to calculate speaker embedding only onces
        chunk_pred = model(chunk, ref, ref_len, start == 0)

        for i in "short", "mid", "long":
            pred[i][:, start:end] += fade(chunk_pred[i])
        

        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
            pred["logits"] = chunk_pred["logits"]
        else:
            start += chunk_len
            pred["logits"] += chunk_pred["logits"]
        end += chunk_len
        if end >= pred_shape:
            fade.fade_out_len = 0

    pred["logits"] /= batch_size  # mean of logits
    return pred

