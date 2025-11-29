import torch
import torch.fft


def colored_noise(x: torch.Tensor, beta: float):
    noise = torch.randn_like(x)
    fft = torch.fft.rfft(noise, dim=-1)
    n = x.size(-1)
    freqs = torch.fft.rfftfreq(n, device=x.device)
    scale = 1.0 / (freqs + 1e-8).pow(beta / 2.0)
    scale[0] = 0.0
    colored = torch.fft.irfft(fft * scale, n=n, dim=-1)
    if colored.std() > 0:
        colored = colored / colored.std()
    return colored


def white_noise(x):
    return torch.randn_like(x)


def pink_noise(x):
    return colored_noise(x, beta=1.0)


def brown_noise(x):
    return colored_noise(x, beta=2.0)


def blue_noise(x):
    return colored_noise(x, beta=-1.0)


def violet_noise(x):
    return colored_noise(x, beta=-2.0)


def black_noise(x):
    return colored_noise(x, beta=3.0)
