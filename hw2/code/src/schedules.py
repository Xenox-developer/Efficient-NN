def interpolate(a, b, fraction):
    return a + (b - a) * fraction


def schedule(config, step, total_steps):
    if total_steps < 2 or not 0 <= step < total_steps:
        raise ValueError('need at least two updates and a valid zero-based step')
    p = step / (total_steps - 1)
    kind = config['schedule']
    if kind == 'range':
        return interpolate(config['lr_start'], config['lr_end'], p), 0.9
    if kind == 'multistep':
        decays = sum(p >= x for x in config.get('milestones', [0.5, 0.75]))
        return config['lr'] * config.get('gamma', 0.1) ** decays, config.get('momentum', 0.9)
    if kind != 'onecycle':
        raise ValueError(f'unknown schedule: {kind}')
    up, down = config.get('up_fraction', 0.45), config.get('down_end', 0.9)
    lo, hi, final = config['lr_start'], config['lr_max'], config['lr_final']
    mh, ml = config.get('momentum_max', 0.95), config.get('momentum_min', 0.85)
    if not 0 < up < down < 1:
        raise ValueError('onecycle phase boundaries must satisfy 0 < up < down < 1')
    if p <= up:
        return interpolate(lo, hi, p / up), interpolate(mh, ml, p / up)
    if p <= down:
        q = (p - up) / (down - up)
        return interpolate(hi, lo, q), interpolate(ml, mh, q)
    return interpolate(lo, final, (p - down) / (1 - down)), mh
