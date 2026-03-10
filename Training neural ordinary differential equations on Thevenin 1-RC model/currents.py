# Different types of currents

def constant_current(t, I0=1.0):
    """
    Constant current profile.
    I0 [A] : constant discharge current (e.g., 1.0 A).
    """
    return I0

def pulse_train_tensor(t, Ih=2.0, Il=0.0, Ton=40.0, Toff=40.0):
    """
    Square-wave pulse profile (zero-order hold).
    Ih  [A] : pulse ON (high) current.
    Il  [A] : pulse OFF (rest) current (often 0).
    Ton [s] : ON duration each cycle.
    Toff[s] : OFF duration each cycle.
    Period = Ton + Toff; duty = Ton / (Ton + Toff).
    """    
    t = torch.as_tensor(t)
    period = Ton + Toff
    i = torch.full(t.shape, Il)
    i[(t%period) < Ton] = Ih
    
    return i

def pulse_train(t, Ih=2.0, Il=0.0, Ton=20.0, Toff=40.0):
    """
    Square-wave pulse profile (zero-order hold).
    Ih  [A] : pulse ON (high) current.
    Il  [A] : pulse OFF (rest) current (often 0).
    Ton [s] : ON duration each cycle.
    Toff[s] : OFF duration each cycle.
    Period = Ton + Toff; duty = Ton / (Ton + Toff).
    """
    period = Ton + Toff
    return Ih if (t%period) < Ton else Il

def sinusoidal(t, Imean=1.0, Iamp=0.5, f=0.01):
    """
    Sinusoidal current around a mean.
    Imean [A] : baseline current.
    Iamp [A]  : amplitude (keep ≤ Imean to avoid negative current i.e. charging).
    f     [Hz]: frequency (e.g., 0.01 Hz = 100 s period).
    """
    return Imean + Iamp * np.sin(2 * np.pi * f * t)

def randomized(t, Imin=0.5, Imax=1.5, seed=42):
    """
    Pseudo-random current held per integer second.
    Imin/Imax [A] : uniform range.
    seed      [-] : reproducibility (same sequence each run).
    """
    np.random.seed(int(t) + seed)  # hold value within each [k, k+1)
    return float(np.random.uniform(Imin, Imax))
