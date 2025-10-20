from kalmed.data.simulate import HRSimConfig, simulate_hr
def test_simulate_shapes():
    x, z, A, Q, H, r = simulate_hr(HRSimConfig(n=100))
    assert x.shape == (2, 100)
    assert z.shape == (100,)
