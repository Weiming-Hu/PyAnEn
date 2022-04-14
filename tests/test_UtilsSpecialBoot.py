import numpy as np

from PyAnEn.utils_special_boot import corr
from PyAnEn.utils_special_boot import brier_decomp
from PyAnEn.VerifyProbGaussian import VerifyProbGaussian

np.random.seed(42)


def test_corr():
    f = np.random.rand(5, 6, 7, 8)
    o = np.random.rand(5, 6, 7, 8)
    
    r1 = corr(f, o, None, 300)
    r2 = corr(f, o, None)
    
    print(r1)
    print(r2)
    assert r1[0] <= r2 <= r1[2]
     

def test_corr_gaussian():
    verifier = VerifyProbGaussian(
        f={'mu': np.random.rand(5, 6, 7, 8) + 10, 'sigma': np.random.rand(5, 6, 7, 8) + 0.5},
        o=np.random.rand(5, 6, 7, 8), avg_axis=[0, 1, 3])
    
    r1 = verifier.set_boot_samples(300).corr()
    r2 = verifier.set_boot_samples(None).corr()
    
    print(r1)
    print(r2)
    
    assert np.all(r1[0] <= r2)
    assert np.all(r2 <= r1[2])
    

def test_brier_decomp():
    f = np.arange(1, 101) / 100
    o = f > 0.3
    
    brier = np.mean((f - o) ** 2)
    rel1, res1, unc1 = brier_decomp(f, o, None, 300)
    rel2, res2, unc2 = brier_decomp(f, o, None)
    
    print(brier)
    print(rel1, res1, unc1)
    print(rel2, res2, unc2)
    
    assert np.abs(rel2 - res2 + unc2 - brier) < 0.001
    assert rel1[0] <= rel2 <= rel1[2]
    assert res1[0] <= res2 <= res1[2]
    assert unc1[0] <= unc2 <= unc1[2]
    

def test_brier_decomp_gaussian():
    verifier = VerifyProbGaussian(
        f={'mu': np.random.rand(2, 6, 7, 3) + 10,
           'sigma': np.random.rand(2, 6, 7, 3) + 0.5},
        o=np.random.rand(2, 6, 7, 3) + 10, avg_axis=[1, 2])
    
    verifier.f['mu'][0] += 100
    verifier.o[0] += 100
    
    brier = verifier.set_boot_samples(None).brier(over=10.5)
    r1 = verifier.set_boot_samples(300).brier_decomp(over=10.5)
    r2 = verifier.set_boot_samples(None).brier_decomp(over=10.5)
    
    assert brier.shape == (2, 3)
    assert len(r1) == 3
    assert r1[0].shape == (3, 2, 3)
    assert r1[1].shape == (3, 2, 3)
    assert r1[2].shape == (3, 2, 3)
    assert len(r2) == 3
    assert r2[0].shape == (2, 3)
    assert r2[1].shape == (2, 3)
    assert r2[2].shape == (2, 3)
    
    assert np.max(np.abs(r2[0] - r2[1] + r2[2] - brier)) < 0.01
    assert np.all(r1[0][0] <= r2[0])
    assert np.all(r2[0] <= r1[0][2])
    
    assert np.all(r1[1][0] <= r2[1])
    assert np.all(r2[1] <= r1[1][2])
    
    assert np.all(r1[2][0] <= r2[2])
    assert np.all(r2[2] <= r1[2][2])
