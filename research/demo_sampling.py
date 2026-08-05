"""
Örnek kullanım demosu – pytest testi DEĞİLDİR.
Çalıştırmak için: python research/demo_sampling.py
"""


from research.sampling import ParameterSpace, RandomSampler, LatinHypercubeSampler

params = [
    ParameterSpace("stop_loss", 0.08, 0.12),
    ParameterSpace("trailing", 0.20, 0.30),
    ParameterSpace("adx", 28, 35, distribution="choice", values=[28,30,32,35]),
]

print("=== Random ===")
rs = RandomSampler(params)
print(rs.sample(5))

print("\n=== Latin Hypercube ===")
lhs = LatinHypercubeSampler(params)
print(lhs.sample(5))

print("\n=== Sobol ===")
try:
    from research.sampling import SobolSampler
    sobol = SobolSampler(params)
    print(sobol.sample(5))
except ImportError:
    print("SobolSampler henüz implemente edilmedi veya scipy yok.")