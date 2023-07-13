import torch
import yaml


def test_produce_noisy_samples(testing_func):

    ts = torch.Tensor([2, 3, 2, 1]).long().reshape(-1,)
    x0 = torch.Tensor([3, 3, 3, 3]).reshape(-1,1)
    betas = torch.Tensor([0.1, 0.2, 0.3, 0.4]).reshape(-1,)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    noise = torch.tensor([0.4199, 0.9844, 1.1147, 0.1688]).reshape(-1, 1)

    noisy_test = testing_func(x0, ts, alphas_bar, noise)
    noisy_actual = torch.tensor([2.4255, 2.4719, 2.9148, 2.6349]).reshape(-1, 1)

    if noisy_test.shape != noisy_actual.shape:
        raise ValueError(f"Shape of produced noisy samples does not match. As per the example, the shape should be {noisy_actual.shape} and it is {noisy_test.shape}")
    else:
        print("\033[92m\033[1m✓ Shape test passed! \033[0m")

    assert torch.allclose(noisy_test, noisy_actual, atol=1e-3)
    
    print("\033[92m\033[1m✓ Value test passed! \033[0m")


def test_improved_beta_schedule(schedule):
    betas = torch.tensor([0.1012940794, 0.2795438460, 0.4736353534, 0.7240523691, 0.9990000000])
    assert torch.all(torch.isclose(schedule(5), betas))
    print("\033[92m\033[1m✓ Value test passed! \033[0m")


def test_unet(unetclass):
    unet = unetclass(28, 112, 1)
    with open(r'unet_string.yaml') as file:
        unet_string = yaml.full_load(file)
    assert unet_string == unet.__str__()
    print("\033[92m\033[1m✓ network structure test passed! \033[0m")

def test_linear_beta_schedule(schedule):

    beta_start = 0.001
    beta_end = 0.02
    num_steps = 5

    betas_true = torch.tensor([0.0010, 0.0058, 0.0105, 0.0152, 0.0200])
    betas_schedule = schedule(beta_start, beta_end, num_steps)
    
    assert torch.allclose(betas_schedule, betas_true, atol=1e-2)

    print("\033[92m\033[1m✓ Value test passed! \033[0m")


def test_forward_continuous(gamma, sigma_squared, produce_noisy_samples_continuous):
    t = torch.tensor([0.3, 0.5, 0.7])
    g = gamma(t)
    assert torch.all(torch.isclose(g, torch.tensor([0.62954998, 0.28118289, 0.08435258]))), "error in gamma-function"
    s = sigma_squared(t)
    assert torch.all(torch.isclose(s, torch.tensor([0.60366678, 0.92093617, 0.99288464]))), "error in sigma_squared-function"
    torch.manual_seed(0)
    x = torch.rand((5, 1))
    noise = torch.randn((5, 1))
    t = torch.rand((5, 1))
    p = produce_noisy_samples_continuous(x, t, noise)
    assert torch.all(torch.isclose(p, torch.tensor([[0.57005614], [0.80221176], [0.54304826], [0.26050627], [-0.33080664]]))), "error in produce_noisy_samples_continuous"
    print("\033[92m\033[1m✓ Value tests passed! \033[0m")


def test_euler_maruyama(em):
    torch.manual_seed(0)
    x_t = torch.rand((5, 1))
    sigma_squared_t = torch.rand((5, 1))
    pred_noise = torch.randn((5, 1))
    beta_t = torch.rand((5, 1))
    delta_t = 0.01
    v = em(x_t, pred_noise, sigma_squared_t, beta_t, delta_t)
    assert torch.all(torch.isclose(v, torch.tensor([[0.563661098], [0.711565137], [0.277760923], [0.081645116], [0.327678204]])))
    print("\033[92m\033[1m✓ Value test passed! \033[0m")

