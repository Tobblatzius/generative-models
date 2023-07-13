import torch

# include hint about the following pattern
# if len(log_prob.shape) > 1:
#     log_prob = log_prob.sum(list(range(1, len(log_prob.shape))))


SUM_LOG_PROBS_MSG = (
    "Log prob should have shape (1,), but is of shape {}. You probably need "
    "to sum the log probabilities of the flows over the non-batch dimensions. "
    "\nFor example, you can do: \n\tif len(log_prob.shape) > 1:\n"
    "\t\tlog_prob = log_prob.sum(list(range(1, len(log_prob.shape))))"
)

LDJ_MSG = "Log determinant shape should be ({},), but is {}"


def test_flow(flow, testdata, extramsg=""):
    """Forward-Backward test."""
    z, ldj = flow.forward(testdata)
    bs = testdata.shape[0]
    assert ldj.shape == (bs,), LDJ_MSG.format(bs, ldj.shape)
    x = flow.inverse(z)
    assert torch.allclose(testdata, x)
    print(f"\033[92m\033[1m✓ Forward-Backward check passed {extramsg}\033[0m")


def test_normflow(flow_class):
    # Simplest possible unit test for NormalizingFlow class with a dummy flow that does nothing
    class DummyFlow(torch.nn.Module):
        def forward(self, x):
            return x + 1, torch.tensor(1.0)

        def inverse(self, z):
            return z - 1

    flows = [DummyFlow() for _ in range(4)]

    distributions = [
        torch.distributions.Normal(0.0, 1.0),
        torch.distributions.MultivariateNormal(torch.zeros(3), torch.eye(3)),
    ]
    for distribution in distributions:
        try:
            _test_with_distribution(flow_class, flows, distribution)
        except AssertionError as e:
            print(
                f"\033[91m\033[1m✗ Test failed for {type(distribution)}! \033[0m \n{e}"
            )
            return

    # Print in green color if all tests passed with a checkmark in the beginning
    print("\033[92m\033[1m✓ All tests passed! :) \033[0m")


def _test_with_distribution(flow_class, flows, distribution):
    normflow = flow_class(flows, distribution)
    x = torch.randn(10, 3)
    z, log_det = normflow(x)
    x_ = normflow.inverse(z)
    assert torch.allclose(x_, x, rtol=1e-4), "x -> z -> x is not consistent"
    assert torch.allclose(log_det, torch.tensor(4.0)), "Log determinant is wrong"

    # Test correct log_prob computation
    test_sample = torch.tensor([[0.0, 0.0, 0.0]])
    expected = torch.tensor([-22.75681495666504])
    received = normflow.log_prob(test_sample)
    assert expected.shape == received.shape, SUM_LOG_PROBS_MSG.format(received.shape)
    assert torch.allclose(received, expected), "Log prob is wrong value"

    # Test correct nll computation
    lll = normflow.nll(test_sample)
    assert torch.allclose(normflow.nll(test_sample), -expected[0]), "NLL is wrong value"


def test_splitflow(flow_class):
    test_input = torch.zeros(1, 2)
    testflow = flow_class()
    z, ldj = testflow(test_input)
    assert z == test_input[:, 0:1], "z should be the first half of the input"
    assert ldj == -0.918938517570, "ldj should be the log likelihood of the second half"
    print("\033[92m\033[1m✓ All tests passed! :) \033[0m")
