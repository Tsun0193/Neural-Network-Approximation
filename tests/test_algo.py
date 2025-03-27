from utils.algo import optimal_approx

def test_optimal_approx_structure():
    f = lambda x: x ** 2
    intervals, errors, rounds = optimal_approx(n=3, f=f, a=0, b=1, stepsize=0.01)
    assert isinstance(intervals, list)
    assert isinstance(errors, list)
    assert isinstance(rounds, int)
    assert len(intervals) == len(errors)