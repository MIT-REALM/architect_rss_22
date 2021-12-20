from .design_parameters import DesignParameters
from .exogenous_parameters import ExogenousParameters
from .types import Simulator, CostFunction


class DesignProblem(object):
    """
    A DesignProblem includes a set of DesignParameters (which include constraints),
    a set of ExogenousParameters (which include a distribution), a simulator, and
    a cost function.

    This class should be generally applicable to a range of design problems; most of the
    problem-specific adaptation should be in subclassing DesignParameters and
    ExogenenousParameters.
    """

    def __init__(
        self,
        design_params: DesignParameters,
        exogenous_params: ExogenousParameters,
        simulator: Simulator,
        cost_fn: CostFunction,
    ):
        """
        Initialize a DesignProblem.

        args:
            design_params: the DesignParameters governing this problem.
            exogenous_params: the ExogenousParameters affecting this system.
            simulator: the simulator function for this design.
            cost_fn: the cost function assigning a performance metric to the output of
                     the simulator.
        """
        super(DesignProblem, self).__init__()
        self.design_params = design_params
        self.exogenous_params = exogenous_params
        self.simulator = simulator
        self.cost_fn = cost_fn
