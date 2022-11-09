from .design_parameters import DesignParameters
from .exogenous_parameters import ExogenousParameters
from .types import CostFunction, Simulator


class DesignProblem(object):
    """
    A DesignProblem includes a set of DesignParameters (which include constraints),
    a set of ExogenousParameters (which include a distribution), and
    a cost function that wraps a simulator

    This class should be generally applicable to a range of design problems; most of the
    problem-specific adaptation should be in subclassing DesignParameters and
    ExogenenousParameters.
    """

    def __init__(
        self,
        design_params: DesignParameters,
        exogenous_params: ExogenousParameters,
        cost_fn: CostFunction,
        simulator: Simulator,
    ):
        """
        Initialize a DesignProblem.

        args:
            design_params: the DesignParameters governing this problem.
            exogenous_params: the ExogenousParameters affecting this system.
            simulator: the simulator function for this design.
            cost_fn: the cost function assigning a performance metric to a specific
                     choice of design and exogenous parameters. Should be composed with
                     the simulator before being passed here.
            simulator: the simulator to use e.g. for plotting results
        """
        super(DesignProblem, self).__init__()
        self.design_params = design_params
        self.exogenous_params = exogenous_params
        self.cost_fn = cost_fn
        self.simulator = simulator
