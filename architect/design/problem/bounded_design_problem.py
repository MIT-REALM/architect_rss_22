from .bounded_design_parameters import BoundedDesignParameters
from .design_problem import DesignProblem
from .bounded_exogenous_parameters import BoundedExogenousParameters
from .types import CostFunction, Simulator


class BoundedDesignProblem(DesignProblem):
    """
    A DesignProblem where the exogenous parameters must be bounded
    """

    def __init__(
        self,
        design_params: BoundedDesignParameters,
        exogenous_params: BoundedExogenousParameters,
        cost_fn: CostFunction,
        simulator: Simulator,
    ):
        """
        Initialize a DesignProblem.

        args:
            design_params: the BoundedDesignParameters governing this problem.
            exogenous_params: the BoundedExogenousParameters affecting this system.
            simulator: the simulator function for this design.
            cost_fn: the cost function assigning a performance metric to a specific
                     choice of design and exogenous parameters. Should be composed with
                     the simulator before being passed here.
            simulator: the simulator to use e.g. for plotting results
        """
        self.design_params: BoundedDesignParameters
        self.exogenous_params: BoundedExogenousParameters
        super(BoundedDesignProblem, self).__init__(
            design_params, exogenous_params, cost_fn, simulator
        )
