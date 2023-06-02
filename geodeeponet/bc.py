import abc
import torch

class BoundaryCondition(abc.ABC):
    """Abstract base class for boundary conditions."""

    @abc.abstractmethod
    def on_boundary(self, x):
        """Returns if a local coordinate 'x' is on a boundary.

        Args:
            x (torch.Tensor): The local point to check.

        Returns:
            bool: True if the point is on a boundary, False otherwise.

        """
    
    @abc.abstractmethod
    def is_dirichlet(self, x):
        """Returns if a local coordinate 'x' is on a Dirichlet boundary.

        Args:
            x (torch.Tensor): The local point to check.

        Returns:
            bool: True if the point is on a Dirichlet boundary, False otherwise.

        """

    @abc.abstractmethod
    def is_neumann(self, x):
        """Returns if a local coordinate 'x' is on a Neumann boundary.

        Args:
            x (torch.Tensor): The local point to check.

        Returns:
            bool: True if the point is on a Neumann boundary, False otherwise.

        """

    @abc.abstractmethod
    def value(self, x):
        """Returns the value of the Dirichlet boundary condition at a local coordinate 'x'.

        Args:
            x (torch.Tensor): The local point to evaluate the boundary condition at.

        Returns:
            torch.Tensor: The value of the boundary condition at the point 'x'.

        """

    @abc.abstractmethod
    def normal(self, x):
        """Returns the local normal vector of the Neumann boundary condition at a local coordinate 'x'.

        Args:
            x (torch.Tensor): The local point to evaluate the boundary condition at.

        Returns:
            torch.Tensor: The local normal vector of the boundary condition at the point 'x'.

        """

    def neumann(self, x):
        """Returns the value of the Neumann boundary condition at a local coordinate 'x'.

        Args:
            x (torch.Tensor): The local point to evaluate the boundary condition at.

        Returns:
            torch.Tensor: The value of the Neumann boundary condition at the point 'x'.

        """
        return torch.tensor(0)


class UnitCubeDirichletBC(BoundaryCondition):
    """Class to represent Dirichlet boundary condition for a unit cube.

    Attributes:
        value_dict (dict): Dictionary containing boundary condition values.

    Methods:
        is_dirichlet(x):
            Checks if a given point 'x' is on a Dirichlet boundary.
        value(x):
            Returns the value of the Dirichlet boundary condition at a given point 'x'.

    """

    def __init__(self, value_dict={}, eps=1e-8, neumann=None):
        """Initializes the UnitCubeDirichletBC class with a value dictionary.

        Args:
            value_dict (dict, optional): A dictionary that contains the boundary condition values.
                The keys should be one of "left", "right", "bottom", "top", "front", "back", and 
                map to float or callable.
                Defaults to {}.
            neumann (callable, optional): A function that returns the value of the Neumann boundary
                condition at a given point 'x'. Defaults to zero.
            eps (float, optional): A small number to check if a point is on a boundary. Defaults to 1e-8.
            
        Raises:
            AssertionError: If the keys in `value_dict` are not one of "left", "right", "bottom", "top".

        """
        assert all(x in ["left", "right", "bottom", "top", "front", "back"] for x in value_dict.keys())
        self.value_dict = value_dict
        self.eps = eps

        if neumann is not None:
            self.neumann = neumann
        else:
            self.neumann = lambda x: torch.tensor(0)

    def _where(self, x):
        """Determines which boundary of the unit cube the given point 'x' belongs to.

        Args:
            x (array_like): A point in the unit cube.

        Returns:
            str: The name of the boundary that the point 'x' belongs to.

        """
        if abs(x[0] - 0.0) < self.eps:
            return "left"
        if abs(x[0] - 1.0) < self.eps:
            return "right"
        
        if len(x) >= 2:
            if abs(x[1] - 0.0) < self.eps:
                return "bottom"
            if abs(x[1] - 1.0) < self.eps:
                return "top"

        if len(x) >= 3:
            if abs(x[2] - 0.0) < self.eps:
                return "front"
            if abs(x[2] - 1.0) < self.eps:
                return "back"

        return None
    
    def on_boundary(self, x):
        """Checks if a given point 'x' is on a boundary of the unit cube.

        Args:
            x (array_like): A point in the unit cube.

        Returns:
            bool: True if the point 'x' is on a boundary, False otherwise.

        """
        return self._where(x) is not None

    def is_dirichlet(self, x):
        """Checks if a given point 'x' is on a Dirichlet boundary.

        Args:
            x (array_like): A point on the boundary of the unit cube.

        Returns:
            bool: True if the point 'x' is on a Dirichlet boundary, False otherwise.

        """
        return self._where(x) in self.value_dict
    
    def is_neumann(self, x):
        """Checks if a given point 'x' is on a Neumann boundary.

        Args:
            x (array_like): A point on the boundary of the unit cube.

        Returns:
            bool: True if the point 'x' is on a Neumann boundary, False otherwise.

        """
        return self.on_boundary(x) and not self.is_dirichlet(x)

    def value(self, x):
        """Returns the value of the Dirichlet boundary condition at a given point 'x'.

        Args:
            x (array_like): A point on the boundary of the unit cube.

        Returns:
            float: The value of the Dirichlet boundary condition at the point 'x'.

        """
        v = self.value_dict[self._where(x)]
        if callable(v):
            return v(x)
        else:
            return v
    
    def normal(self, x):
        """Returns the normal of the boundary at a given point 'x'.

        Args:
            x (array_like): A point on the boundary of the unit cube.

        Returns:
            float: The normal of the boundary condition at the point 'x'.

        """
        n = torch.zeros_like(x)

        w = self._where(x)
        if w == "left":
            n[0] = -1.
        if w == "right":
            n[0] = 1.

        if len(x) >= 2:
            if w == "bottom":
                n[1] = -1.
            if w == "top":
                n[1] = 1.

        if len(x) >= 3:
            if w == "front":
                n[2] = -1.
            if w == "back":
                n[2] = 1.
        
        return torch.as_tensor(n)
    
    def neumann(self, x):
        """Returns the value of the Neumann boundary condition at a local coordinate 'x'.

        Args:
            x (torch.Tensor): The local point to evaluate the boundary condition at.

        Returns:
            torch.Tensor: The value of the Neumann boundary condition at the point 'x'.

        """
        return self.neumann(x)