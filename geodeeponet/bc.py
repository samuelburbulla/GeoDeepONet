class UnitCubeDirichletBC:
    """Class to represent Dirichlet boundary condition for a unit cube.

    Attributes:
        value_dict (dict): Dictionary containing boundary condition values.

    Methods:
        is_dirichlet(x):
            Checks if a given point 'x' is on a Dirichlet boundary.
        value(x):
            Returns the value of the Dirichlet boundary condition at a given point 'x'.

    """

    def __init__(self, value_dict={}):
        """Initializes the UnitCubeDirichletBC class with a value dictionary.

        Args:
            value_dict (dict, optional): A dictionary that contains the boundary condition values.
                The keys should be one of "left", "right", "bottom", "top" and map to float or callable.
                Defaults to {}.

        Raises:
            AssertionError: If the keys in `value_dict` are not one of "left", "right", "bottom", "top".

        """
        assert all(x in ["left", "right", "bottom", "top"] for x in value_dict.keys())
        self.value_dict = value_dict

    @staticmethod
    def _where(x):
        """Determines which boundary of the unit cube the given point 'x' belongs to.

        Args:
            x (array_like): A point in the unit cube.

        Returns:
            str: The name of the boundary that the point 'x' belongs to.

        """
        if x[0] == 0:
            return "left"
        elif x[0] == 1:
            return "right"
        elif x[1] == 0:
            return "bottom"
        elif x[1] == 1:
            return "top"
        else:
            return None

    def is_dirichlet(self, x):
        """Checks if a given point 'x' is on a Dirichlet boundary.

        Args:
            x (array_like): A point in the unit cube.

        Returns:
            bool: True if the point 'x' is on a Dirichlet boundary, False otherwise.

        """
        return self._where(x) in self.value_dict

    def value(self, x):
        """Returns the value of the Dirichlet boundary condition at a given point 'x'.

        Args:
            x (array_like): A point in the unit cube.

        Returns:
            float: The value of the Dirichlet boundary condition at the point 'x'.

        """
        return self.value_dict[self._where(x)]